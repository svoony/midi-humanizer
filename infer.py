"""
Full-piece inference, split into two phases so interactive adjustments
(rubato, dynamics, pedal, articulation, tempo) don't require re-running the
model:

  predict_raw()      - expensive: normalizes the input (quantizes timing,
                        ignores velocity/pedal - same as normalize_midi.py),
                        computes context features, and runs the model over
                        the whole piece in chunks. Depends on era, since
                        era is a genuine model input.
  apply_adjustments() - cheap: post-processes the raw per-note predictions
                        with user-controllable scalars and renders the
                        final MIDI. No model forward pass involved, so chat
                        commands that only touch these knobs are instant.

render() runs both phases for a one-shot call (e.g. from render.py-style
scripts); the Flask app calls them separately to cache the raw predictions
across chat turns.
"""
import math
import os

import music21
import numpy as np
import pretty_midi
import torch

from model import PerformanceNet
from normalize_midi import estimate_grid, quantize_time
from note_dataset import (
    ERA_NAMES, FLOAT_FIELDS, LONG_FIELDS, N_STYLE, SCORE_FEATURE_FIELDS,
    SUBDIVISIONS_PER_BEAT, _compute_chord_features, _compute_local_density,
    _compute_melodic_intervals, compute_input_features,
)
from score_features import compute_score_features

MUSICXML_EXTENSIONS = (".musicxml", ".xml", ".mxl")
DEFAULT_TEMPO_BPM = 120.0

WINDOW_NOTES = 192  # matches the training window size
MIN_LOG_RATIO, MAX_LOG_RATIO = np.log(0.1), np.log(10.0)
ONSET_LEAK = 0.05  # leaky-anchor strength when integrating local tempo -> onsets
SUSTAIN_CC = 64

# scale factors mapping the 0..~2 user knobs to musically reasonable amounts
# (velocity values are in the model's 0-1 space; *127 happens at the end)
MELODY_EMPHASIS_TOP_GAIN = 0.20      # added to the top voice at emphasis=1
MELODY_EMPHASIS_ACCOMP_CUT = 0.12    # removed from inner/bottom voices at emphasis=1
METRIC_ACCENT_GAIN = 0.15            # added on a downbeat at accent=1
METRIC_ACCENT_WEAK_BEAT = 0.45       # accent weight on non-downbeat beats, relative to a downbeat
CHORD_ROLL_SECONDS_PER_NOTE = 0.018  # onset stagger per chord note (bottom->top) at roll=1

_model_cache = {}


def _load_model(checkpoint_path, device):
    """Loads PerformanceNet plus the target + style standardization stats (the
    model predicts in z-scored space and is conditioned on a z-scored style
    vector)."""
    key = (checkpoint_path, device)
    if key not in _model_cache:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = PerformanceNet(**ckpt.get("config", {})).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        stats = {
            "target_mean": np.asarray(ckpt["target_mean"], dtype=np.float32),
            "target_std": np.asarray(ckpt["target_std"], dtype=np.float32),
            "style_mean": np.asarray(ckpt["style_mean"], dtype=np.float32),
            "style_std": np.asarray(ckpt["style_std"], dtype=np.float32),
        }
        _model_cache[key] = (model, stats)
    return _model_cache[key]


def reconstruct_onsets(q_starts, log_ioi_ratio, leak=ONSET_LEAK):
    """Turn per-note local-tempo deviations into absolute onsets by leaky
    integration anchored to the constant-tempo grid. Pure cumulative integration
    would drift (clipped pauses, prediction error); the gentle pull toward
    q_starts keeps global timing sane while preserving phrase-level rubato.
    Notes sharing a quantized onset (a chord) share the reconstructed onset."""
    n = len(q_starts)
    if n == 0:
        return np.zeros(0)
    grp = np.zeros(n, dtype=np.int64)
    grp[1:] = np.cumsum(np.diff(q_starts) > 1e-6)
    n_groups = int(grp[-1]) + 1
    score_on = np.zeros(n_groups)
    score_on[grp] = q_starts
    dev = np.zeros(n_groups)
    cnt = np.zeros(n_groups)
    np.add.at(dev, grp, log_ioi_ratio)
    np.add.at(cnt, grp, 1.0)
    dev /= np.maximum(cnt, 1.0)

    recon = np.zeros(n_groups)
    recon[0] = score_on[0]
    for g in range(1, n_groups):
        score_ioi = score_on[g] - score_on[g - 1]
        free = recon[g - 1] + score_ioi * math.exp(dev[g])
        recon[g] = (1.0 - leak) * free + leak * score_on[g]
    return recon[grp]


CONSTANT_VELOCITY = 80


def load_and_quantize(input_path):
    """Parse the uploaded MIDI and quantize it to its estimated grid - the
    same normalization the model's input represents, and exactly what the
    'flat' version of the piece sounds like before any expression is added."""
    pm = pretty_midi.PrettyMIDI(input_path)
    notes = sorted(
        (n for inst in pm.instruments for n in inst.notes if not inst.is_drum),
        key=lambda n: n.start,
    )
    if not notes:
        raise ValueError("no notes found in the uploaded MIDI")

    pitches = np.array([n.pitch for n in notes], dtype=np.int64)
    raw_starts = np.array([n.start for n in notes], dtype=np.float64)
    raw_ends = np.array([n.end for n in notes], dtype=np.float64)

    grid_unit, phase = estimate_grid(pm)
    starts = np.array([quantize_time(t, grid_unit, phase) for t in raw_starts])
    ends = np.array([quantize_time(t, grid_unit, phase) for t in raw_ends])
    ends = np.maximum(ends, starts + grid_unit)

    if pm.time_signature_changes:
        beats_per_measure = pm.time_signature_changes[0].numerator
    else:
        beats_per_measure = 4

    return pitches, starts, ends, grid_unit, phase, beats_per_measure


def load_musicxml(input_path):
    """Parse a MusicXML score directly - exact pitch/duration/measure data,
    no grid estimation needed, unlike an arbitrary downloaded MIDI which may
    be mistranscribed or carry no reliable tempo/meter metadata. Tempo
    defaults to 120bpm if the score has no metronome marking; the chat
    'play it faster/slower' adjustment can correct this after the fact."""
    score = music21.converter.parse(input_path)
    flat = score.flatten()

    tempo_bpm = DEFAULT_TEMPO_BPM
    for mm in flat.getElementsByClass("MetronomeMark"):
        number = mm.number or mm.numberSounding
        if number:
            tempo_bpm = float(number)
            break

    beats_per_measure = 4
    for ts in flat.getElementsByClass("TimeSignature"):
        beats_per_measure = ts.numerator
        break

    raw_notes = []
    for n in flat.notes:
        start_q = float(n.offset)
        end_q = start_q + float(n.duration.quarterLength)
        if n.isChord:
            for p in n.pitches:
                raw_notes.append((start_q, end_q, p.midi))
        elif n.isNote:
            raw_notes.append((start_q, end_q, n.pitch.midi))
    if not raw_notes:
        raise ValueError("no notes found in the MusicXML score")
    raw_notes.sort(key=lambda x: x[0])

    seconds_per_quarter = 60.0 / tempo_bpm
    starts = np.array([s * seconds_per_quarter for s, e, p in raw_notes])
    ends = np.array([e * seconds_per_quarter for s, e, p in raw_notes])
    pitches = np.array([p for s, e, p in raw_notes], dtype=np.int64)

    grid_unit = seconds_per_quarter / SUBDIVISIONS_PER_BEAT
    ends = np.maximum(ends, starts + grid_unit)
    phase = 0.0

    return pitches, starts, ends, grid_unit, phase, beats_per_measure


def load_input(input_path):
    """Dispatches to the MusicXML or MIDI loader by file extension. Both
    return the same (pitches, starts, ends, grid_unit, phase,
    beats_per_measure) shape, so everything downstream is format-agnostic."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext in MUSICXML_EXTENSIONS:
        return load_musicxml(input_path)
    return load_and_quantize(input_path)


def normalize_to_flat_midi(input_path):
    """The model's input, rendered as a playable MIDI: quantized timing,
    constant velocity, no pedal."""
    pitches, starts, ends, *_ = load_input(input_path)
    out_pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    for p, s, e in zip(pitches, starts, ends):
        inst.notes.append(pretty_midi.Note(velocity=CONSTANT_VELOCITY, pitch=int(p), start=float(s), end=float(e)))
    out_pm.instruments.append(inst)
    return out_pm


def predict_raw(input_path, era, checkpoint_path="checkpoints/best.pt", device=None, style_z=None):
    """Returns a dict of per-note arrays: pitches, q_starts, q_ends,
    log_ioi_ratio (local-tempo deviation), log_ratio (duration), velocity
    (0-1), pedal (0-1) - the model's raw predictions.

    style_z is the standardized global style vector (shape (N_STYLE,)): 0 =
    neutral / dataset-average interpretation; raise rubato_magnitude or
    dynamic_range above 0 for a more expressive, committed render. None ->
    neutral."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    era_id = ERA_NAMES.index(era) if isinstance(era, str) else era

    pitches, starts, ends, grid_unit, phase, beats_per_measure = load_input(input_path)
    piece_duration = float(ends.max())

    chord_size, voice_role = _compute_chord_features(starts, pitches)
    interval_prev, interval_next = _compute_melodic_intervals(pitches)
    local_density = _compute_local_density(starts, grid_unit)

    # harmonic / tension / phrase features over the whole piece (same as the
    # training cache), then sliced per chunk below
    piece_feats = compute_input_features(pitches, starts, ends, grid_unit, phase, beats_per_measure)
    score_feats = compute_score_features(
        pitches, piece_feats["grid_steps"], piece_feats["end_grid_steps"],
        SUBDIVISIONS_PER_BEAT, beats_per_measure,
    )

    model, stats = _load_model(checkpoint_path, device)
    target_mean, target_std = stats["target_mean"], stats["target_std"]
    if style_z is None:
        style_z = np.zeros(N_STYLE, dtype=np.float32)
    style_t = torch.from_numpy(np.asarray(style_z, dtype=np.float32)).unsqueeze(0).to(device)

    n_total = len(pitches)
    out_q_starts, out_q_ends, out_timing, out_logratio, out_vel, out_pedal = [], [], [], [], [], []
    out_beat_pos, out_beat_in_measure = [], []

    for chunk_start in range(0, n_total, WINDOW_NOTES):
        chunk_end = min(chunk_start + WINDOW_NOTES, n_total)
        c_pitches = pitches[chunk_start:chunk_end]
        c_starts = starts[chunk_start:chunk_end]
        c_ends = ends[chunk_start:chunk_end]
        n = len(c_pitches)

        feats = compute_input_features(c_pitches, c_starts, c_ends, grid_unit, phase, beats_per_measure)

        field_arrays = {
            "pitch": feats["pitches"],
            "rel_step": feats["rel_steps"].astype(np.float32),
            "beat_pos": feats["beat_pos"],
            "bar_pos": feats["bar_pos"],
            "beat_in_measure": feats["beat_in_measure"],
            "measure_number": feats["measure_number"],
            "ioi": feats["ioi"].astype(np.float32),
            "dur_grid": feats["dur_grid"].astype(np.float32),
            "interval_prev": interval_prev[chunk_start:chunk_end].astype(np.float32),
            "interval_next": interval_next[chunk_start:chunk_end].astype(np.float32),
            "chord_size": chord_size[chunk_start:chunk_end].astype(np.float32),
            "voice_role": voice_role[chunk_start:chunk_end].astype(np.int64),
            "local_density": local_density[chunk_start:chunk_end].astype(np.float32),
            "tempo_scalar": np.full(n, math.log(grid_unit), dtype=np.float32),
            "piece_position": (c_starts / piece_duration).astype(np.float32),
            "era": np.full(n, era_id, dtype=np.int64),
        }
        # score-derived categoricals + continuous, sliced from the whole-piece pass
        for f in SCORE_FEATURE_FIELDS:
            field_arrays[f] = score_feats[f][chunk_start:chunk_end]

        x = {}
        for f in LONG_FIELDS:
            x[f] = torch.from_numpy(field_arrays[f].astype(np.int64)).unsqueeze(0).to(device)
        for f in FLOAT_FIELDS:
            x[f] = torch.from_numpy(field_arrays[f].astype(np.float32)).unsqueeze(0).to(device)
        x["pad_mask"] = torch.zeros(1, n, dtype=torch.bool).to(device)

        mean, _logvar = model.predict(x, style=style_t)
        # model predicts in standardized space; invert to real target units
        pred = mean[0].cpu().numpy() * target_std + target_mean

        out_q_starts.append(feats["q_starts"])
        out_q_ends.append(feats["q_ends"])
        out_timing.append(pred[:, 0])
        out_logratio.append(pred[:, 1])
        out_vel.append(np.clip(pred[:, 2], 0.0, 1.0))
        out_pedal.append(np.clip(pred[:, 3], 0.0, 1.0))
        out_beat_pos.append(feats["beat_pos"])
        out_beat_in_measure.append(feats["beat_in_measure"])

    return {
        "pitches": pitches,
        "q_starts": np.concatenate(out_q_starts),
        "q_ends": np.concatenate(out_q_ends),
        "log_ioi_ratio": np.concatenate(out_timing),
        "log_ratio": np.concatenate(out_logratio),
        "velocity": np.concatenate(out_vel),
        "pedal": np.concatenate(out_pedal),
        # score-derived per-note context, carried through for the post-processing
        # levers in apply_adjustments (melody emphasis, chord roll, metric accent).
        "voice_role": voice_role,        # 0 solo, 1 top, 2 bottom, 3 inner
        "chord_size": chord_size,
        "beat_pos": np.concatenate(out_beat_pos),                # subdivision within beat (0 = on the beat)
        "beat_in_measure": np.concatenate(out_beat_in_measure),  # which beat of the measure (0 = downbeat)
        "grid_unit": grid_unit,
        "era": ERA_NAMES[era_id],
    }


def apply_adjustments(raw, params):
    """Cheap post-processing: scales the raw predictions by the user-
    controllable knobs and renders the final MIDI. No model forward pass."""
    grid_unit = raw["grid_unit"]

    # rubato lever scales the predicted local-tempo deviation (in log space, so
    # it amplifies/suppresses how much the tempo ebbs and flows)
    log_ioi_ratio = np.clip(
        raw["log_ioi_ratio"] * params.get("rubato_intensity", 1.0), np.log(0.2), np.log(5.0),
    )
    log_ratio = np.clip(
        raw["log_ratio"] * params.get("articulation_intensity", 1.0),
        MIN_LOG_RATIO, MAX_LOG_RATIO,
    )

    vel = raw["velocity"]
    dynamics_intensity = params.get("dynamics_intensity", 1.0)
    vel_adjusted = vel.mean() + (vel - vel.mean()) * dynamics_intensity

    # melody emphasis: lift the top-of-chord voice and ease back the
    # inner/bottom accompaniment so the tune sits above the texture.
    melody_emphasis = params.get("melody_emphasis", 0.0)
    if melody_emphasis:
        voice_role = raw["voice_role"]
        is_top = voice_role == 1
        is_accomp = (voice_role == 2) | (voice_role == 3)
        vel_adjusted = vel_adjusted + melody_emphasis * MELODY_EMPHASIS_TOP_GAIN * is_top
        vel_adjusted = vel_adjusted - melody_emphasis * MELODY_EMPHASIS_ACCOMP_CUT * is_accomp

    # metric accent: stress beat onsets, downbeats most of all.
    metric_accent = params.get("metric_accent", 0.0)
    if metric_accent:
        on_beat = raw["beat_pos"] == 0
        downbeat = on_beat & (raw["beat_in_measure"] == 0)
        weight = np.where(downbeat, 1.0, np.where(on_beat, METRIC_ACCENT_WEAK_BEAT, 0.0))
        vel_adjusted = vel_adjusted + metric_accent * METRIC_ACCENT_GAIN * weight

    vel_adjusted = np.clip(vel_adjusted, 0.0, 1.0)

    pedal_adjusted = np.clip(
        raw["pedal"] * params.get("pedal_scale", 1.0) + params.get("pedal_boost", 0.0),
        0.0, 1.0,
    )

    tempo_multiplier = max(params.get("tempo_multiplier", 1.0), 0.1)

    # reconstruct onsets from the (lever-scaled) local-tempo curve, anchored to
    # the grid, then apply the global tempo multiplier
    pred_starts = reconstruct_onsets(raw["q_starts"], log_ioi_ratio) / tempo_multiplier
    score_dur = np.maximum(raw["q_ends"] - raw["q_starts"], 1e-4)
    pred_ends = pred_starts + np.exp(log_ratio) * score_dur / tempo_multiplier

    # chord roll: stagger simultaneous chord onsets bottom-to-top for an
    # arpeggiated, less mechanical attack. Notes sharing a quantized onset
    # form a chord (they're consecutive, since input is start-sorted); the
    # lowest pitch stays put and higher notes enter progressively later.
    chord_roll = params.get("chord_roll", 0.0)
    if chord_roll:
        q_starts = raw["q_starts"]
        chord_pitches = raw["pitches"]
        delays = np.zeros(len(q_starts))
        i, nq = 0, len(q_starts)
        while i < nq:
            j = i + 1
            while j < nq and abs(q_starts[j] - q_starts[i]) < 1e-6:
                j += 1
            if j - i > 1:
                grp = np.arange(i, j)
                ranks = np.argsort(np.argsort(chord_pitches[grp]))  # 0 = lowest pitch
                delays[grp] = chord_roll * CHORD_ROLL_SECONDS_PER_NOTE * ranks
            i = j
        pred_starts = pred_starts + delays
        pred_ends = pred_ends + delays

    pred_velocities = np.clip(np.round(vel_adjusted * 127), 1, 127)

    out_pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    for p, s, e, v in zip(raw["pitches"], pred_starts, pred_ends, pred_velocities):
        e = max(e, s + 0.01)
        inst.notes.append(pretty_midi.Note(velocity=int(v), pitch=int(p), start=float(s), end=float(e)))
    inst.control_changes.extend(sorted(
        (pretty_midi.ControlChange(number=SUSTAIN_CC, value=int(np.clip(round(v * 127), 0, 127)), time=float(t))
         for t, v in zip(pred_starts, pedal_adjusted)),
        key=lambda cc: cc.time,
    ))
    out_pm.instruments.append(inst)
    return out_pm


def render(input_path, era, params=None, checkpoint_path="checkpoints/best.pt", device=None):
    """One-shot convenience wrapper: predict + adjust in a single call."""
    raw = predict_raw(input_path, era, checkpoint_path, device)
    return apply_adjustments(raw, params or {})
