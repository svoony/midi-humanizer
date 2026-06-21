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

from model import PerformanceRegressor
from normalize_midi import estimate_grid, quantize_time
from note_dataset import (
    ERA_NAMES, FLOAT_FIELDS, LONG_FIELDS, SUBDIVISIONS_PER_BEAT,
    _compute_chord_features, _compute_local_density, _compute_melodic_intervals,
    compute_input_features,
)

MUSICXML_EXTENSIONS = (".musicxml", ".xml", ".mxl")
DEFAULT_TEMPO_BPM = 120.0

WINDOW_NOTES = 192  # matches the training window size
MAX_TIMING_OFFSET_GRID_UNITS = 2.0
MIN_LOG_RATIO, MAX_LOG_RATIO = np.log(0.1), np.log(10.0)
SUSTAIN_CC = 64

_model_cache = {}


def _load_model(checkpoint_path, device):
    key = (checkpoint_path, device)
    if key not in _model_cache:
        model = PerformanceRegressor().to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        _model_cache[key] = model
    return _model_cache[key]


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


def predict_raw(input_path, era, checkpoint_path="checkpoints/best.pt", device=None):
    """Returns a dict of per-note arrays: pitches, q_starts, q_ends,
    timing_offset, log_ratio, velocity (0-1), pedal (0-1) - the model's raw,
    unadjusted predictions."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    era_id = ERA_NAMES.index(era) if isinstance(era, str) else era

    pitches, starts, ends, grid_unit, phase, beats_per_measure = load_input(input_path)
    piece_duration = float(ends.max())

    chord_size, voice_role = _compute_chord_features(starts, pitches)
    interval_prev, interval_next = _compute_melodic_intervals(pitches)
    local_density = _compute_local_density(starts, grid_unit)

    model = _load_model(checkpoint_path, device)

    n_total = len(pitches)
    out_q_starts, out_q_ends, out_timing, out_logratio, out_vel, out_pedal = [], [], [], [], [], []

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

        x = {}
        for f in LONG_FIELDS:
            x[f] = torch.from_numpy(field_arrays[f]).unsqueeze(0).to(device)
        for f in FLOAT_FIELDS:
            x[f] = torch.from_numpy(field_arrays[f]).unsqueeze(0).to(device)
        x["pad_mask"] = torch.zeros(1, n, dtype=torch.bool).to(device)

        with torch.no_grad():
            pred = model(x)[0].cpu().numpy()

        out_q_starts.append(feats["q_starts"])
        out_q_ends.append(feats["q_ends"])
        out_timing.append(pred[:, 0])
        out_logratio.append(pred[:, 1])
        out_vel.append(np.clip(pred[:, 2], 0.0, 1.0))
        out_pedal.append(np.clip(pred[:, 3], 0.0, 1.0))

    return {
        "pitches": pitches,
        "q_starts": np.concatenate(out_q_starts),
        "q_ends": np.concatenate(out_q_ends),
        "timing_offset": np.concatenate(out_timing),
        "log_ratio": np.concatenate(out_logratio),
        "velocity": np.concatenate(out_vel),
        "pedal": np.concatenate(out_pedal),
        "grid_unit": grid_unit,
        "era": ERA_NAMES[era_id],
    }


def apply_adjustments(raw, params):
    """Cheap post-processing: scales the raw predictions by the user-
    controllable knobs and renders the final MIDI. No model forward pass."""
    grid_unit = raw["grid_unit"]

    timing_offset = np.clip(
        raw["timing_offset"] * params.get("rubato_intensity", 1.0),
        -MAX_TIMING_OFFSET_GRID_UNITS * grid_unit, MAX_TIMING_OFFSET_GRID_UNITS * grid_unit,
    )
    log_ratio = np.clip(
        raw["log_ratio"] * params.get("articulation_intensity", 1.0),
        MIN_LOG_RATIO, MAX_LOG_RATIO,
    )

    vel = raw["velocity"]
    dynamics_intensity = params.get("dynamics_intensity", 1.0)
    vel_adjusted = np.clip(vel.mean() + (vel - vel.mean()) * dynamics_intensity, 0.0, 1.0)

    pedal_adjusted = np.clip(
        raw["pedal"] * params.get("pedal_scale", 1.0) + params.get("pedal_boost", 0.0),
        0.0, 1.0,
    )

    tempo_multiplier = max(params.get("tempo_multiplier", 1.0), 0.1)

    pred_starts = (raw["q_starts"] + timing_offset) / tempo_multiplier
    pred_ends = pred_starts + np.exp(log_ratio) * (raw["q_ends"] - raw["q_starts"]) / tempo_multiplier
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
