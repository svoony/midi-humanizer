"""
Run the trained model on a flat/normalized MIDI window and render the
predicted expressive performance back out as a playable .mid file.

All input features (melodic intervals, chord/voice context, local density,
metric position, tempo, piece position) are computed from the flat input
alone, matching real deployment - only era is looked up from MAESTRO's
composer metadata, since these renders are demo pieces from the dataset
rather than a genuinely novel user-supplied score.

Writes three files for comparison:
  input_flat.mid     - the quantized/flat input as given to the model
  model_rendered.mid - the model's predicted performance (now including
                        a predicted sustain pedal curve)
  ground_truth.mid   - the real human performance for the same window
                        (only available because this is a test-set piece
                        with a paired original.midi; a real deployment
                        wouldn't have this)
"""
import argparse
import math
import os

import numpy as np
import pretty_midi
import torch

from model import PerformanceRegressor
from normalize_midi import estimate_grid
from note_dataset import (
    DEFAULT_ERA, FLOAT_FIELDS, LONG_FIELDS, build_era_map,
    compute_input_features, _compute_chord_features, _compute_local_density,
    _compute_melodic_intervals,
)

# safety clamps on predictions so an under-trained or noisy model can't
# produce degenerate MIDI (zero/negative durations, extreme timing jumps)
MAX_TIMING_OFFSET_GRID_UNITS = 2.0
MIN_LOG_RATIO, MAX_LOG_RATIO = np.log(0.1), np.log(10.0)
SUSTAIN_CC = 64


def load_piece(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = sorted(pm.instruments[0].notes, key=lambda n: n.start)
    pitches = np.array([n.pitch for n in notes], dtype=np.int64)
    starts = np.array([n.start for n in notes], dtype=np.float64)
    ends = np.array([n.end for n in notes], dtype=np.float64)
    return pm, pitches, starts, ends


def notes_to_midi(pitches, starts, ends, velocities, out_path, control_changes=None):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    for p, s, e, v in zip(pitches, starts, ends, velocities):
        e = max(e, s + 0.01)
        inst.notes.append(pretty_midi.Note(velocity=int(v), pitch=int(p), start=float(s), end=float(e)))
    for cc in (control_changes or []):
        inst.control_changes.append(cc)
    pm.instruments.append(inst)
    pm.write(out_path)


def load_sustain_pedal(midi_path, t_lo, t_hi, pad=0.1):
    pm = pretty_midi.PrettyMIDI(midi_path)
    ccs = pm.instruments[0].control_changes
    return [cc for cc in ccs if cc.number == SUSTAIN_CC and t_lo - pad <= cc.time <= t_hi + pad]


def pedal_predictions_to_ccs(starts, pedal_values):
    """One CC64 event per note onset, sampling the model's predicted pedal
    curve - the inverse of how the training target was extracted."""
    return [
        pretty_midi.ControlChange(number=SUSTAIN_CC, value=int(np.clip(round(v * 127), 0, 127)), time=float(t))
        for t, v in zip(starts, pedal_values)
    ]


def render(piece_dir, start=0, window_notes=192, checkpoint_path="checkpoints/best.pt",
           out_dir="renders", root="paired_data", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    flat_path = os.path.join(piece_dir, "normalized.midi")
    flat_pm, full_pitches, full_starts, full_ends = load_piece(flat_path)
    grid_unit, phase = estimate_grid(flat_pm)
    if flat_pm.time_signature_changes:
        beats_per_measure = flat_pm.time_signature_changes[0].numerator
    else:
        beats_per_measure = 4
    piece_duration = float(full_ends.max()) if len(full_ends) else 1.0

    full_chord_size, full_voice_role = _compute_chord_features(full_starts, full_pitches)
    full_interval_prev, full_interval_next = _compute_melodic_intervals(full_pitches)
    full_local_density = _compute_local_density(full_starts, grid_unit)

    era_map = build_era_map(root)
    era_id = era_map.get(piece_dir, DEFAULT_ERA)

    end = start + window_notes
    pitches = full_pitches[start:end]
    starts = full_starts[start:end]
    ends = full_ends[start:end]
    n = len(pitches)

    feats = compute_input_features(pitches, starts, ends, grid_unit, phase, beats_per_measure)

    field_arrays = {
        "pitch": feats["pitches"],
        "rel_step": feats["rel_steps"].astype(np.float32),
        "beat_pos": feats["beat_pos"],
        "bar_pos": feats["bar_pos"],
        "beat_in_measure": feats["beat_in_measure"],
        "measure_number": feats["measure_number"],
        "ioi": feats["ioi"].astype(np.float32),
        "dur_grid": feats["dur_grid"].astype(np.float32),
        "interval_prev": full_interval_prev[start:end].astype(np.float32),
        "interval_next": full_interval_next[start:end].astype(np.float32),
        "chord_size": full_chord_size[start:end].astype(np.float32),
        "voice_role": full_voice_role[start:end].astype(np.int64),
        "local_density": full_local_density[start:end].astype(np.float32),
        "tempo_scalar": np.full(n, math.log(grid_unit), dtype=np.float32),
        "piece_position": (starts / piece_duration).astype(np.float32),
        "era": np.full(n, era_id, dtype=np.int64),
    }

    x = {}
    for f in LONG_FIELDS:
        x[f] = torch.from_numpy(field_arrays[f]).unsqueeze(0).to(device)
    for f in FLOAT_FIELDS:
        x[f] = torch.from_numpy(field_arrays[f]).unsqueeze(0).to(device)
    x["pad_mask"] = torch.zeros(1, n, dtype=torch.bool).to(device)

    model = PerformanceRegressor().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        pred = model(x)[0].cpu().numpy()  # [N, 4]

    timing_offset = np.clip(pred[:, 0], -MAX_TIMING_OFFSET_GRID_UNITS * grid_unit,
                              MAX_TIMING_OFFSET_GRID_UNITS * grid_unit)
    log_ratio = np.clip(pred[:, 1], MIN_LOG_RATIO, MAX_LOG_RATIO)
    velocity = np.clip(pred[:, 2], 0.0, 1.0)
    pedal = np.clip(pred[:, 3], 0.0, 1.0)

    q_starts, q_ends = feats["q_starts"], feats["q_ends"]
    pred_starts = q_starts + timing_offset
    pred_ends = pred_starts + np.exp(log_ratio) * (q_ends - q_starts)
    pred_velocities = np.clip(np.round(velocity * 127), 1, 127)
    pred_pedal_ccs = pedal_predictions_to_ccs(pred_starts, pedal)

    piece_name = os.path.basename(piece_dir)
    run_dir = os.path.join(out_dir, f"{piece_name}_w{start}")
    os.makedirs(run_dir, exist_ok=True)

    notes_to_midi(pitches, starts, ends, [80] * len(pitches),
                  os.path.join(run_dir, "input_flat.mid"))
    notes_to_midi(pitches, pred_starts, pred_ends, pred_velocities,
                  os.path.join(run_dir, "model_rendered.mid"), control_changes=pred_pedal_ccs)

    gt_path = os.path.join(piece_dir, "original.midi")
    if os.path.exists(gt_path):
        _, gt_full_pitches, gt_full_starts, gt_full_ends = load_piece(gt_path)
        gt_pitches = gt_full_pitches[start:end]
        gt_starts = gt_full_starts[start:end]
        gt_ends = gt_full_ends[start:end]
        gt_pm = pretty_midi.PrettyMIDI(gt_path)
        gt_notes = sorted(gt_pm.instruments[0].notes, key=lambda n: n.start)[start:end]
        gt_vels = [note.velocity for note in gt_notes]
        gt_pedal = load_sustain_pedal(gt_path, gt_starts.min(), gt_ends.max())
        notes_to_midi(gt_pitches, gt_starts, gt_ends, gt_vels,
                      os.path.join(run_dir, "ground_truth.mid"), control_changes=gt_pedal)

    print(f"wrote {run_dir}/{{input_flat, model_rendered, ground_truth}}.mid")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("piece_dir")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--window_notes", type=int, default=192)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()
    render(args.piece_dir, args.start, args.window_notes, args.checkpoint)
