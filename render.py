"""
Run the trained model on a flat/normalized MIDI window and render the
predicted expressive performance back out as a playable .mid file.

Writes three files for comparison:
  input_flat.mid     - the quantized/flat input as given to the model
  model_rendered.mid - the model's predicted performance
  ground_truth.mid   - the real human performance for the same window
                        (only available because this is a test-set piece
                        with a paired original.midi; a real deployment
                        wouldn't have this)
"""
import argparse
import os

import numpy as np
import pretty_midi
import torch

from model import PerformanceRegressor
from normalize_midi import estimate_grid
from note_dataset import PITCH_MIN, compute_input_features

# safety clamps on predictions so an under-trained or noisy model can't
# produce degenerate MIDI (zero/negative durations, extreme timing jumps)
MAX_TIMING_OFFSET_GRID_UNITS = 2.0
MIN_LOG_RATIO, MAX_LOG_RATIO = np.log(0.1), np.log(10.0)


def load_window(midi_path, start, window_notes):
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = sorted(pm.instruments[0].notes, key=lambda n: n.start)
    window = notes[start:start + window_notes]
    pitches = np.array([n.pitch for n in window], dtype=np.int64)
    starts = np.array([n.start for n in window], dtype=np.float64)
    ends = np.array([n.end for n in window], dtype=np.float64)
    return pitches, starts, ends, window


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


SUSTAIN_CC = 64


def load_sustain_pedal(midi_path, t_lo, t_hi, pad=0.1):
    pm = pretty_midi.PrettyMIDI(midi_path)
    ccs = pm.instruments[0].control_changes
    return [cc for cc in ccs if cc.number == SUSTAIN_CC and t_lo - pad <= cc.time <= t_hi + pad]


def render(piece_dir, start=0, window_notes=192, checkpoint_path="checkpoints/best.pt",
           out_dir="renders", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    flat_path = os.path.join(piece_dir, "normalized.midi")
    pitches, starts, ends, flat_notes = load_window(flat_path, start, window_notes)
    flat_pm = pretty_midi.PrettyMIDI(flat_path)
    grid_unit, phase = estimate_grid(flat_pm)

    feats = compute_input_features(pitches, starts, ends, grid_unit, phase)

    x = {
        "pitch": torch.from_numpy(feats["pitches"]).unsqueeze(0).to(device),
        "rel_step": torch.from_numpy(feats["rel_steps"].astype(np.float32)).unsqueeze(0).to(device),
        "beat_pos": torch.from_numpy(feats["beat_pos"]).unsqueeze(0).to(device),
        "bar_pos": torch.from_numpy(feats["bar_pos"]).unsqueeze(0).to(device),
        "ioi": torch.from_numpy(feats["ioi"].astype(np.float32)).unsqueeze(0).to(device),
        "dur_grid": torch.from_numpy(feats["dur_grid"].astype(np.float32)).unsqueeze(0).to(device),
        "pad_mask": torch.zeros(1, len(pitches), dtype=torch.bool).to(device),
    }

    model = PerformanceRegressor().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        pred = model(x)[0].cpu().numpy()  # [N, 3]

    timing_offset = np.clip(pred[:, 0], -MAX_TIMING_OFFSET_GRID_UNITS * grid_unit,
                              MAX_TIMING_OFFSET_GRID_UNITS * grid_unit)
    log_ratio = np.clip(pred[:, 1], MIN_LOG_RATIO, MAX_LOG_RATIO)
    velocity = np.clip(pred[:, 2], 0.0, 1.0)

    q_starts, q_ends = feats["q_starts"], feats["q_ends"]
    pred_starts = q_starts + timing_offset
    pred_ends = pred_starts + np.exp(log_ratio) * (q_ends - q_starts)
    pred_velocities = np.clip(np.round(velocity * 127), 1, 127)

    piece_name = os.path.basename(piece_dir)
    run_dir = os.path.join(out_dir, f"{piece_name}_w{start}")
    os.makedirs(run_dir, exist_ok=True)

    notes_to_midi(pitches, starts, ends, [80] * len(pitches),
                  os.path.join(run_dir, "input_flat.mid"))
    notes_to_midi(pitches, pred_starts, pred_ends, pred_velocities,
                  os.path.join(run_dir, "model_rendered.mid"))

    gt_path = os.path.join(piece_dir, "original.midi")
    if os.path.exists(gt_path):
        gt_pitches, gt_starts, gt_ends, _ = load_window(gt_path, start, window_notes)
        gt_pm = pretty_midi.PrettyMIDI(gt_path)
        gt_notes = sorted(gt_pm.instruments[0].notes, key=lambda n: n.start)[start:start + window_notes]
        gt_vels = [n.velocity for n in gt_notes]
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
