"""
Compute the standardization stats the gen-3 model needs, from the per-piece
caches (so run note_dataset.py prebuild first):

  paired_data/target_stats.json - per-target (mean, std) in TARGET_ORDER, where
      the timing target is now the local-tempo deviation log(perf_IOI/score_IOI).
  paired_data/style_stats.json  - per-descriptor (mean, std) in STYLE_ORDER for
      the global style conditioning vector.

Both are computed over the train split only.
"""
import glob
import json
import os

import numpy as np

from note_dataset import (
    DURATION_RATIO_EPS, STYLE_ORDER, STYLE_STATS_PATH, TARGET_ORDER,
    TARGET_STATS_PATH, VELOCITY_SCALE, _load_piece_arrays, compute_input_features,
    pedal_at_times,
)

ROOT = "paired_data/train"


def _stat(arr):
    return {"mean": float(np.mean(arr)), "std": float(max(np.std(arr), 1e-4))}


def main():
    dirs = sorted(glob.glob(os.path.join(ROOT, "*", "*")))
    print(f"computing stats over {len(dirs)} train pieces")

    tempo_dev, log_dur, vel, ped, styles = [], [], [], [], []
    for i, d in enumerate(dirs):
        p = _load_piece_arrays(d)
        if len(p["pitches"]) == 0:
            continue
        feats = compute_input_features(
            p["pitches"], p["starts"], p["ends"],
            float(p["grid_unit"]), float(p["phase"]), int(p["time_sig_numerator"]),
        )
        score_dur = np.maximum(feats["q_ends"] - feats["q_starts"], DURATION_RATIO_EPS)
        ldr = np.log(np.maximum((p["ends"] - p["starts"]) / score_dur, DURATION_RATIO_EPS))

        tempo_dev.append(p["tempo_dev"])
        log_dur.append(ldr)
        vel.append(p["velocities"].astype(np.float64) / VELOCITY_SCALE)
        ped.append(pedal_at_times(p["starts"], p["pedal_times"], p["pedal_values"]) / VELOCITY_SCALE)
        styles.append(p["style"])
        if i % 200 == 0:
            print(f"  {i}/{len(dirs)}")

    tempo_dev = np.concatenate(tempo_dev)
    log_dur = np.concatenate(log_dur)
    vel = np.concatenate(vel)
    ped = np.concatenate(ped)
    styles = np.stack(styles)  # (n_pieces, N_STYLE)

    per_target = [tempo_dev, log_dur, vel, ped]
    target_stats = {"order": TARGET_ORDER}
    for name, arr in zip(TARGET_ORDER, per_target):
        target_stats[name] = _stat(arr)
    with open(TARGET_STATS_PATH, "w") as f:
        json.dump(target_stats, f, indent=2)

    style_stats = {"order": STYLE_ORDER}
    for j, name in enumerate(STYLE_ORDER):
        style_stats[name] = _stat(styles[:, j])
    with open(STYLE_STATS_PATH, "w") as f:
        json.dump(style_stats, f, indent=2)

    print(f"\n=== target stats -> {TARGET_STATS_PATH} ===")
    for name in TARGET_ORDER:
        print(f"  {name:14s} mean {target_stats[name]['mean']:+.5f}  std {target_stats[name]['std']:.5f}")
    print(f"\n=== style stats -> {STYLE_STATS_PATH} ===")
    for name in STYLE_ORDER:
        print(f"  {name:16s} mean {style_stats[name]['mean']:+.5f}  std {style_stats[name]['std']:.5f}")


if __name__ == "__main__":
    main()
