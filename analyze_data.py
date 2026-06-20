"""
Characterize the train split to ground normalization constants and head
design for the note-wise performance-regression model: how big are timing
offsets (rubato), duration ratios (articulation), and what's the velocity
range and notes-per-piece distribution.
"""
import glob
import os

import numpy as np
import pretty_midi

from normalize_midi import quantize_time, estimate_grid

ROOT = "paired_data/train"


def percentiles(arr, ps=(1, 5, 25, 50, 75, 95, 99)):
    return {p: round(float(np.percentile(arr, p)), 4) for p in ps}


def main():
    piece_dirs = sorted(glob.glob(os.path.join(ROOT, "*", "*")))
    print(f"analyzing {len(piece_dirs)} train pieces")

    timing_offsets = []
    duration_ratios = []
    velocities = []
    note_counts = []
    grid_units = []

    for i, piece_dir in enumerate(piece_dirs):
        pm = pretty_midi.PrettyMIDI(os.path.join(piece_dir, "original.midi"))
        notes = sorted(pm.instruments[0].notes, key=lambda n: n.start)
        note_counts.append(len(notes))
        if not notes:
            continue

        grid_unit, phase = estimate_grid(pm)
        grid_units.append(grid_unit)

        for n in notes:
            q_start = quantize_time(n.start, grid_unit, phase)
            q_end = quantize_time(n.end, grid_unit, phase)
            if q_end <= q_start:
                q_end = q_start + grid_unit

            timing_offsets.append(n.start - q_start)
            duration_ratios.append((n.end - n.start) / (q_end - q_start))
            velocities.append(n.velocity)

        if i % 200 == 0:
            print(f"{i}/{len(piece_dirs)}")

    timing_offsets = np.array(timing_offsets)
    duration_ratios = np.array(duration_ratios)
    velocities = np.array(velocities)
    note_counts = np.array(note_counts)
    grid_units = np.array(grid_units)

    print("\n=== timing_offset (seconds, performed - quantized onset) ===")
    print("mean", round(float(timing_offsets.mean()), 4), "std", round(float(timing_offsets.std()), 4))
    print(percentiles(timing_offsets))

    print("\n=== duration_ratio (performed_dur / quantized_dur) ===")
    print("mean", round(float(duration_ratios.mean()), 4), "std", round(float(duration_ratios.std()), 4))
    print(percentiles(duration_ratios))

    print("\n=== velocity (raw, 0-127) ===")
    print("mean", round(float(velocities.mean()), 2), "std", round(float(velocities.std()), 2))
    print(percentiles(velocities, ps=(1, 5, 25, 50, 75, 95, 99)))
    print("min/max", int(velocities.min()), int(velocities.max()))

    print("\n=== notes per piece ===")
    print("mean", round(float(note_counts.mean()), 1), "median", int(np.median(note_counts)))
    print(percentiles(note_counts))
    print("min/max", int(note_counts.min()), int(note_counts.max()))

    print("\n=== estimated grid_unit (seconds, i.e. 16th-note length) ===")
    print("mean", round(float(grid_units.mean()), 4))
    print(percentiles(grid_units))

    np.savez(
        "data_stats.npz",
        timing_offsets=timing_offsets,
        duration_ratios=duration_ratios,
        velocities=velocities,
        note_counts=note_counts,
        grid_units=grid_units,
    )
    print("\nsaved raw arrays to data_stats.npz")


if __name__ == "__main__":
    main()
