"""
Normalize an expressive performance MIDI into a "flat" score-like MIDI:
removes rubato (quantizes timing to a grid at one estimated tempo per piece),
removes dynamics (constant velocity), and removes pedaling (drops all CCs).
"""
import sys
import numpy as np
import pretty_midi

CONSTANT_VELOCITY = 80
SUBDIVISIONS_PER_BEAT = 4  # 16th-note grid
PHASE_SEARCH_STEPS = 200


def estimate_grid(pm):
    tempo = pm.estimate_tempo()
    beat_duration = 60.0 / tempo
    grid_unit = beat_duration / SUBDIVISIONS_PER_BEAT

    onsets = np.array([n.start for inst in pm.instruments for n in inst.notes])
    if len(onsets) == 0:
        return grid_unit, 0.0

    phases = np.linspace(0, grid_unit, PHASE_SEARCH_STEPS, endpoint=False)
    best_phase, best_error = 0.0, np.inf
    for phase in phases:
        residual = np.mod(onsets - phase, grid_unit)
        error = np.sum(np.minimum(residual, grid_unit - residual))
        if error < best_error:
            best_error = error
            best_phase = phase
    return grid_unit, best_phase


def quantize_time(t, grid_unit, phase):
    return round((t - phase) / grid_unit) * grid_unit + phase


def normalize_midi(in_path, out_path):
    pm = pretty_midi.PrettyMIDI(in_path)
    grid_unit, phase = estimate_grid(pm)

    for inst in pm.instruments:
        inst.control_changes = []
        inst.pitch_bends = []
        for note in inst.notes:
            q_start = quantize_time(note.start, grid_unit, phase)
            q_end = quantize_time(note.end, grid_unit, phase)
            if q_end <= q_start:
                q_end = q_start + grid_unit
            note.start = q_start
            note.end = q_end
            note.velocity = CONSTANT_VELOCITY
        inst.notes.sort(key=lambda n: (n.start, n.pitch))

    pm.write(out_path)


if __name__ == "__main__":
    normalize_midi(sys.argv[1], sys.argv[2])
