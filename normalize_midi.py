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

# pretty_midi.estimate_tempo systematically returns an octave (2x/4x) too fast
# on dense piano writing - e.g. 239 bpm on Liszt's Liebestraum, ~4x the real
# beat - which misaligns the metric grid and corrupts the timing target. Fold
# the estimate into a perceptually central one-octave band [60, 120) bpm.
TEMPO_FOLD_MIN = 60.0
TEMPO_FOLD_MAX = 120.0


def fold_tempo(tempo):
    if tempo <= 0:
        return TEMPO_FOLD_MIN
    while tempo >= TEMPO_FOLD_MAX:
        tempo /= 2.0
    while tempo < TEMPO_FOLD_MIN:
        tempo *= 2.0
    return tempo


def estimate_grid(pm):
    tempo = fold_tempo(pm.estimate_tempo())
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
