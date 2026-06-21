"""
Note-wise regression dataset: for each window of consecutive score notes,
produce per-note input features (pitch, metric position, melodic context,
chord context, tempo, density, era) and per-note performance targets
(timing offset, log duration ratio, velocity, sustain pedal).

Same alignment approach throughout: quantization, chord structure, melodic
intervals, and local density are all recomputed directly from
original.midi's note list (cached per piece) rather than read back from
normalized.midi, since MIDI files are stored time-sorted and ties on the
quantized grid can silently reorder notes on disk.
"""

import csv as csv_module
import glob
import json
import math
import os

import numpy as np
import pretty_midi
import torch
from torch.utils.data import Dataset

from normalize_midi import estimate_grid

PITCH_MIN, PITCH_MAX = 21, 108
N_PITCHES = PITCH_MAX - PITCH_MIN + 1
SUBDIVISIONS_PER_BEAT = 4

BAR_POS_VOCAB = 48          # generous cap: covers meters up to 12 beats/measure
BEAT_IN_MEASURE_VOCAB = 16  # generous cap: covers meters up to 16 beats/measure
MEASURE_MOD = 8             # measure number wraps mod this, same convention as bar_pos/beat_pos
N_VOICE_ROLES = 4           # solo, top-of-chord, bottom-of-chord, inner
N_ERAS = 4                  # baroque, classical, romantic, modern

CHORD_ONSET_TOL = 0.01            # seconds; onsets closer than this count as one chord
LOCAL_DENSITY_LOOKBACK_BEATS = 2.0

# from the train-split characterization pass (analyze_data.py)
DURATION_RATIO_EPS = 1e-3
VELOCITY_SCALE = 127.0

CACHE_FILENAME = "_note_cache.npz"
CACHE_VERSION = 3  # bump when the cached fields change, to invalidate stale caches
SUSTAIN_CC = 64

MAESTRO_CSV_PATH = os.path.join("raw_midi", "maestro-v3.0.0", "maestro-v3.0.0.csv")

ERA_NAMES = ["baroque", "classical", "romantic", "modern"]

ERA_BY_COMPOSER = {
    # baroque
    "Johann Sebastian Bach": 0, "George Frideric Handel": 0, "Domenico Scarlatti": 0,
    "Antonio Soler": 0, "Henry Purcell": 0, "Johann Pachelbel": 0,
    "Jean-Philippe Rameau": 0, "Orlando Gibbons": 0,
    # classical
    "Joseph Haydn": 1, "Wolfgang Amadeus Mozart": 1, "Muzio Clementi": 1,
    "Ludwig van Beethoven": 1, "Johann Christian Fischer": 1,
    # romantic
    "Franz Schubert": 2, "Carl Maria von Weber": 2, "Felix Mendelssohn": 2,
    "Frédéric Chopin": 2, "Robert Schumann": 2, "Franz Liszt": 2,
    "Richard Wagner": 2, "Giuseppe Verdi": 2, "Johannes Brahms": 2,
    "César Franck": 2, "Mikhail Glinka": 2, "Mily Balakirev": 2,
    "Modest Mussorgsky": 2, "Pyotr Ilyich Tchaikovsky": 2, "Edvard Grieg": 2,
    "Isaac Albéniz": 2, "Charles Gounod": 2, "Georges Bizet": 2,
    "Niccolò Paganini": 2, "Nikolai Rimsky-Korsakov": 2, "Johann Strauss": 2,
    "Fritz Kreisler": 2,
    # modern
    "Claude Debussy": 3, "Sergei Rachmaninoff": 3, "Alexander Scriabin": 3,
    "Nikolai Medtner": 3, "Leoš Janáček": 3, "Alban Berg": 3,
    "George Enescu": 3, "Percy Grainger": 3,
}
DEFAULT_ERA = 2  # romantic - the most populous bucket; only used for an unmapped composer

_warned_composers = set()


def era_for_composer(canonical_composer):
    """MAESTRO lists the original composer first and any arranger/transcriber
    second (e.g. 'Johann Sebastian Bach / Ferruccio Busoni'); era is decided
    by the original composer."""
    primary = canonical_composer.split("/")[0].strip()
    if primary not in ERA_BY_COMPOSER and primary not in _warned_composers:
        _warned_composers.add(primary)
        print(f"WARNING: unmapped composer '{primary}', defaulting to romantic era")
    return ERA_BY_COMPOSER.get(primary, DEFAULT_ERA)


def build_era_map(root, csv_path=MAESTRO_CSV_PATH):
    """piece_dir -> era_id for every piece listed in the MAESTRO csv."""
    era_map = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv_module.DictReader(f):
            year, filename = row["midi_filename"].split("/", 1)
            piece_name = os.path.splitext(filename)[0]
            piece_dir = os.path.join(root, row["split"], year, piece_name)
            era_map[piece_dir] = era_for_composer(row["canonical_composer"])
    return era_map


def _piece_dirs(root, split):
    return sorted(glob.glob(os.path.join(root, split, "*", "*")))


def _compute_chord_features(starts, pitches, tol=CHORD_ONSET_TOL):
    """For each note: chord_size (how many notes share its onset) and
    voice_role (solo / top / bottom / inner within that chord)."""
    n = len(starts)
    chord_size = np.ones(n, dtype=np.int16)
    voice_role = np.zeros(n, dtype=np.int8)  # default: solo
    if n == 0:
        return chord_size, voice_role

    cluster_id = np.zeros(n, dtype=np.int64)
    cluster_id[1:] = np.cumsum(np.diff(starts) > tol)

    boundaries = np.flatnonzero(np.diff(cluster_id)) + 1
    groups = np.split(np.arange(n), boundaries)
    for g in groups:
        size = len(g)
        chord_size[g] = size
        if size > 1:
            p = pitches[g]
            voice_role[g] = 3  # inner
            voice_role[g[np.argmax(p)]] = 1  # top
            voice_role[g[np.argmin(p)]] = 2  # bottom
    return chord_size, voice_role


def _compute_melodic_intervals(pitches):
    n = len(pitches)
    interval_prev = np.zeros(n, dtype=np.int16)
    interval_next = np.zeros(n, dtype=np.int16)
    if n > 1:
        diffs = np.diff(pitches.astype(np.int16))
        interval_prev[1:] = diffs
        interval_next[:-1] = diffs
    return interval_prev, interval_next


def _compute_local_density(starts, grid_unit, lookback_beats=LOCAL_DENSITY_LOOKBACK_BEATS):
    """Notes per beat in the trailing lookback window before (and including)
    each note's onset."""
    beat_duration = grid_unit * SUBDIVISIONS_PER_BEAT
    lookback_seconds = lookback_beats * beat_duration
    lo_idx = np.searchsorted(starts, starts - lookback_seconds, side="left")
    hi_idx = np.searchsorted(starts, starts, side="right")
    counts = (hi_idx - lo_idx).astype(np.float32)
    return counts / lookback_beats


def _load_piece_arrays(piece_dir):
    """Parse original.midi once and cache all derived per-note arrays plus
    piece-level scalars to disk as a small .npz. Subsequent calls for the
    same piece, in any process, hit the disk cache instead of re-parsing."""
    cache_path = os.path.join(piece_dir, CACHE_FILENAME)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        if int(data.get("cache_version", -1)) == CACHE_VERSION:
            return {k: data[k] for k in data.files}

    pm = pretty_midi.PrettyMIDI(os.path.join(piece_dir, "original.midi"))
    notes = sorted(pm.instruments[0].notes, key=lambda n: n.start)
    pitches = np.array([n.pitch for n in notes], dtype=np.int16)
    starts = np.array([n.start for n in notes], dtype=np.float64)
    ends = np.array([n.end for n in notes], dtype=np.float64)
    velocities = np.array([n.velocity for n in notes], dtype=np.uint8)
    grid_unit, phase = estimate_grid(pm)

    pedal_ccs = sorted(
        (cc for cc in pm.instruments[0].control_changes if cc.number == SUSTAIN_CC),
        key=lambda cc: cc.time,
    )
    pedal_times = np.array([cc.time for cc in pedal_ccs], dtype=np.float64)
    pedal_values = np.array([cc.value for cc in pedal_ccs], dtype=np.uint8)

    chord_size, voice_role = _compute_chord_features(starts, pitches)
    interval_prev, interval_next = _compute_melodic_intervals(pitches)
    local_density = _compute_local_density(starts, grid_unit)

    if pm.time_signature_changes:
        numerator = pm.time_signature_changes[0].numerator
        denominator = pm.time_signature_changes[0].denominator
    else:
        numerator, denominator = 4, 4

    data = dict(
        pitches=pitches, starts=starts, ends=ends, velocities=velocities,
        pedal_times=pedal_times, pedal_values=pedal_values,
        chord_size=chord_size, voice_role=voice_role,
        interval_prev=interval_prev, interval_next=interval_next,
        local_density=local_density,
        grid_unit=grid_unit, phase=phase,
        time_sig_numerator=numerator, time_sig_denominator=denominator,
        cache_version=CACHE_VERSION,
    )
    np.savez(cache_path, **data)
    return data


def pedal_at_times(query_times, pedal_times, pedal_values):
    """Sustain pedal value in effect at each query time (step-function
    sample of CC64): the most recent pedal event at or before that time,
    or 0 (pedal up) if no event has occurred yet."""
    if len(pedal_times) == 0:
        return np.zeros(len(query_times), dtype=np.float32)
    idx = np.searchsorted(pedal_times, query_times, side="right") - 1
    return np.where(idx >= 0, pedal_values[np.clip(idx, 0, None)], 0).astype(np.float32)


def _note_count(piece_dir):
    return len(_load_piece_arrays(piece_dir)["pitches"])


def compute_input_features(pitches, starts, ends, grid_unit, phase, beats_per_measure=4):
    """Derive the grid-dependent input features (metric position, IOI,
    duration) from a window of notes and a known quantization grid plus
    time signature. Used both by the training dataset (grid recomputed from
    the original performance) and by inference (grid estimated directly
    from a flat/normalized MIDI, where it's the only input)."""
    grid_steps = np.round((starts - phase) / grid_unit).astype(np.int64)
    end_grid_steps = np.round((ends - phase) / grid_unit).astype(np.int64)
    end_grid_steps = np.maximum(end_grid_steps, grid_steps + 1)
    first_step = grid_steps[0]

    measure_length_steps = max(1, beats_per_measure * SUBDIVISIONS_PER_BEAT)

    return {
        "pitches": pitches.astype(np.int64) - PITCH_MIN,
        "rel_steps": grid_steps - first_step,
        "beat_pos": grid_steps % SUBDIVISIONS_PER_BEAT,
        "bar_pos": (grid_steps % measure_length_steps) % BAR_POS_VOCAB,
        "beat_in_measure": ((grid_steps // SUBDIVISIONS_PER_BEAT) % beats_per_measure) % BEAT_IN_MEASURE_VOCAB,
        "measure_number": (grid_steps // measure_length_steps) % MEASURE_MOD,
        "dur_grid": end_grid_steps - grid_steps,
        "ioi": np.diff(grid_steps, prepend=first_step),
        "grid_steps": grid_steps,
        "end_grid_steps": end_grid_steps,
        "q_starts": grid_steps * grid_unit + phase,
        "q_ends": end_grid_steps * grid_unit + phase,
    }


class NoteRegressionDataset(Dataset):
    def __init__(self, root="paired_data", split="train", window_notes=192,
                 stride=192, index_cache=True):
        self.root = root
        self.split = split
        self.window_notes = window_notes
        self._cache = {}  # piece_dir -> arrays dict, populated lazily per process

        piece_dirs = _piece_dirs(root, split)
        if not piece_dirs:
            raise ValueError(f"no pieces found under {root}/{split}")

        self.era_map = build_era_map(root)

        counts = self._load_or_build_note_counts(piece_dirs, index_cache)

        self.windows = []
        for piece_dir in piece_dirs:
            n = counts[piece_dir]
            if n == 0:
                continue
            for s in range(0, n, stride):
                if s < n:
                    self.windows.append((piece_dir, s))

    def _load_or_build_note_counts(self, piece_dirs, index_cache):
        cache_path = os.path.join(self.root, f"{self.split}_note_counts.json")
        if index_cache and os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            if set(cached.keys()) == set(piece_dirs):
                return cached

        counts = {d: _note_count(d) for d in piece_dirs}
        if index_cache:
            with open(cache_path, "w") as f:
                json.dump(counts, f)
        return counts

    def __len__(self):
        return len(self.windows)

    def _get_piece(self, piece_dir):
        arrays = self._cache.get(piece_dir)
        if arrays is None:
            arrays = _load_piece_arrays(piece_dir)
            self._cache[piece_dir] = arrays
        return arrays

    def __getitem__(self, idx):
        piece_dir, start = self.windows[idx]
        p = self._get_piece(piece_dir)
        grid_unit, phase = float(p["grid_unit"]), float(p["phase"])
        beats_per_measure = int(p["time_sig_numerator"])
        piece_duration = float(p["ends"].max()) if len(p["ends"]) else 1.0

        end = start + self.window_notes
        w_pitches = p["pitches"][start:end]
        w_starts = p["starts"][start:end]
        w_ends = p["ends"][start:end]
        w_vels = p["velocities"][start:end]
        w_chord_size = p["chord_size"][start:end]
        w_voice_role = p["voice_role"][start:end]
        w_interval_prev = p["interval_prev"][start:end]
        w_interval_next = p["interval_next"][start:end]
        w_local_density = p["local_density"][start:end]

        feats = compute_input_features(w_pitches, w_starts, w_ends, grid_unit, phase, beats_per_measure)
        q_starts, q_ends = feats["q_starts"], feats["q_ends"]

        timing_offset = w_starts - q_starts
        ratio = (w_ends - w_starts) / np.maximum(q_ends - q_starts, DURATION_RATIO_EPS)
        log_dur_ratio = np.log(np.maximum(ratio, DURATION_RATIO_EPS))
        velocity = w_vels.astype(np.float32) / VELOCITY_SCALE
        pedal = pedal_at_times(w_starts, p["pedal_times"], p["pedal_values"]) / VELOCITY_SCALE

        n = len(w_pitches)
        tempo_scalar = np.full(n, math.log(grid_unit), dtype=np.float32)
        piece_position = (w_starts / piece_duration).astype(np.float32)
        era_id = np.full(n, self.era_map.get(piece_dir, DEFAULT_ERA), dtype=np.int64)

        x = {
            "pitch": torch.from_numpy(feats["pitches"]),
            "rel_step": torch.from_numpy(feats["rel_steps"].astype(np.float32)),
            "beat_pos": torch.from_numpy(feats["beat_pos"]),
            "bar_pos": torch.from_numpy(feats["bar_pos"]),
            "beat_in_measure": torch.from_numpy(feats["beat_in_measure"]),
            "measure_number": torch.from_numpy(feats["measure_number"]),
            "ioi": torch.from_numpy(feats["ioi"].astype(np.float32)),
            "dur_grid": torch.from_numpy(feats["dur_grid"].astype(np.float32)),
            "interval_prev": torch.from_numpy(w_interval_prev.astype(np.float32)),
            "interval_next": torch.from_numpy(w_interval_next.astype(np.float32)),
            "chord_size": torch.from_numpy(w_chord_size.astype(np.float32)),
            "voice_role": torch.from_numpy(w_voice_role.astype(np.int64)),
            "local_density": torch.from_numpy(w_local_density.astype(np.float32)),
            "tempo_scalar": torch.from_numpy(tempo_scalar),
            "piece_position": torch.from_numpy(piece_position),
            "era": torch.from_numpy(era_id),
        }
        y = torch.from_numpy(
            np.stack([timing_offset, log_dur_ratio, velocity, pedal], axis=-1).astype(np.float32)
        )
        return x, y


LONG_FIELDS = ["pitch", "beat_pos", "bar_pos", "beat_in_measure", "measure_number", "voice_role", "era"]
FLOAT_FIELDS = ["rel_step", "ioi", "dur_grid", "interval_prev", "interval_next",
                 "chord_size", "local_density", "tempo_scalar", "piece_position"]


def collate_fn(batch):
    lengths = [y.shape[0] for _, y in batch]
    max_len = max(lengths)
    bsz = len(batch)

    out = {f: torch.zeros(bsz, max_len, dtype=torch.long) for f in LONG_FIELDS}
    out.update({f: torch.zeros(bsz, max_len, dtype=torch.float) for f in FLOAT_FIELDS})
    y = torch.zeros(bsz, max_len, 4, dtype=torch.float)
    pad_mask = torch.ones(bsz, max_len, dtype=torch.bool)  # True = padding

    for i, (x, yi) in enumerate(batch):
        n = x["pitch"].shape[0]
        for f in LONG_FIELDS + FLOAT_FIELDS:
            out[f][i, :n] = x[f]
        y[i, :n] = yi
        pad_mask[i, :n] = False

    out["y"] = y
    out["pad_mask"] = pad_mask
    return out


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = NoteRegressionDataset(split="validation", window_notes=192, stride=192)
    print(f"validation windows: {len(ds)}")

    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dl))
    for k, v in batch.items():
        print(k, v.shape, v.dtype)
