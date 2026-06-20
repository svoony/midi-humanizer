"""
Note-wise regression dataset: for each window of consecutive score notes,
produce per-note input features (pitch, metric position, IOI, duration -
all derived from the quantized/flat grid) and per-note performance targets
(timing offset, log duration ratio, velocity).

Same alignment approach as dataset.py: quantization is recomputed directly
from original.midi's note list rather than read back from normalized.midi,
since MIDI files are stored time-sorted and ties on the quantized grid can
silently reorder notes on disk.
"""

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
BAR_SUBDIVISIONS = SUBDIVISIONS_PER_BEAT * 4  # assume 4/4 for the bar-position feature

# from the train-split characterization pass (analyze_data.py)
DURATION_RATIO_EPS = 1e-3
VELOCITY_SCALE = 127.0


CACHE_FILENAME = "_note_cache.npz"
CACHE_VERSION = 2  # bump when the cached fields change, to invalidate stale caches
SUSTAIN_CC = 64


def _piece_dirs(root, split):
    return sorted(glob.glob(os.path.join(root, split, "*", "*")))


def _load_piece_arrays(piece_dir):
    """Parse original.midi once and cache (pitches, starts, ends, velocities,
    sustain pedal events, grid_unit, phase) to disk as a small .npz.
    Subsequent calls for the same piece, in any process, hit the disk cache
    instead of re-parsing MIDI."""
    cache_path = os.path.join(piece_dir, CACHE_FILENAME)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        if int(data.get("cache_version", -1)) == CACHE_VERSION:
            return (
                data["pitches"], data["starts"], data["ends"], data["velocities"],
                data["pedal_times"], data["pedal_values"],
                float(data["grid_unit"]), float(data["phase"]),
            )

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

    np.savez(cache_path, pitches=pitches, starts=starts, ends=ends,
              velocities=velocities, pedal_times=pedal_times, pedal_values=pedal_values,
              grid_unit=grid_unit, phase=phase, cache_version=CACHE_VERSION)
    return pitches, starts, ends, velocities, pedal_times, pedal_values, grid_unit, phase


def pedal_at_times(query_times, pedal_times, pedal_values):
    """Sustain pedal value in effect at each query time (step-function
    sample of CC64): the most recent pedal event at or before that time,
    or 0 (pedal up) if no event has occurred yet."""
    if len(pedal_times) == 0:
        return np.zeros(len(query_times), dtype=np.float32)
    idx = np.searchsorted(pedal_times, query_times, side="right") - 1
    return np.where(idx >= 0, pedal_values[np.clip(idx, 0, None)], 0).astype(np.float32)


def _note_count(piece_dir):
    pitches, *_ = _load_piece_arrays(piece_dir)
    return len(pitches)


def compute_input_features(pitches, starts, ends, grid_unit, phase):
    """Derive the model's input features (pitch, metric position, IOI,
    duration - all in grid-step units) from a window of notes and a known
    quantization grid. Used both by the training dataset (grid recomputed
    from the original performance) and by inference (grid estimated
    directly from a flat/normalized MIDI, where it's the only input)."""
    grid_steps = np.round((starts - phase) / grid_unit).astype(np.int64)
    end_grid_steps = np.round((ends - phase) / grid_unit).astype(np.int64)
    end_grid_steps = np.maximum(end_grid_steps, grid_steps + 1)
    first_step = grid_steps[0]

    return {
        "pitches": pitches.astype(np.int64) - PITCH_MIN,
        "rel_steps": grid_steps - first_step,
        "beat_pos": grid_steps % SUBDIVISIONS_PER_BEAT,
        "bar_pos": grid_steps % BAR_SUBDIVISIONS,
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
        self._cache = {}  # piece_dir -> arrays, populated lazily per process

        piece_dirs = _piece_dirs(root, split)
        if not piece_dirs:
            raise ValueError(f"no pieces found under {root}/{split}")

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
        (pitches_arr, starts_arr, ends_arr, vels_arr,
         pedal_times, pedal_values, grid_unit, phase) = self._get_piece(piece_dir)

        end = start + self.window_notes
        w_pitches = pitches_arr[start:end]
        w_starts = starts_arr[start:end]
        w_ends = ends_arr[start:end]
        w_vels = vels_arr[start:end]

        feats = compute_input_features(w_pitches, w_starts, w_ends, grid_unit, phase)
        pitches = feats["pitches"]
        rel_steps, beat_pos, bar_pos = feats["rel_steps"], feats["beat_pos"], feats["bar_pos"]
        ioi, dur_grid = feats["ioi"], feats["dur_grid"]
        q_starts, q_ends = feats["q_starts"], feats["q_ends"]

        timing_offset = w_starts - q_starts
        ratio = (w_ends - w_starts) / np.maximum(q_ends - q_starts, DURATION_RATIO_EPS)
        log_dur_ratio = np.log(np.maximum(ratio, DURATION_RATIO_EPS))
        velocity = w_vels.astype(np.float32) / VELOCITY_SCALE
        pedal = pedal_at_times(w_starts, pedal_times, pedal_values) / VELOCITY_SCALE

        x = {
            "pitch": torch.from_numpy(pitches),
            "rel_step": torch.from_numpy(rel_steps.astype(np.float32)),
            "beat_pos": torch.from_numpy(beat_pos),
            "bar_pos": torch.from_numpy(bar_pos),
            "ioi": torch.from_numpy(ioi.astype(np.float32)),
            "dur_grid": torch.from_numpy(dur_grid.astype(np.float32)),
        }
        y = torch.from_numpy(
            np.stack([timing_offset, log_dur_ratio, velocity, pedal], axis=-1).astype(np.float32)
        )
        return x, y


def collate_fn(batch):
    lengths = [y.shape[0] for _, y in batch]
    max_len = max(lengths)
    bsz = len(batch)

    pitch = torch.zeros(bsz, max_len, dtype=torch.long)
    beat_pos = torch.zeros(bsz, max_len, dtype=torch.long)
    bar_pos = torch.zeros(bsz, max_len, dtype=torch.long)
    rel_step = torch.zeros(bsz, max_len, dtype=torch.float)
    ioi = torch.zeros(bsz, max_len, dtype=torch.float)
    dur_grid = torch.zeros(bsz, max_len, dtype=torch.float)
    y = torch.zeros(bsz, max_len, 4, dtype=torch.float)
    pad_mask = torch.ones(bsz, max_len, dtype=torch.bool)  # True = padding

    for i, (x, yi) in enumerate(batch):
        n = x["pitch"].shape[0]
        pitch[i, :n] = x["pitch"]
        beat_pos[i, :n] = x["beat_pos"]
        bar_pos[i, :n] = x["bar_pos"]
        rel_step[i, :n] = x["rel_step"]
        ioi[i, :n] = x["ioi"]
        dur_grid[i, :n] = x["dur_grid"]
        y[i, :n] = yi
        pad_mask[i, :n] = False

    return {
        "pitch": pitch,
        "beat_pos": beat_pos,
        "bar_pos": bar_pos,
        "rel_step": rel_step,
        "ioi": ioi,
        "dur_grid": dur_grid,
        "y": y,
        "pad_mask": pad_mask,
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = NoteRegressionDataset(split="validation", window_notes=192, stride=192)
    print(f"validation windows: {len(ds)}")

    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dl))
    for k, v in batch.items():
        print(k, v.shape, v.dtype)
