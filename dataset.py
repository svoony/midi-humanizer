"""
PyTorch Dataset for the normalized -> performed MIDI translation task.

Each example is a window of consecutive notes from one piece. The source
sequence is that window quantized to the piece's estimated grid with flat
velocity (the "score"); the target sequence is the same notes as actually
performed (the "performance"). Both are encoded with the shared event
vocabulary in tokenizer.py.

Alignment note: we recompute the quantized version directly from
original.midi's note list (same deterministic grid-estimation code used to
build normalized.midi) instead of reading normalized.midi back from disk.
MIDI files are always stored in time-sorted order, so when quantization
causes two notes to tie or swap order, re-reading normalized.midi would
silently shuffle it out of correspondence with original.midi. Recomputing
keeps the two sequences index-aligned note-for-note.
"""

import glob
import json
import os
from collections import namedtuple

import pretty_midi
import torch
from torch.utils.data import Dataset

from normalize_midi import CONSTANT_VELOCITY, estimate_grid, quantize_time
from tokenizer import EOS_ID, PAD_ID, events_to_token_ids, notes_to_events

SimpleNote = namedtuple("SimpleNote", ["pitch", "start", "end", "velocity"])


def _piece_dirs(root, split):
    return sorted(glob.glob(os.path.join(root, split, "*", "*")))


def _note_count(piece_dir):
    pm = pretty_midi.PrettyMIDI(os.path.join(piece_dir, "original.midi"))
    return len(pm.instruments[0].notes)


class MaestroPairDataset(Dataset):
    def __init__(self, root="paired_data", split="train", window_notes=256,
                 stride=256, max_seq_len=2048, index_cache=True):
        self.root = root
        self.split = split
        self.window_notes = window_notes
        self.max_seq_len = max_seq_len

        piece_dirs = _piece_dirs(root, split)
        if not piece_dirs:
            raise ValueError(f"no pieces found under {root}/{split}")

        counts = self._load_or_build_note_counts(piece_dirs, index_cache)

        self.windows = []  # list of (piece_dir, start_idx)
        for piece_dir in piece_dirs:
            n = counts[piece_dir]
            if n == 0:
                continue
            starts = list(range(0, n, stride)) or [0]
            for s in starts:
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

    def __getitem__(self, idx):
        piece_dir, start = self.windows[idx]
        pm = pretty_midi.PrettyMIDI(os.path.join(piece_dir, "original.midi"))
        notes = sorted(pm.instruments[0].notes, key=lambda n: n.start)
        window = notes[start:start + self.window_notes]

        grid_unit, phase = estimate_grid(pm)
        t0 = window[0].start

        tgt_notes = [SimpleNote(n.pitch, n.start - t0, n.end - t0, n.velocity) for n in window]
        src_notes = []
        for n in window:
            q_start = quantize_time(n.start, grid_unit, phase) - t0
            q_end = quantize_time(n.end, grid_unit, phase) - t0
            if q_end <= q_start:
                q_end = q_start + grid_unit
            src_notes.append(SimpleNote(n.pitch, q_start, q_end, CONSTANT_VELOCITY))

        src_ids = events_to_token_ids(notes_to_events(src_notes))
        tgt_ids = events_to_token_ids(notes_to_events(tgt_notes))

        src_ids = self._truncate(src_ids)
        tgt_ids = self._truncate(tgt_ids)

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
            "piece": os.path.basename(piece_dir),
            "start": start,
        }

    def _truncate(self, ids):
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len - 1] + [EOS_ID]
        return ids


def collate_fn(batch):
    def pad(seqs):
        max_len = max(len(s) for s in seqs)
        out = torch.full((len(seqs), max_len), PAD_ID, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = s
        return out

    src = pad([b["src"] for b in batch])
    tgt = pad([b["tgt"] for b in batch])
    return {
        "src": src,
        "src_pad_mask": src == PAD_ID,
        "tgt": tgt,
        "tgt_pad_mask": tgt == PAD_ID,
        "pieces": [b["piece"] for b in batch],
        "starts": [b["start"] for b in batch],
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = MaestroPairDataset(split="validation", window_notes=128, stride=128)
    print(f"validation windows: {len(ds)}")

    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dl))
    print("src shape", batch["src"].shape, "tgt shape", batch["tgt"].shape)
    print("pieces", batch["pieces"], "starts", batch["starts"])

    from tokenizer import token_ids_to_notes

    sample = ds[0]
    decoded_src = token_ids_to_notes(sample["src"].tolist())
    decoded_tgt = token_ids_to_notes(sample["tgt"].tolist())
    print(f"round trip note counts: src={len(decoded_src)} tgt={len(decoded_tgt)}")
    src_pitches = [n[0] for n in decoded_src]
    tgt_pitches = [n[0] for n in decoded_tgt]
    print("pitch sequences match:", src_pitches == tgt_pitches)
