"""
Run the archived gen-2 PerformanceCVAE for side-by-side comparison in the eval
viewer. Gen-2's class was replaced by gen-3's PerformanceNet, and gen-2 used
the un-folded tempo grid + the bounded onset-offset timing target, so it gets
its own (faithful) inference path here.

Fairness note: gen-2 and gen-3 predict *different* timing quantities (gen-2:
onset minus nearest grid, capped at +-half a grid step; gen-3: local-tempo
log-ratio). They are therefore shown each against their own ground truth in
their own units, never forced onto a shared rubato axis. Velocity (0-1) is the
same target in both and is overlaid directly.
"""
import math
import os

import numpy as np
import pretty_midi
import torch
import torch.nn as nn

from infer import load_input
from normalize_midi import PHASE_SEARCH_STEPS, SUBDIVISIONS_PER_BEAT, quantize_time
from note_dataset import (
    FLOAT_FIELDS, LONG_FIELDS, N_TARGETS, SCORE_FEATURE_FIELDS,
    BAR_POS_VOCAB, BEAT_IN_MEASURE_VOCAB, MEASURE_MOD, N_CHORD_QUALITY,
    N_CHORD_ROOT, N_ERAS, N_KEY_MODE, N_KEY_TONIC, N_PITCHES, N_ROMAN_DEGREE,
    N_SCALE_DEGREE, N_VOICE_ROLES, ERA_NAMES, _compute_chord_features,
    _compute_local_density, _compute_melodic_intervals, compute_input_features,
)
from score_features import compute_score_features

GEN2_CHECKPOINT = os.path.join("archive", "20260622_gen2_pre_gen3", "best.pt")
WINDOW_NOTES = 192
LOGVAR_MIN, LOGVAR_MAX = -7.0, 3.0


# ---- gen-2 architecture (PerformanceCVAE), reconstructed to load the state dict ----
class _PE(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1]].unsqueeze(0)


def _masked_mean(x, pad_mask):
    keep = (~pad_mask).unsqueeze(-1).float()
    return (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)


class _RecognitionEncoder(nn.Module):
    def __init__(self, in_dim, z_dim, d_model=256, n_layers=3, n_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim + N_TARGETS, d_model)
        self.pos_encoding = _PE(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.to_mu = nn.Linear(d_model, z_dim)
        self.to_logvar = nn.Linear(d_model, z_dim)

    def forward(self, feat, y, pad_mask):
        h = self.pos_encoding(self.in_proj(torch.cat([feat, y], dim=-1)))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        pooled = _masked_mean(h, pad_mask)
        return self.to_mu(pooled), self.to_logvar(pooled).clamp(-8.0, 8.0)


class PerformanceCVAE(nn.Module):
    def __init__(self, d_model=256, n_layers=6, n_heads=8, dim_feedforward=1024, dropout=0.1, z_dim=48):
        super().__init__()
        self.z_dim = z_dim
        self.pitch_embed = nn.Embedding(N_PITCHES, 64)
        self.beat_embed = nn.Embedding(SUBDIVISIONS_PER_BEAT, 32)
        self.bar_embed = nn.Embedding(BAR_POS_VOCAB, 32)
        self.beat_in_measure_embed = nn.Embedding(BEAT_IN_MEASURE_VOCAB, 16)
        self.measure_embed = nn.Embedding(MEASURE_MOD, 16)
        self.voice_role_embed = nn.Embedding(N_VOICE_ROLES, 16)
        self.era_embed = nn.Embedding(N_ERAS, 16)
        self.key_tonic_embed = nn.Embedding(N_KEY_TONIC, 16)
        self.key_mode_embed = nn.Embedding(N_KEY_MODE, 8)
        self.scale_degree_embed = nn.Embedding(N_SCALE_DEGREE, 16)
        self.chord_root_embed = nn.Embedding(N_CHORD_ROOT, 16)
        self.chord_quality_embed = nn.Embedding(N_CHORD_QUALITY, 16)
        self.roman_degree_embed = nn.Embedding(N_ROMAN_DEGREE, 12)
        self.continuous_proj = nn.Linear(len(FLOAT_FIELDS), 64)
        in_dim = 64 + 32 + 32 + 16 + 16 + 16 + 16 + 16 + 8 + 16 + 16 + 16 + 12 + 64
        self.input_proj = nn.Linear(in_dim, d_model)
        self.z_to_bias = nn.Linear(z_dim, d_model)
        self.pos_encoding = _PE(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 2 * N_TARGETS)
        self.recognition = _RecognitionEncoder(in_dim, z_dim, d_model=d_model, n_heads=n_heads,
                                               dim_feedforward=dim_feedforward, dropout=dropout)

    def _embed_score(self, b):
        parts = [
            self.pitch_embed(b["pitch"]), self.beat_embed(b["beat_pos"]), self.bar_embed(b["bar_pos"]),
            self.beat_in_measure_embed(b["beat_in_measure"]), self.measure_embed(b["measure_number"]),
            self.voice_role_embed(b["voice_role"]), self.era_embed(b["era"]),
            self.key_tonic_embed(b["key_tonic"]), self.key_mode_embed(b["key_mode"]),
            self.scale_degree_embed(b["scale_degree"]), self.chord_root_embed(b["chord_root"]),
            self.chord_quality_embed(b["chord_quality"]), self.roman_degree_embed(b["roman_degree"]),
            self.continuous_proj(torch.stack([b[f] for f in FLOAT_FIELDS], dim=-1)),
        ]
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def predict(self, batch):
        feat = self._embed_score(batch)
        z = torch.zeros(feat.shape[0], self.z_dim, device=feat.device)  # deadpan
        x = self.input_proj(feat) + self.z_to_bias(z).unsqueeze(1)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=batch["pad_mask"])
        out = self.head(x)
        return out[..., :N_TARGETS], out[..., N_TARGETS:].clamp(LOGVAR_MIN, LOGVAR_MAX)


_cache = {}


def _load_gen2(device):
    if device not in _cache:
        ckpt = torch.load(GEN2_CHECKPOINT, map_location=device, weights_only=False)
        m = PerformanceCVAE(**ckpt.get("config", {})).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        _cache[device] = (m, np.asarray(ckpt["target_mean"], np.float32), np.asarray(ckpt["target_std"], np.float32))
    return _cache[device]


def _unfolded_grid(pm):
    """Gen-2's grid: pretty_midi's raw (un-folded) tempo estimate."""
    tempo = pm.estimate_tempo()
    grid_unit = (60.0 / tempo) / SUBDIVISIONS_PER_BEAT
    onsets = np.array([n.start for inst in pm.instruments for n in inst.notes])
    if len(onsets) == 0:
        return grid_unit, 0.0
    best_phase, best_err = 0.0, np.inf
    for phase in np.linspace(0, grid_unit, PHASE_SEARCH_STEPS, endpoint=False):
        residual = np.mod(onsets - phase, grid_unit)
        err = np.sum(np.minimum(residual, grid_unit - residual))
        if err < best_err:
            best_err, best_phase = err, phase
    return grid_unit, best_phase


def gen2_predict(input_path, era, device=None):
    """Gen-2 predictions on its native un-folded grid. Returns per-note arrays:
    timing_offset (predicted, seconds), velocity (0-1), and q_unfolded (grid
    onsets) so the caller can form gen-2's ground-truth offset = human - grid."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    era_id = ERA_NAMES.index(era) if isinstance(era, str) else era

    pm = pretty_midi.PrettyMIDI(input_path)
    notes = sorted((n for inst in pm.instruments for n in inst.notes if not inst.is_drum), key=lambda n: n.start)
    pitches = np.array([n.pitch for n in notes], dtype=np.int64)
    raw_starts = np.array([n.start for n in notes], dtype=np.float64)
    raw_ends = np.array([n.end for n in notes], dtype=np.float64)
    grid_unit, phase = _unfolded_grid(pm)
    starts = np.array([quantize_time(t, grid_unit, phase) for t in raw_starts])
    ends = np.maximum(np.array([quantize_time(t, grid_unit, phase) for t in raw_ends]), starts + grid_unit)
    beats_per_measure = pm.time_signature_changes[0].numerator if pm.time_signature_changes else 4
    piece_duration = float(ends.max())

    chord_size, voice_role = _compute_chord_features(starts, pitches)
    interval_prev, interval_next = _compute_melodic_intervals(pitches)
    local_density = _compute_local_density(starts, grid_unit)
    piece_feats = compute_input_features(pitches, starts, ends, grid_unit, phase, beats_per_measure)
    score_feats = compute_score_features(pitches, piece_feats["grid_steps"], piece_feats["end_grid_steps"],
                                         SUBDIVISIONS_PER_BEAT, beats_per_measure)

    model, tmean, tstd = _load_gen2(device)
    n_total = len(pitches)
    out_q, out_timing, out_vel = [], [], []
    for cs in range(0, n_total, WINDOW_NOTES):
        ce = min(cs + WINDOW_NOTES, n_total)
        n = ce - cs
        feats = compute_input_features(pitches[cs:ce], starts[cs:ce], ends[cs:ce], grid_unit, phase, beats_per_measure)
        fa = {
            "pitch": feats["pitches"], "rel_step": feats["rel_steps"].astype(np.float32),
            "beat_pos": feats["beat_pos"], "bar_pos": feats["bar_pos"],
            "beat_in_measure": feats["beat_in_measure"], "measure_number": feats["measure_number"],
            "ioi": feats["ioi"].astype(np.float32), "dur_grid": feats["dur_grid"].astype(np.float32),
            "interval_prev": interval_prev[cs:ce].astype(np.float32), "interval_next": interval_next[cs:ce].astype(np.float32),
            "chord_size": chord_size[cs:ce].astype(np.float32), "voice_role": voice_role[cs:ce].astype(np.int64),
            "local_density": local_density[cs:ce].astype(np.float32),
            "tempo_scalar": np.full(n, math.log(grid_unit), dtype=np.float32),
            "piece_position": (starts[cs:ce] / piece_duration).astype(np.float32),
            "era": np.full(n, era_id, dtype=np.int64),
        }
        for f in SCORE_FEATURE_FIELDS:
            fa[f] = score_feats[f][cs:ce]
        x = {}
        for f in LONG_FIELDS:
            x[f] = torch.from_numpy(fa[f].astype(np.int64)).unsqueeze(0).to(device)
        for f in FLOAT_FIELDS:
            x[f] = torch.from_numpy(fa[f].astype(np.float32)).unsqueeze(0).to(device)
        x["pad_mask"] = torch.zeros(1, n, dtype=torch.bool).to(device)
        mean, _ = model.predict(x)
        pred = mean[0].cpu().numpy() * tstd + tmean
        out_q.append(feats["q_starts"])
        out_timing.append(pred[:, 0])
        out_vel.append(np.clip(pred[:, 2], 0.0, 1.0))

    return {
        "q_unfolded": np.concatenate(out_q),
        "timing_offset_pred": np.concatenate(out_timing),
        "velocity_pred": np.concatenate(out_vel),
        "human_starts": raw_starts,
    }
