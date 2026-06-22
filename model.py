"""
PerformanceNet (gen-3) - a style-conditioned transformer that predicts a
distribution over expressive performance per note.

Three pieces:

  1. Shared note embedding. Each note's score features - pitch, metric
     position, melodic/chord context, tempo, density, era, and the harmonic /
     tonal-tension / phrase features - are embedded/projected per note.

  2. Global style conditioning. A small per-piece descriptor vector (tempo
     flexibility, dynamic range, articulation, pedalling, base tempo) is
     projected and added to every note. At training these come from the real
     performance; at inference the user sets them - the expressive control
     surface. Because the conditioning is observed (not an inferred latent), it
     can't posterior-collapse the way the gen-2 VAE did.

  3. Heteroscedastic head. The transformer backbone outputs a Gaussian per note
     per target - mean AND log-variance for the local-tempo deviation, log
     duration ratio, velocity, and pedal. The predicted std doubles as the
     per-note importance/uncertainty map.

Targets and the style vector are standardized (see note_dataset); the
checkpoint stores the stats so inference can invert/condition correctly.
"""
import math

import torch
import torch.nn as nn

from note_dataset import (
    BAR_POS_VOCAB, BEAT_IN_MEASURE_VOCAB, FLOAT_FIELDS, MEASURE_MOD,
    N_CHORD_QUALITY, N_CHORD_ROOT, N_ERAS, N_KEY_MODE, N_KEY_TONIC,
    N_PITCHES, N_ROMAN_DEGREE, N_SCALE_DEGREE, N_STYLE, N_TARGETS,
    N_VOICE_ROLES, SUBDIVISIONS_PER_BEAT,
)

LOGVAR_MIN = -7.0
LOGVAR_MAX = 3.0


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class PerformanceNet(nn.Module):
    def __init__(self, d_model=320, n_layers=8, n_heads=8, dim_feedforward=1280, dropout=0.1):
        super().__init__()

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

        in_dim = (64 + 32 + 32 + 16 + 16 + 16 + 16          # existing categoricals
                  + 16 + 8 + 16 + 16 + 16 + 12              # score categoricals
                  + 64)                                     # continuous
        self.in_dim = in_dim
        self.input_proj = nn.Linear(in_dim, d_model)

        # global style conditioning, broadcast to every note
        self.style_mlp = nn.Sequential(
            nn.Linear(N_STYLE, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 2 * N_TARGETS)  # mean + log-variance per target

    def _embed_score(self, batch):
        parts = [
            self.pitch_embed(batch["pitch"]),
            self.beat_embed(batch["beat_pos"]),
            self.bar_embed(batch["bar_pos"]),
            self.beat_in_measure_embed(batch["beat_in_measure"]),
            self.measure_embed(batch["measure_number"]),
            self.voice_role_embed(batch["voice_role"]),
            self.era_embed(batch["era"]),
            self.key_tonic_embed(batch["key_tonic"]),
            self.key_mode_embed(batch["key_mode"]),
            self.scale_degree_embed(batch["scale_degree"]),
            self.chord_root_embed(batch["chord_root"]),
            self.chord_quality_embed(batch["chord_quality"]),
            self.roman_degree_embed(batch["roman_degree"]),
            self.continuous_proj(torch.stack([batch[f] for f in FLOAT_FIELDS], dim=-1)),
        ]
        return torch.cat(parts, dim=-1)

    def _decode(self, feat, style, pad_mask):
        x = self.input_proj(feat) + self.style_mlp(style).unsqueeze(1)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        out = self.head(x)
        mean, logvar = out[..., :N_TARGETS], out[..., N_TARGETS:]
        return mean, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)

    def forward(self, batch):
        feat = self._embed_score(batch)
        mean, logvar = self._decode(feat, batch["style"], batch["pad_mask"])
        return {"mean": mean, "logvar": logvar}

    @torch.no_grad()
    def predict(self, batch, style=None):
        """Inference. style overrides batch['style'] if given; otherwise uses
        batch['style'] (neutral = standardized zeros). Returns (mean, logvar)."""
        feat = self._embed_score(batch)
        if style is None:
            style = batch.get("style")
            if style is None:
                style = torch.zeros(feat.shape[0], N_STYLE, device=feat.device)
        return self._decode(feat, style, batch["pad_mask"])
