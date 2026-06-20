"""
Transformer encoder over a window of score notes. Each note's features
(pitch, metric position, IOI, duration) are embedded/projected into a
shared d_model space; standard sinusoidal positional encoding adds note-
sequence order. Four linear heads predict timing offset, log duration
ratio, velocity, and sustain pedal value per note.
"""
import math

import torch
import torch.nn as nn

from note_dataset import BAR_SUBDIVISIONS, N_PITCHES, SUBDIVISIONS_PER_BEAT


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


class PerformanceRegressor(nn.Module):
    def __init__(self, d_model=256, n_layers=6, n_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        embed_dim = d_model // 4

        self.pitch_embed = nn.Embedding(N_PITCHES, embed_dim)
        self.beat_embed = nn.Embedding(SUBDIVISIONS_PER_BEAT, embed_dim // 2)
        self.bar_embed = nn.Embedding(BAR_SUBDIVISIONS, embed_dim // 2)
        self.continuous_proj = nn.Linear(3, embed_dim)  # rel_step, ioi, dur_grid

        in_dim = embed_dim + embed_dim // 2 + embed_dim // 2 + embed_dim
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Linear(d_model, 4)  # timing_offset, log_dur_ratio, velocity, pedal

    def forward(self, batch):
        pitch = self.pitch_embed(batch["pitch"])
        beat = self.beat_embed(batch["beat_pos"])
        bar = self.bar_embed(batch["bar_pos"])
        continuous = self.continuous_proj(
            torch.stack([batch["rel_step"], batch["ioi"], batch["dur_grid"]], dim=-1)
        )

        x = torch.cat([pitch, beat, bar, continuous], dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        x = self.encoder(x, src_key_padding_mask=batch["pad_mask"])
        return self.head(x)
