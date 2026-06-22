"""
Training loop for the gen-3 PerformanceNet (style-conditioned, no VAE).

Loss = heteroscedastic beta-NLL on the four standardized targets + a
multi-scale timing-trajectory term that matches the *smoothed* local-tempo
prediction to the human at several scales. The trajectory term is the lever
for phrase-level rubato: the per-note NLL alone is blind to phrase shape (the
exact gap the eval surfaced), so we optimize the smoothed timing curve
directly.

Evaluation is in the loop: every epoch we compute the smoothed velocity and
rubato correlation against the held-out human performance (the metrics we
actually care about) and select the checkpoint on their mean - NOT on NLL,
which we found goes the wrong way. A short L1 warmup stabilises the means
before the NLL variance term switches on.
"""
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import PerformanceNet
from note_dataset import NoteRegressionDataset, TARGET_ORDER, N_TARGETS, collate_fn

CHECKPOINT_DIR = "checkpoints"
LOG_PATH = os.path.join(CHECKPOINT_DIR, "train_log.csv")

TIMING_IDX = TARGET_ORDER.index("log_ioi_ratio")
VEL_IDX = TARGET_ORDER.index("velocity")


def gaussian_nll(mean, logvar, target, pad_mask, beta=0.5):
    var = logvar.exp()
    nll = 0.5 * (logvar + (target - mean) ** 2 / var)
    if beta > 0:
        nll = nll * (var.detach() ** beta)
    valid = (~pad_mask).unsqueeze(-1).float()
    nll = nll * valid
    n_valid = valid.sum().clamp(min=1.0)
    return nll.sum() / (n_valid * N_TARGETS)


def masked_l1_mean(mean, target, pad_mask):
    valid = (~pad_mask).unsqueeze(-1).float()
    return ((mean - target).abs() * valid).sum() / (valid.sum().clamp(min=1.0) * N_TARGETS)


def _smooth(sig, valid, k):
    """Mask-normalized moving average over time. sig, valid: (B, T); odd k."""
    w = torch.ones(1, 1, k, device=sig.device) / k
    num = F.conv1d((sig * valid).unsqueeze(1), w, padding=k // 2).squeeze(1)
    den = F.conv1d(valid.unsqueeze(1), w, padding=k // 2).squeeze(1).clamp(min=1e-6)
    return num / den


def timing_trajectory_loss(mean, target, pad_mask, scales=(5, 17)):
    """MSE between the smoothed predicted and target local-tempo curves, at
    several scales - the phrase-rubato lever."""
    m, t = mean[..., TIMING_IDX], target[..., TIMING_IDX]
    valid = (~pad_mask).float()
    loss = 0.0
    for k in scales:
        diff = (_smooth(m, valid, k) - _smooth(t, valid, k)) ** 2 * valid
        loss = loss + diff.sum() / valid.sum().clamp(min=1.0)
    return loss / len(scales)


def make_loader(split, batch_size, window_notes, device, shuffle, augment=False):
    ds = NoteRegressionDataset(split=split, window_notes=window_notes, stride=window_notes, augment=augment)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
        num_workers=2, persistent_workers=True, pin_memory=(device == "cuda"),
    )
    return ds, dl


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return None if d < 1e-8 else float((a * b).sum() / d)


def _np_smooth(x, w):
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


@torch.no_grad()
def evaluate(model, dl, device, smooth=8):
    """Validation NLL + smoothed velocity/rubato correlation (true-style
    prediction vs the human performance)."""
    model.eval()
    nll_sum, nb = 0.0, 0
    vel_c, rub_c = [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        nll_sum += gaussian_nll(out["mean"], out["logvar"], batch["y"], batch["pad_mask"], beta=0.0).item()
        nb += 1
        mean, _ = model.predict(batch, style=batch["style"])
        pred, actual, pad = mean.cpu().numpy(), batch["y"].cpu().numpy(), batch["pad_mask"].cpu().numpy()
        for i in range(pred.shape[0]):
            v = ~pad[i]
            if v.sum() < smooth + 2:
                continue
            for idx, store in ((VEL_IDX, vel_c), (TIMING_IDX, rub_c)):
                r = _pearson(_np_smooth(pred[i, v, idx], smooth), _np_smooth(actual[i, v, idx], smooth))
                if r is not None:
                    store.append(r)
    model.train()
    vel = float(np.mean(vel_c)) if vel_c else 0.0
    rub = float(np.mean(rub_c)) if rub_c else 0.0
    return nll_sum / nb, vel, rub


def save_checkpoint(path, model, optimizer, epoch, step, best_score, train_ds, config):
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "step": step, "best_score": best_score, "config": config,
        "target_order": TARGET_ORDER, "target_mean": train_ds.target_mean, "target_std": train_ds.target_std,
        "style_mean": train_ds.style_mean, "style_std": train_ds.style_std,
    }, path)


def train(max_epochs=300, batch_size=32, window_notes=192, lr=3e-4,
          beta_nll=0.5, traj_weight=1.0, l1_warmup_epochs=2, patience=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_ds, train_dl = make_loader("train", batch_size, window_notes, device, shuffle=True, augment=True)
    val_ds, val_dl = make_loader("validation", batch_size, window_notes, device, shuffle=False)
    print(f"train windows: {len(train_ds)} | validation windows: {len(val_ds)}")

    config = dict(d_model=320, n_layers=8, n_heads=8, dim_feedforward=1280)
    model = PerformanceNet(**config).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "train_loss", "val_nll",
                                 "val_vel_corr", "val_rubato_corr", "expr_score", "elapsed_s"])

    step, best_score, no_improve, t0 = 0, -1.0, 0, time.time()
    for epoch in range(max_epochs):
        warmup = epoch < l1_warmup_epochs
        running, nb = 0.0, 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            if warmup:
                loss = masked_l1_mean(out["mean"], batch["y"], batch["pad_mask"])
            else:
                nll = gaussian_nll(out["mean"], out["logvar"], batch["y"], batch["pad_mask"], beta=beta_nll)
                traj = timing_trajectory_loss(out["mean"], batch["y"], batch["pad_mask"])
                loss = nll + traj_weight * traj
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item(); nb += 1; step += 1

        train_loss = running / nb
        val_nll, vel_corr, rub_corr = evaluate(model, val_dl, device)
        expr = 0.5 * (vel_corr + rub_corr)
        elapsed = time.time() - t0
        print(f"=== epoch {epoch:3d} | train {train_loss:.4f} | val_nll {val_nll:.4f} | "
              f"vel_corr {vel_corr:.4f} rubato_corr {rub_corr:.4f} | expr {expr:.4f} | {elapsed:.0f}s ===")
        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([epoch, step, f"{train_loss:.6f}", f"{val_nll:.6f}",
                                     f"{vel_corr:.6f}", f"{rub_corr:.6f}", f"{expr:.6f}", f"{elapsed:.1f}"])

        save_checkpoint(os.path.join(CHECKPOINT_DIR, "last.pt"), model, optimizer, epoch, step, best_score, train_ds, config)
        if not warmup and expr > best_score + 1e-4:
            best_score = expr
            no_improve = 0
            save_checkpoint(os.path.join(CHECKPOINT_DIR, "best.pt"), model, optimizer, epoch, step, best_score, train_ds, config)
            print(f"  -> new best expr_score {best_score:.4f}, saved checkpoints/best.pt")
        elif not warmup:
            no_improve += 1
            if no_improve >= patience:
                print(f"no improvement for {patience} epochs, stopping early")
                break

    print(f"training done. best expr_score: {best_score:.4f}")
    return model


if __name__ == "__main__":
    train()
