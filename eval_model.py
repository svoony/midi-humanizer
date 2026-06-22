"""
Evaluation metrics for the PerformanceCVAE.

Keeps the training-time validation NLL (reconstruction NLL under the posterior
z) and adds two musically-motivated metrics that compare the model's
*predicted* performance against the held-out human performance:

  smoothed velocity correlation - Pearson r between the model's and the human's
      smoothed dynamic contour
  smoothed rubato correlation   - Pearson r between the model's and the human's
      smoothed onset-timing (rubato) contour

Why these, and why this way:
  * Prediction uses the prior / deadpan z (model.predict, z=0) - i.e. what the
    model would actually produce at inference with no access to the
    performance. The kept NLL, by contrast, runs the posterior path (z encodes
    the real performance), so it measures reconstruction, not prediction.
  * Smoothing (a short moving average over the note sequence) targets the
    phrase-level shape rather than per-note jitter - what matters perceptually
    and what a pointwise NLL is blind to.
  * Correlation is affine-invariant, so it's computed directly in standardized
    target space (both predicted and actual are already standardized).

Correlations are computed per 192-note window, then averaged across windows.
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import PerformanceNet
from note_dataset import NoteRegressionDataset, TARGET_ORDER, collate_fn
from train import gaussian_nll

VEL_IDX = TARGET_ORDER.index("velocity")
TIMING_IDX = TARGET_ORDER.index("log_ioi_ratio")


def moving_average(x, w):
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-8:
        return None  # one side is flat -> correlation undefined
    return float((a * b).sum() / denom)


def _window_corrs(pred, actual, pad, idx, smooth, out_smoothed, out_raw):
    for i in range(pred.shape[0]):
        valid = ~pad[i]
        if valid.sum() < smooth + 2:
            continue
        p, a = pred[i, valid, idx], actual[i, valid, idx]
        r_raw = pearson(p, a)
        if r_raw is not None:
            out_raw.append(r_raw)
        r_sm = pearson(moving_average(p, smooth), moving_average(a, smooth))
        if r_sm is not None:
            out_smoothed.append(r_sm)


@torch.no_grad()
def evaluate(checkpoint_path="checkpoints/best.pt", split="validation", smooth=8, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PerformanceNet(**ckpt.get("config", {})).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = NoteRegressionDataset(split=split, window_notes=192, stride=192)
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)

    nll_sum, n_batches = 0.0, 0
    vel_sm, vel_raw, rub_sm, rub_raw = [], [], [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}

        # kept validation NLL, same as train.py
        out = model(batch)
        nll = gaussian_nll(out["mean"], out["logvar"], batch["y"], batch["pad_mask"], beta=0.0)
        nll_sum += nll.item()
        n_batches += 1

        # predicted performance conditioned on the true per-piece style
        mean, _ = model.predict(batch, style=batch["style"])
        pred = mean.cpu().numpy()
        actual = batch["y"].cpu().numpy()
        pad = batch["pad_mask"].cpu().numpy()

        _window_corrs(pred, actual, pad, VEL_IDX, smooth, vel_sm, vel_raw)
        _window_corrs(pred, actual, pad, TIMING_IDX, smooth, rub_sm, rub_raw)

    def summ(xs):
        return f"mean {np.mean(xs):+.4f}  median {np.median(xs):+.4f}  (n={len(xs)})"

    print(f"=== {split} | {checkpoint_path} | smoothing {smooth} notes ===")
    print(f"validation NLL (posterior recon):  {nll_sum / n_batches:.4f}")
    print(f"smoothed velocity correlation:     {summ(vel_sm)}")
    print(f"smoothed rubato   correlation:     {summ(rub_sm)}")
    print(f"  (unsmoothed velocity:            {summ(vel_raw)})")
    print(f"  (unsmoothed rubato:              {summ(rub_raw)})")
    return {
        "val_nll": nll_sum / n_batches,
        "vel_corr_smoothed": float(np.mean(vel_sm)),
        "rubato_corr_smoothed": float(np.mean(rub_sm)),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--split", default="validation")
    p.add_argument("--smooth", type=int, default=8)
    args = p.parse_args()
    evaluate(args.checkpoint, args.split, args.smooth)
