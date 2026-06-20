"""
Full training loop for the note-wise performance regressor: trains over
epochs with periodic validation, checkpointing (last + best), CSV logging,
and early stopping on validation loss plateau.
"""
import csv
import os
import time

import torch
from torch.utils.data import DataLoader

from model import PerformanceRegressor
from note_dataset import NoteRegressionDataset, collate_fn

CHECKPOINT_DIR = "checkpoints"
LOG_PATH = os.path.join(CHECKPOINT_DIR, "train_log.csv")


N_TARGETS = 4  # timing_offset, log_dur_ratio, velocity, pedal


def masked_l1_loss(pred, target, pad_mask):
    valid = (~pad_mask).unsqueeze(-1).float()
    loss_per_target = (pred - target).abs() * valid
    denom = valid.sum().clamp(min=1.0)
    total = loss_per_target.sum() / (denom * N_TARGETS)
    per_dim = loss_per_target.sum(dim=(0, 1)) / denom.squeeze(-1).sum().clamp(min=1.0)
    return total, per_dim


def make_loader(split, batch_size, window_notes, device, shuffle):
    ds = NoteRegressionDataset(split=split, window_notes=window_notes, stride=window_notes)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=2, persistent_workers=True,
        pin_memory=(device == "cuda"),
    )
    return ds, dl


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    total_loss, total_per_dim, n_batches = 0.0, torch.zeros(N_TARGETS), 0
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch)
        loss, per_dim = masked_l1_loss(pred, batch["y"], batch["pad_mask"])
        total_loss += loss.item()
        total_per_dim += per_dim.cpu()
        n_batches += 1
    model.train()
    return total_loss / n_batches, total_per_dim / n_batches


def save_checkpoint(path, model, optimizer, epoch, step, best_val_loss):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
    }, path)


def train(
    max_epochs=200,
    batch_size=32,
    window_notes=192,
    lr=3e-4,
    log_every=50,
    patience=15,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_ds, train_dl = make_loader("train", batch_size, window_notes, device, shuffle=True)
    val_ds, val_dl = make_loader("validation", batch_size, window_notes, device, shuffle=False)
    print(f"train windows: {len(train_ds)} | validation windows: {len(val_ds)}")

    model = PerformanceRegressor().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "train_loss", "val_loss",
                                 "val_timing", "val_dur", "val_vel", "val_pedal", "elapsed_s"])

    step = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    t0 = time.time()

    for epoch in range(max_epochs):
        running_loss, n_batches = 0.0, 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch)
            loss, per_dim = masked_l1_loss(pred, batch["y"], batch["pad_mask"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            step += 1

            if step % log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"epoch {epoch:3d} step {step:6d} | train_loss {loss.item():.4f} "
                    f"(timing {per_dim[0].item():.4f}, dur {per_dim[1].item():.4f}, "
                    f"vel {per_dim[2].item():.4f}, pedal {per_dim[3].item():.4f}) | {elapsed:.0f}s"
                )

        train_loss = running_loss / n_batches
        val_loss, val_per_dim = evaluate(model, val_dl, device)
        elapsed = time.time() - t0

        print(
            f"=== epoch {epoch:3d} done | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} (timing {val_per_dim[0]:.4f}, "
            f"dur {val_per_dim[1]:.4f}, vel {val_per_dim[2]:.4f}, "
            f"pedal {val_per_dim[3]:.4f}) | {elapsed:.0f}s ==="
        )
        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, step, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{val_per_dim[0]:.6f}", f"{val_per_dim[1]:.6f}", f"{val_per_dim[2]:.6f}",
                f"{val_per_dim[3]:.6f}", f"{elapsed:.1f}",
            ])

        save_checkpoint(os.path.join(CHECKPOINT_DIR, "last.pt"), model, optimizer, epoch, step, best_val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(os.path.join(CHECKPOINT_DIR, "best.pt"), model, optimizer, epoch, step, best_val_loss)
            print(f"  -> new best val_loss {best_val_loss:.4f}, saved checkpoints/best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"no improvement for {patience} epochs, stopping early")
                break

    print(f"training done. best val_loss: {best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    train()
