"""finetune_cls.py
===================
Finetune **only the classifier head** of a pretrained ConvVAE model that
was trained with good reconstruction loss, keeping the encoder/decoder
frozen.  It relies entirely on the *original* dataset pipeline and
`ConvVAE` definition that already live in `train_v4.py` so you don’t have
 to copy code around.

Usage (example)
---------------
```bash
python finetune.py \
       --ckpt /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4/finetune/checkpoint/finetune_epoch1_acc0.069.pt \
       --data_root /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset \
       --epochs 20 --batch 128 --lr 3e-4
```

Command‑line flags
------------------
```
--ckpt        Path to the VAE checkpoint with good recon loss (REQUIRED)
--data_root   Directory that contains `train_data_128.npz` (defaults to cwd)
--epochs      Fine‑tuning epochs (default 15)
--batch       Batch‑size (default 128)
--lr          Learning‑rate for the classifier head (default 3e‑4)
--num_workers DataLoader workers (default 4)
--freeze_latent  Freeze quant_conv + post_quant_conv too (bool flag)
```

The script prints classification accuracy on the **validation split**
after each epoch and saves a new checkpoint named like
`finetune_epoch10_acc0.845.pt` in the same directory as the original
checkpoint.
"""
from __future__ import annotations
import argparse, os, time, math, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------
# 1. Re‑use everything from the original training script --------------
# ---------------------------------------------------------------------
import train_v4 as base                     # << your original script
CustomDataset     = base.CustomDataset     # dataset + transforms
train_transforms  = base.train_transforms
val_transforms    = base.val_transforms
ConvVAE           = base.ConvVAE

# If you also defined CenterLoss etc. in train_v4 and want to use it,
# uncomment the next line and adjust the loss block below.
# CenterLoss        = base.CenterLoss

# ---------------------------------------------------------------------
# 2. Helpers -----------------------------------------------------------
# ---------------------------------------------------------------------

def set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------
# 3. Training loop -----------------------------------------------------
# ---------------------------------------------------------------------

def finetune(model: ConvVAE, loader, val_loader, optimizer, device, epochs: int):
    best_acc = 0.0
    ckpt_dir = Path(args.ckpt).expanduser().resolve().parent

    for epoch in range(1, epochs + 1):
        model.eval()                 # keep BatchNorm in encoder frozen
        model.classifier.train()     # but let the head update

        pbar = tqdm(loader, desc=f"Finetune {epoch}/{epochs}")
        running_loss = running_acc = 0.0

        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                # Forward through frozen path to get latent z
                _, z, _, _, _ = model(x)
            logits = model.classifier(z.view(z.size(0), -1))
            loss   = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_acc  += (logits.argmax(1) == y).float().sum().item()
            pbar.set_postfix(loss=running_loss / len(loader.dataset),
                             acc =running_acc  / len(loader.dataset))

        # -------- validation --------
        val_acc = evaluate(model, val_loader, device)
        print(f"\nEpoch {epoch}: val acc = {val_acc:.4f}\n")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            out = ckpt_dir / f"finetune_epoch{epoch}_acc{val_acc:.3f}.pt"
            torch.save(model.state_dict(), out)
            print(f"  ↑ New best, saved to {out}")


def evaluate(model: ConvVAE, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device)
            _, z, _, _, _ = model(x)
            logits = model.classifier(z.view(z.size(0), -1))
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return correct / total


# ---------------------------------------------------------------------
# 4. Main --------------------------------------------------------------
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine‑tune ConvVAE classifier only")
    parser.add_argument("--ckpt", required=True, help="checkpoint to load")
    parser.add_argument("--data_root", default=".", help="root directory with train_data_128.npz")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_latent", action="store_true", help="also freeze quant + post‑quant convs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Dataset ----------
    data_root = Path(args.data_root).expanduser().resolve()
    full_ds   = CustomDataset(str(data_root / "train_data_128.npz"), transform=None)
    num_cls   = int(np.unique(full_ds.labels).shape[0])

    train_idx, val_idx = train_test_split(
        list(range(len(full_ds))), test_size=0.1, random_state=42, stratify=full_ds.labels)

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)

    # add transforms lazily via wrapper Dataset
    train_ds.dataset.transform = train_transforms
    val_ds.dataset.transform   = val_transforms

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ---------- Model ----------
    model = ConvVAE(input_channels=3, latent_channels=32, num_classes=num_cls)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.to(device)

    # ---------- Freeze encoder/decoder ----------
    freeze_parts = [model.encoder, model.decoder]
    if args.freeze_latent:
        freeze_parts += [model.quant_conv, model.post_quant_conv]
    for part in freeze_parts:
        set_requires_grad(part, False)
    # Classifier stays trainable

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-4)

    # ---------- Run ----------
    finetune(model, train_loader, val_loader, optimizer, device, args.epochs)
