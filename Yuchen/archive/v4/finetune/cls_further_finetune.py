#!/usr/bin/env python3
"""cls_further_finetune.py — Resume two-layer MLP head fine-tuning from checkpoint
==============================================================================
This script picks up exactly where your last two-layer MLP fine-tune left off.
It loads the pretrained MLP head from your checkpoint, freezes the ConvVAE
encoder/decoder/bottleneck, and continues training only the MLP head.

Usage
-----
```bash
python cls_further_finetune.py \
  --ckpt /path/to/cls_finetune_epoch03_acc0.071.pt \
  --data_root /path/to/dataset \
  --batch_size 128 --lr 3e-4 --epochs 15 --num_workers 1
```

Command-line arguments
----------------------
--ckpt         Path to your two-layer MLP checkpoint (REQUIRED)
--data_root    Directory containing train_data_128.npz (default: cwd)
--batch_size   Batch size (default: 128)
--lr           Learning rate for MLP head (default: 3e-4)
--epochs       Number of fine-tune epochs (default: 15)
--num_workers  DataLoader workers (default: 1)

"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm

# Import original definitions
import train_v4 as base
CustomDataset    = base.CustomDataset
train_transforms = base.train_transforms
val_transforms   = base.val_transforms
ConvVAE          = base.ConvVAE

# ---------------------------------------------------------------------
# Utility to freeze modules
# ---------------------------------------------------------------------
def set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag

# ---------------------------------------------------------------------
# Two-layer MLP head matching your checkpoint
# ---------------------------------------------------------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------------------------------------------------------------------
# Validation routine
# ---------------------------------------------------------------------
def evaluate(model: ConvVAE, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, z, _, _, _ = model(x)
            logits = model.classifier(z.view(z.size(0), -1))
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Resume two-layer MLP head fine-tune from checkpoint"
    )
    parser.add_argument("--ckpt", required=True, help="path to MLP checkpoint")
    parser.add_argument("--data_root", default=".", help="directory with train_data_128.npz")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Data ----------
    data_root = Path(args.data_root).expanduser().resolve()
    full_ds = CustomDataset(str(data_root / "train_data_128.npz"), transform=None)
    num_classes = int(np.unique(full_ds.labels).shape[0])

    idxs = list(range(len(full_ds)))
    train_idx, val_idx = train_test_split(
        idxs, test_size=0.1, random_state=42, stratify=full_ds.labels
    )
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    train_ds.dataset.transform = train_transforms
    val_ds.dataset.transform   = val_transforms

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # ---------- Model ----------
    model = ConvVAE(input_channels=3, latent_channels=32, num_classes=num_classes)
    # swap in MLPHead to match checkpoint
    head_dim = model.latent_channels * 16 * 16
    model.classifier = MLPHead(head_dim, num_classes).to(device)

    # freeze all but classifier
    set_requires_grad(model.encoder,      False)
    set_requires_grad(model.quant_conv,   False)
    set_requires_grad(model.post_quant_conv, False)
    set_requires_grad(model.decoder,      False)

    # load full state (includes MLPHead weights)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=args.lr, weight_decay=1e-4
    )

    best_acc = 0.0
    ckpt_dir = Path(args.ckpt).expanduser().resolve().parent

    # ---------- Training ----------
    for epoch in range(1, args.epochs + 1):
        # encoder/decoder frozen    → .eval() freezes BN
        model.eval()
        # but train MLP head
        model.classifier.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                _, z, _, _, _ = model(x)
            logits = model.classifier(z.view(z.size(0), -1))
            loss   = F.cross_entropy(logits, y, label_smoothing=0.05)
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(1) == y).float().mean().item()
            pbar.set_postfix(acc=f"{acc:.3f}")

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: val acc = {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            out = ckpt_dir / f"resume_mlp_epoch{epoch:02d}_acc{val_acc:.3f}.pt"
            torch.save(model.state_dict(), out)
            print(f"  ↑ New best, saved to {out}")

if __name__ == "__main__":
    main()
