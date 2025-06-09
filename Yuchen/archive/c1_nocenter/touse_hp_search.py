#!/usr/bin/env python
"""
Usage example:
python touse_hp_search.py \
  --data     /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_data_128.npz \
  --batch    128 \
  --trials   100 \
  --timeout  999999999 \
  --epochs   5
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import optuna
from importlib.machinery import SourceFileLoader

# ── Dynamically load your improved training script ──
SCRIPT_PATH = "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/c1_nocenter/c1_nocenter.py"
loader      = SourceFileLoader("train_mod", SCRIPT_PATH)
train_mod   = loader.load_module()

# ── Pull in exactly what we need from c1_improved_train.py ──
ConvVAE         = train_mod.ConvVAE
train_model     = train_mod.train_vae
validate        = train_mod.validate
CustomDataset   = train_mod.CustomDataset
train_transforms= train_mod.train_transforms
val_transforms  = train_mod.val_transforms
CenterLoss      = train_mod.CenterLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(npz_path, batch_size):
    ds = CustomDataset(npz_path, transform=None)
    idx = list(range(len(ds)))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.1, random_state=42, stratify=ds.labels
    )
    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx)
    train_ds.dataset.transform = train_transforms
    val_ds.dataset.transform   = val_transforms

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True
    )
    num_classes = int(np.unique(ds.labels).shape[0])
    return train_loader, val_loader, num_classes


def set_dropout(module, p):
    for m in module.modules():
        if isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d):
            m.p = p


def objective(trial, args):
    # Sample hyperparameters used by train_vae()
    latent  = trial.suggest_categorical("latent_channels", [8])
    alpha   = trial.suggest_float("alpha", 0.0001, 0.01, log=True)
    #gamma   = trial.suggest_float("gamma", 0.0001, 0.01, log=True)
    lr      = 1e-3
    #lr      = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    dropout_p_encode     = 0.1
    dropout_p_classifier = 0.5
    #dropout_p_encode     = trial.suggest_float("dropout_p_encode", 0.05, 0.5, log=True)
    #dropout_p_classifier = trial.suggest_float("dropout_p_classifier", 0.1, 0.7, log=True)

    # Data loaders
    train_loader, val_loader, num_classes = make_loaders(args.data, args.batch)

    # Build model & optimizer
    model = ConvVAE(input_channels=3, latent_channels=latent,
                    num_classes=num_classes).to(device)
    set_dropout(model.encoder,    p=dropout_p_encode)
    set_dropout(model.classifier, p=dropout_p_classifier)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Short loop with pruning
    for epoch in range(args.epochs):
        train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            num_epochs=1,
            alpha=alpha,
            gamma=gamma,
            resume=False
        )

        # Recreate center loss and compute validation metrics
        feat_dim    = latent * 32 * 32
        center_loss = CenterLoss(num_classes, feat_dim, device)
        kl_weight   = min(0.001, (epoch+1) / 20 * 0.001)
        stats       = validate(
            model,
            val_loader,
            device,
            kl_weight,
            alpha,
            gamma,
            center_loss
        )
        recon      = stats["recon"]
        class_loss = stats["class"]
        center_loss_val = stats["center"]

        # define a scalar score to minimize:
        score = recon + class_loss + center_loss_val
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    type=str, required=True,
                        help="path to train_data_128.npz")
    parser.add_argument("--batch",   type=int, default=128)
    parser.add_argument("--trials",  type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=3600,
                        help="seconds")
    parser.add_argument("--epochs",  type=int, default=3,
                        help="epochs per trial")
    args = parser.parse_args()

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        study_name="c1_improved",  # updated study name
        storage="sqlite:///convvae_hpo.db",
        load_if_exists=True
    )
    study.optimize(lambda t: objective(t, args),
                   n_trials=args.trials,
                   timeout=args.timeout)

    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
