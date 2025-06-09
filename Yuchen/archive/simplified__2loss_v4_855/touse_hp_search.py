#!/usr/bin/env python
"""
Usage example:
python touse_hp_search.py \
  --data     /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_data_128.npz \
  --batch    128 \
  --trials   200 \
  --timeout  9999999 \
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

# ── Dynamically load your simplified training script ──
SCRIPT_PATH = "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/simplified__2loss_v4_855/simplified_train_855_pixelaug.py"
loader      = SourceFileLoader("train_mod", SCRIPT_PATH)
train_mod   = loader.load_module()

# ── Pull in exactly what we need ──
ConvVAE         = train_mod.ConvVAE
train_model     = train_mod.train_model
validate        = train_mod.validate
CustomDataset   = train_mod.CustomDataset
train_transforms= train_mod.train_transforms
val_transforms  = train_mod.val_transforms

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
    # ── Sample only the hyperparameters your train_model() expects ──
    latent  = trial.suggest_categorical("latent_channels", [8])
    #alpha   = trial.suggest_float("alpha", 0.0005, 0.005, log=True)
    alpha   = 0.0046011068847746925
    #gamma   = trial.suggest_float("gamma", 0.0005, 0.005, log=True)
    gamma   = 0.001235889615945341
    lr      = 1e-4
    dropout_p_encode  = trial.suggest_float("dropout_p_encode", 0.05, 0.5, log=True)
    dropout_p_classifier  = trial.suggest_float("dropout_p_classifier", 0.1, 0.7, log=True)
    # ── Data loaders ──
    train_loader, val_loader, num_classes = make_loaders(args.data, args.batch)

    # ── Build model & optimizer ──
    model = ConvVAE(input_channels=3, latent_channels=latent,
                    num_classes=num_classes).to(device)
    # (keep the same dropout settings as your simplified script)
    set_dropout(model.encoder,    p=dropout_p_encode)
    set_dropout(model.classifier, p=dropout_p_classifier)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # ── Short loop with pruning ──
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
            scheduler=None,       # your simplified_train has an optional scheduler—None is fine
            save_ckpt=False
        )

        stats = validate(model, val_loader, device)
        recon      = stats["recon"]
        class_loss = stats["class"]
        # define a scalar score to minimize:
        score = recon + class_loss
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    type=str, required=True,
                        help="path to train_data_128.npz")
    parser.add_argument("--batch",   type=int, default=128)
    parser.add_argument("--trials",  type=int, default=50)
    parser.add_argument("--timeout", type=int, default=3600,
                        help="seconds")
    parser.add_argument("--epochs",  type=int, default=3,
                        help="epochs per trial")
    args = parser.parse_args()

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        study_name="Simplifier855",
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
