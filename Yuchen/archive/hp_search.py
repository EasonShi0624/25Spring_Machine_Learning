'''
python hp_search.py \
  --data     /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_data_128.npz \
  --batch    128 \
  --trials   100 \
  --timeout  16000
'''
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import optuna
from importlib.machinery import SourceFileLoader

# ── Load your existing train script as a module ──
loader = SourceFileLoader("train_mod", "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4_SiLU_8latent_pretrained/train_v4_SiLU_8latent_8*32*32.py")
train_mod = loader.load_module()

# Extract items
ConvVAE         = train_mod.ConvVAE
train_vae       = train_mod.train_vae
validate        = train_mod.validate
CustomDataset   = train_mod.CustomDataset
train_transforms= train_mod.train_transforms
val_transforms  = train_mod.val_transforms
CenterLoss      = train_mod.CenterLoss
vgg             = train_mod.vgg if hasattr(train_mod, "vgg") else None
imagenet_mean   = getattr(train_mod, "imagenet_mean", None)
imagenet_std    = getattr(train_mod, "imagenet_std", None)

def make_loaders(npz_path, batch_size):
    ds = CustomDataset(npz_path, transform=None)
    N = len(ds)
    idx = list(range(N))
    train_idx, val_idx = train_test_split(idx, test_size=0.1,
                                          random_state=42, stratify=ds.labels)
    train_ds = Subset(ds, train_idx); val_ds = Subset(ds, val_idx)
    train_ds.dataset.transform = train_transforms
    val_ds.dataset.transform   = val_transforms
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
        len(np.unique(ds.labels))
    )

def set_dropout(module, p):
    for m in module.modules():
        if isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d):
            m.p = p

def objective(trial, args):
    # 1) Sample HPO params
    latent    = trial.suggest_categorical("latent_channels", [8])
    beta_max  = trial.suggest_loguniform("beta_max",   1e-4, 1e-1)
    alpha     = trial.suggest_loguniform("alpha",      1e-4, 1e-1)
    gamma     = trial.suggest_loguniform("gamma",      1e-4, 1e-1)
    delta     = trial.suggest_loguniform("delta",      1e-2, 10)
    lr        = trial.suggest_loguniform("lr",         1e-5, 1e-2)
    dropout_p = trial.suggest_uniform("dropout_p",     0.0,  0.5)

    # 2) DataLoaders
    train_loader, val_loader, num_classes = make_loaders(args.data, args.batch)

    # 3) Model + optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvVAE(input_channels=3, latent_channels=latent, num_classes=num_classes).to(device)
    set_dropout(model.encoder,    dropout_p)
    set_dropout(model.classifier, dropout_p)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # 4) Short training + pruning
    for epoch in range(10):
        train_vae(model, train_loader, val_loader, optimizer, device,
                  num_epochs=1, beta_max=beta_max,
                  alpha=alpha, gamma=gamma,
                  vgg=vgg,
                  imagenet_mean=imagenet_mean,
                  imagenet_std=imagenet_std,
                  delta=delta)
        stats = validate(
            model, val_loader, device,
            beta_max, alpha, gamma,
            CenterLoss(num_classes, latent*32*32, device),
            vgg, imagenet_mean, imagenet_std, delta
        )
        recon = stats["recon"]
        # Assuming stats["class"] is classification loss → approximate acc
        acc   = 1.0 - stats["class"]  
        score = recon / acc
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return score

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    type=str, required=True)
    p.add_argument("--batch",   type=int, default=128)
    p.add_argument("--trials",  type=int, default=50)
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(lambda t: objective(t, args),
                   n_trials=args.trials,
                   timeout=args.timeout)

    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
