'''
python hp_search_no_vgg.py \
  --data     /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_data_128.npz \
  --batch    128 \
  --trials   50 \
  --timeout  20000
'''
# hp_search.py

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import optuna
from importlib.machinery import SourceFileLoader
from torchvision import models

# ── Dynamically load your training script ──
SCRIPT_PATH = "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4_855_pixelshu_augmen/train_855_pixelaug.py"
loader = SourceFileLoader("train_mod", SCRIPT_PATH)
train_mod = loader.load_module()

# ── Extract necessary components ──
ConvVAE         = train_mod.ConvVAE
train_vae       = train_mod.train_vae
validate        = train_mod.validate
CustomDataset   = train_mod.CustomDataset
train_transforms= train_mod.train_transforms
val_transforms  = train_mod.val_transforms
CenterLoss      = train_mod.CenterLoss

# ── Prepare VGG16 & ImageNet stats for perceptual loss ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg16(pretrained=False).features[:16].to(device).eval()
for p in vgg.parameters():
    p.requires_grad = False
imagenet_mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
imagenet_std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)

def make_loaders(npz_path, batch_size):
    ds = CustomDataset(npz_path, transform=None)
    indices = list(range(len(ds)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=42, stratify=ds.labels
    )
    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx)
    train_ds.dataset.transform = train_transforms
    val_ds.dataset.transform   = val_transforms

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=1, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True),
        len(np.unique(ds.labels))
    )

def set_dropout(module, p):
    for m in module.modules():
        if isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d):
            m.p = p

def objective(trial, args):
    # ── Sample hyperparameters ──
    latent    = trial.suggest_categorical("latent_channels", [8])
    beta_max  = 0.0005
    #beta_max  = trial.suggest_float("beta_max",   1e-4, 1e-1, log=True)
    alpha     = trial.suggest_float("alpha",      0.001, 0.005, log=False)
    #alpha     = 0.0029355919056392875
    #gamma     = trial.suggest_float("gamma",      0.0005, 0.01, log=True)
    gamma     = 0.001
    #delta     = trial.suggest_float("delta",      1e-2, 10,   log=True)
    delta     = 1
    #lr        = trial.suggest_float("lr",         1e-5, 1e-2, log=True)
    lr        = 1e-3
    #dropout_p_encode = trial.suggest_float("dropout_p_encode",  0.0,  0.5)
    #dropout_p_classifier = trial.suggest_float("dropout_p_classifier",  0.0,  0.5)
    dropout_p_encode = 0.1
    dropout_p_classifier = 0.5
    # ── Data loaders ──
    train_loader, val_loader, num_classes = make_loaders(args.data, args.batch)

    # ── Model + optimizer ──
    model = ConvVAE(input_channels=3, latent_channels=latent, num_classes=num_classes).to(device)
    set_dropout(model.encoder,    dropout_p_encode)
    set_dropout(model.classifier, dropout_p_classifier)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # ── Short training loop with pruning ──
    for epoch in range(5):
        train_vae(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=1, beta_max=beta_max,
            alpha=alpha, gamma=gamma,
            vgg=vgg, imagenet_mean=imagenet_mean,
            imagenet_std=imagenet_std, delta=delta, save_ckpt=False
        )
        stats = validate(
            model, val_loader, device, beta_max, alpha, gamma,
            CenterLoss(num_classes, latent*32*32, device),
            vgg, imagenet_mean, imagenet_std, delta
        )
        recon = stats["recon"]
        acc   = 1.0 - stats["class"]
        score = recon * 2 / acc
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    type=str, required=True, help="path to train_data_128.npz")
    parser.add_argument("--batch",   type=int, default=128)
    parser.add_argument("--trials",  type=int, default=50)
    parser.add_argument("--timeout", type=int, default=3600, help="seconds")
    parser.add_argument("--epochs",  type=int, default=3, help="epochs per trial")
    args = parser.parse_args()

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(),
                                study_name="855_aug_pixel",                   # give it a name
    storage="sqlite:///convvae_hpo.db",             # store in this     file
    load_if_exists=True                           # re‐use existing trials
    )
    study.optimize(lambda t: objective(t, args), n_trials=args.trials, timeout=args.timeout)

    print("Best hyperparameters:")
    for k,v in study.best_params.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()

