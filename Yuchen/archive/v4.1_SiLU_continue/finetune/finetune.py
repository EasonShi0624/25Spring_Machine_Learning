"""
python finetune_probe.py \
  --checkpoint /path/to/checkpoint_epoch350.pt \
  --data-dir /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset \
  --npz   train_data_128.npz \
  --batch 128 \
  --epochs  50
"""
# finetune_probe.py
import os, argparse, random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.functional import normalize
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms, models

# ----------------------------
# 1. Dataset & Transforms
# ----------------------------
class CustomDataset(Dataset):
    """As in original training script” :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}"""
    def __init__(self, npz_file, transform=None):
        data = np.load(npz_file)
        self.images = data['images']
        self.labels = data['labels']
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = (self.images[idx].transpose(1,2,0)*255).astype(np.uint8)
        img = Image.fromarray(img)
        img = self.transform(img) if self.transform else transforms.ToTensor()(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
])
val_tf = transforms.ToTensor()

# ----------------------------
# 2. Centre Loss
# ----------------------------
class CenterLoss(nn.Module):
    """Parametric centre loss” :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}"""
    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device))
    def forward(self, features, labels):
        c = self.centers[labels]
        return 0.5 * (features - c).pow(2).sum() / features.size(0)

# ----------------------------
# 3. Load VAE model
# ----------------------------
from train_v4_SiLU_tuning_continue import ConvVAE  # your original model class :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}

# ----------------------------
# 4. SupCon Loss
# ----------------------------
class SupConLoss(nn.Module):
    def __init__(self, T=0.07): super().__init__(); self.T = T
    def forward(self, features, labels):
        f = normalize(features, dim=1)
        sim = torch.mm(f, f.t()) / self.T
        mask = (labels.unsqueeze(1)==labels.unsqueeze(0)).float()
        mask.fill_diagonal_(0)
        exp = torch.exp(sim)
        log_prob = sim - torch.log(exp.sum(1, keepdim=True))
        loss = -(mask * log_prob).sum(1) / mask.sum(1)
        return loss.mean()

# ----------------------------
# 5. Argument parsing
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to VAE checkpoint (.pt) with model_state_dict")
parser.add_argument("--data-dir",   type=str, default=".", help="dataset directory")
parser.add_argument("--npz",        type=str, required=True, help=".npz filename")
parser.add_argument("--batch",      type=int, default=128)
parser.add_argument("--epochs",     type=int, default=50)
parser.add_argument("--lr",         type=float, default=3e-4)
parser.add_argument("--supcon",     type=float, default=0.1)
parser.add_argument("--center",     type=float, default=0.001)
parser.add_argument("--freeze-enc", action="store_true",
                    help="freeze all but last 2 encoder blocks")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.chdir(args.data_dir)

# --- build data loaders ---
full = CustomDataset(args.npz, transform=None)
ncls = int(np.unique(full.labels).shape[0])
idx_train, idx_val = train_test_split(
    list(range(len(full))), test_size=0.1, random_state=42,
    stratify=full.labels
)
train_ds, val_ds = Subset(full, idx_train), Subset(full, idx_val)
train_ds.dataset.transform, val_ds.dataset.transform = train_tf, val_tf
train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=4, pin_memory=True)

# --- instantiate model & load checkpoint ---
model = ConvVAE(input_channels=3, latent_channels=8, num_classes=ncls).to(device)
ckpt = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
print(f"Loaded checkpoint from {args.checkpoint}")

# --- freeze decoder & (optionally) early encoder ---
for p in model.decoder.parameters():          p.requires_grad = False
for p in model.post_quant_conv.parameters():  p.requires_grad = False
if args.freeze_enc:
    for name, p in model.encoder.named_parameters():
        if not name.startswith(("3", "4")):  p.requires_grad = False

# --- add projection head + probe ---
feat_dim = model.latent_channels * 32 * 32
model.projector = nn.Sequential(
    nn.Flatten(),
    nn.Linear(feat_dim, 512), nn.SiLU(),
    nn.Linear(512, 128)
).to(device)
model.probe = nn.Linear(128, ncls).to(device)

# --- loss fns & optimizer ---
supcon = SupConLoss().to(device)
center = CenterLoss(ncls, 128, device)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr, weight_decay=1e-4
)

# ----------------------------
# 6. Training loop
# ----------------------------
best_acc = 0.0
for epoch in range(1, args.epochs+1):
    model.train()
    tot_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():                   # keep recon path fixed
            x_rec, z, _, _, _ = model(x)
        f = model.projector(z.view(z.size(0), -1))
        logits = model.probe(f)

        ce  = F.cross_entropy(logits, y)
        sc  = supcon(f, y)
        cc  = center(f, y)
        loss = ce + args.supcon*sc + args.center*cc

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), 5.0
        )
        optimizer.step()
        tot_loss += loss.item() * x.size(0)
    avg_loss = tot_loss / len(train_loader.dataset)

    # --- validation ---
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, z, _, _, _ = model(x)
            f = model.projector(z.view(z.size(0), -1))
            pred = model.probe(f).argmax(1)
            correct += (pred == y).sum().item()
    acc = correct / len(val_loader.dataset)
    print(f"[Epoch {epoch}] TrainLoss={avg_loss:.4f}  ValAcc={acc:.4f}")

    # --- checkpoint best probe only ---
    if acc > best_acc:
        best_acc = acc
        torch.save({"probe_state_dict": model.probe.state_dict(),
                    "proj_state_dict": model.projector.state_dict(),
                    "epoch": epoch},
                   "best_probe.pt")
        print(f"→ Saved best_probe.pt (acc={best_acc:.4f})")
