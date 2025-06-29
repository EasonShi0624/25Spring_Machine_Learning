import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from tqdm import tqdm

"""
python VE_cls_continue.py --resume /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v5_VE_cls_only/checkpoint/firstround_checkpoint_epoch178.pt
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--resume",
    type=str,
    default="",
    help="path to checkpoint .pt file to resume from"
)
args = parser.parse_args()

# ----------------------------
# 1. Dataset & Augmentations
# ----------------------------
class CustomDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        data = np.load(npz_file)
        self.images = data['images']   # (N, 3, 128, 128), float32 [0,1]
        self.labels = data['labels']   # (N,), int
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_np = (self.images[idx].transpose(1,2,0) * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])
val_transforms = transforms.Compose([transforms.ToTensor(),])

os.chdir("/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset")
full_dataset = CustomDataset("train_data_128.npz", transform=None)
num_classes = int(np.unique(full_dataset.labels).shape[0])

train_idx, val_idx = train_test_split(
    list(range(len(full_dataset))), test_size=0.1,
    random_state=42, stratify=full_dataset.labels
)
train_subset = Subset(full_dataset, train_idx)
val_subset   = Subset(full_dataset, val_idx)
train_subset.dataset.transform = train_transforms
val_subset.dataset.transform   = val_transforms

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_subset,   batch_size=128, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

# ----------------------------
# 2. Center Loss (parametric)
# ----------------------------
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device))

    def forward(self, features, labels):  # features: [B, feat_dim]
        batch_centers = self.centers[labels]
        diff = features - batch_centers
        return 0.5 * diff.pow(2).sum() / features.size(0)

# ----------------------------
# 3. ConvAE Model (Autoencoder)
# ----------------------------
class ConvAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels
        # Encoder: 128->32 spatial
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),  # 128��64
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.SiLU(),             # 64��32
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.SiLU(), nn.Dropout2d(0.25),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.SiLU(),             # refine 32��32
        )
        # Project to deterministic latent
        self.quant_conv = nn.Conv2d(256, latent_channels, 1)
        # Project back before decoder
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)
        # Decoder: 32->128 spatial
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),  # 32��64
            nn.ConvTranspose2d(64, input_channels, 4, 2, 1), nn.Sigmoid(),         # 64��128
        )
        feat_dim = latent_channels * 32 * 32
        self.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(feat_dim, num_classes))

    def preprocess(self, x):
        return 2 * x - 1  # [0,1] �� [-1,1]

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        z = self.quant_conv(h)
        return z

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        logits = self.classifier(z.view(z.size(0), -1))
        return x_rec, z, logits

# ----------------------------
# 4. Loss & Validation
# ----------------------------
def ae_loss(x, x_recon):
    return F.mse_loss(x, x_recon, reduction='mean')

# ----------------------------
# 4. Loss & Validation
# ----------------------------
def validate(model, loader, device):
    model.eval()
    total_recon, total_correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_rec, z, logits = model(x)
            recon = F.mse_loss(x_rec, x, reduction='sum')
            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()
            bsz = x.size(0)
            total_recon += recon.item()
            total_correct += correct
            total_n += bsz
    avg_recon = total_recon / total_n
    acc = total_correct / total_n
    return avg_recon, acc


def train(model, train_loader, val_loader, optimizer, scheduler, device,
          num_epochs, alpha):
    best_recon = float('inf')

    for epoch in range(1, num_epochs+1):
        model.train()
        running_recon, running_cls, running_n = 0.0, 0.0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for i, (x, y) in enumerate(loop):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            x_rec, z, logits = model(x)
            recon_loss = F.mse_loss(x_rec, x, reduction='mean')
            cls_loss = F.cross_entropy(logits, y)
            loss = recon_loss + alpha * cls_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            # scheduler step (per batch) for CosineAnnealingWarmRestarts
            scheduler.step(epoch + i / len(train_loader))

            bsz = x.size(0)
            running_total = 0.0
            running_recon = 0.0
            running_cls   = 0.0
            n_batches     = 0
            running_recon += recon_loss.item() * bsz
            running_cls += cls_loss.item() * bsz
            running_n += bsz
            running_total += loss.item()
            running_recon += recon_loss.item()
            running_cls   += cls_loss.item()
            n_batches    += 1

        # average over all batches
        avg_total = running_total / n_batches
        avg_recon = running_recon / n_batches
        avg_cls   = running_cls   / n_batches

        print(f"Epoch {epoch:3d}/{num_epochs}  "
              f"Train total={avg_total:.6f}  "
              f"recon={avg_recon:.6f}  "
              f"cls={avg_cls:.6f}")
        # end of epoch
        val_recon, val_acc = validate(model, val_loader, device)
        print(f" → VAL recon={val_recon:.4f}, acc={val_acc:.4f}")
        # save best model
        ckpt_dir = "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v5_VE_cls_only/checkpoint"
        ckpt_dict = {
            #"epoch": epoch,
            "model_state_dict": model.state_dict(),
            #"optimizer_state_dict": optimizer.state_dict()
            #"scheduler_state_dict": scheduler.state_dict()
        }
        torch.save(ckpt_dict, f"{ckpt_dir}/secondround_checkpoint_epoch{epoch}.pt")
        print(f"Saved checkpoint_epoch{epoch}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="path to checkpoint .pt file to resume from"
    )
    args = parser.parse_args()

    # Hyperparameters (no extra parser args)
    data_path = "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_data_128.npz"
    batch_size = 128
    num_epochs = 500
    lr = 5e-5
    weight_decay = 5e-5
    alpha = 0.002
    T0 = 30
    T_mult = 2
    save_dir = "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v5_VE_cls_only/checkpoint"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and loaders (hardcoded paths)
    os.chdir(os.path.dirname(data_path))
    full_dataset = CustomDataset(os.path.basename(data_path), transform=transforms.ToTensor())
    num_classes = int(np.unique(full_dataset.labels).shape[0])
    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))), test_size=0.2, random_state=42
    )
    train_loader = DataLoader(
        Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    model = ConvAE(input_channels=3, latent_channels=8, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)


    if args.resume:
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt)
    train(
        model, train_loader, val_loader, optimizer, scheduler, device,
        num_epochs=num_epochs, alpha=alpha
    )