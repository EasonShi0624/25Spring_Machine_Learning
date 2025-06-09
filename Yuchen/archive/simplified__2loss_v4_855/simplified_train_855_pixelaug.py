import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn import PixelShuffle
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from tqdm import tqdm

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

mean = [0.11082429438829422, 0.11747730523347855, 0.13228189945220947]
std  = [0.24317263066768646, 0.24709554016590118, 0.2777163088321686]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0), ratio=(3/4, 4/3)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])
val_transforms = transforms.Compose([transforms.ToTensor(),])

# adjust to your dataset path
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

train_loader = DataLoader(
    train_subset, batch_size=128, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True
)
val_loader   = DataLoader(
    val_subset,   batch_size=128, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True
)

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
# 3. ConvVAE Model
# ----------------------------
class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: 128→64→32→32→32→32
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels,  64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64,            128, 3, 2, 1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128,           256, 3, 1, 1), nn.BatchNorm2d(256), nn.SiLU(),
            nn.Conv2d(256,           256, 3, 1, 1), nn.BatchNorm2d(256), nn.SiLU(),
            nn.Conv2d(256,           256, 3, 1, 1), nn.BatchNorm2d(256), nn.SiLU(),
        )
        self.quant_conv = nn.Conv2d(256, latent_channels*2, 1)
        
        # Decoder: refine @32→ upsample to 64→ upsample to 128
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 64*4, 3, 1, 1), PixelShuffle(2), nn.BatchNorm2d(64), nn.SiLU(),
            nn.ConvTranspose2d(64, input_channels, 4, 2, 1), nn.Sigmoid(),
        )

        feat_dim = latent_channels * 32 * 32
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(feat_dim, num_classes)
        )

    def encode(self, x):
        h = self.encoder(2*x - 1)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        if self.training:
            std = (0.5*logvar).exp()
            z   = mean + torch.randn_like(std)*std
            return z, mean, logvar
        return mean, mean, logvar

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_rec         = self.decode(z)
        logits        = self.classifier(z.view(z.size(0), -1))
        return x_rec, z, mean, logvar, logits

# ----------------------------
# 4. Loss & Validation
# ----------------------------
def recon_loss(x, x_rec):
    return F.mse_loss(x_rec, x, reduction='mean')

def validate(model, loader, device):
    model.eval()
    stats = {'recon': 0.0, 'class': 0.0, 'n': 0}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_rec, _, _, _, logits = model(x)
            r = recon_loss(x, x_rec)
            c = F.cross_entropy(logits, y)
            bsz = x.size(0)
            stats['recon'] += r.item() * bsz
            stats['class'] += c.item() * bsz
            stats['n']     += bsz
    return {k: stats[k]/stats['n'] for k in ('recon','class')}

# ----------------------------
# 5. Training with CosineAnnealingWarmRestarts
# ----------------------------
def train_model(model, train_loader, val_loader, optimizer, device,
                num_epochs=30, scheduler=None, save_ckpt=True,
                alpha=0.001, gamma=0.002):
    model.to(device)
    # instantiate SGDR scheduler if none provided
    if scheduler is None:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=2, eta_min=5e-8
        )

    for epoch in range(1, num_epochs + 1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            x_rec, _, _, _, logits = model(x)
            r_loss = recon_loss(x, x_rec)
            c_loss = F.cross_entropy(logits, y)
            feat_dim = model.latent_channels * 32 * 32
            # center loss
            centers = CenterLoss(num_classes, feat_dim, device)
            zf = model.encode(x)[0].view(x.size(0), -1)
            cen_loss = centers(zf, y)
            loss = r_loss * alpha + c_loss * gamma + cen_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        val_stats = validate(model, val_loader, device)
        print(f"[Epoch {epoch}] lr={lr:.2e}  "
              f"val_recon={val_stats['recon']:.6f}  "
              f"val_cls={val_stats['class']:.6f}")

        if save_ckpt:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved {ckpt_path}")

# ----------------------------
# 6. Main
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvVAE(input_channels=3, latent_channels=8, num_classes=num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=1500,
        alpha=0.0046011068847746925,
        gamma=0.001235889615945341
    )
