import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

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

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
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
# 3. ConvVAE Model
# ----------------------------
class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

        # ─── Encoder: five conv blocks ───
        #   -- first two downsample to 32×32, next three are stride‐1 refinements
        self.encoder = nn.Sequential(
            # 128→64
            nn.Conv2d(input_channels,  64, 3, 2, 1),
            nn.BatchNorm2d(64), nn.SiLU(),
            # 64→32
            nn.Conv2d(64,            128, 3, 2, 1),
            nn.BatchNorm2d(128), nn.SiLU(),
            # refine @32×32 (3 blocks)
            nn.Conv2d(128,           256, 3, 1, 1),
            nn.BatchNorm2d(256), nn.SiLU(),

            nn.Conv2d(256,           256, 3, 1, 1),
            nn.BatchNorm2d(256), nn.SiLU(),

            nn.Conv2d(256,           256, 3, 1, 1),
            nn.BatchNorm2d(256), nn.SiLU(),
        )

        # ─── Quantization ───
        # projects 256→ latent*2 for mean + logvar
        self.quant_conv = nn.Conv2d(256, latent_channels*2, 1)

        # ─── Decoder: five blocks ───
        #   -- first three are stride‐1 refinements @32×32, then two upsample steps
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)
        self.decoder = nn.Sequential(
            # refine @32×32 (3 blocks)
            nn.Conv2d(256,           256, 3, 1, 1),
            nn.BatchNorm2d(256), nn.SiLU(),

            nn.Conv2d(256,           256, 3, 1, 1),
            nn.BatchNorm2d(256), nn.SiLU(),

            nn.Conv2d(256,           128, 3, 1, 1),
            nn.BatchNorm2d(128), nn.SiLU(),

            # upsample 32→64
            nn.ConvTranspose2d(128,  64, 4, 2, 1),
            nn.BatchNorm2d(64),  nn.SiLU(),
            # upsample 64→128 & output
            nn.ConvTranspose2d(64,   input_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

        # classifier (unchanged)
        feat_dim = latent_channels * 32 * 32  # = 8192
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feat_dim, num_classes)
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
        x_recon         = self.decode(z)
        logits          = self.classifier(z.view(z.size(0), -1))
        return x_recon, z, mean, logvar, logits

# ----------------------------
# 4. Loss & Validation
# ----------------------------
def vae_loss(x, x_recon, mean, logvar, kl_weight):
    recon = F.mse_loss(x, x_recon, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon + kl_weight * kl, recon, kl


def validate(model, loader, device, kl_weight, alpha, gamma, center_loss, vgg, imagenet_mean, imagenet_std, delta):
    model.eval()
    stats = {'recon': 0, 'kl': 0, 'class': 0, 'center': 0, 'perc': 0, 'n': 0}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_rec, z, mean, logvar, logits = model(x)
            total, r, k = vae_loss(x, x_rec, mean, logvar, kl_weight)
            c_loss = F.cross_entropy(logits, y)
            zf = z.view(z.size(0), -1)
            cen = center_loss(zf, y)
            x_norm = (x - imagenet_mean) / imagenet_std
            recon_norm = (x_rec - imagenet_mean) / imagenet_std
            real_feat = vgg(x_norm)
            recon_feat = vgg(recon_norm)
            p_loss = F.mse_loss(recon_feat, real_feat)
            bsz = x.size(0)
            stats['recon'] += r.item() * bsz
            stats['kl'] += k.item() * bsz
            stats['class'] += c_loss.item() * bsz
            stats['center'] += cen.item() * bsz
            stats['perc'] += p_loss.item() * bsz
            stats['n'] += bsz
    return {k: stats[k] / stats['n'] for k in ['recon', 'kl', 'class', 'center', 'perc']}

# ----------------------------
# 5. Training Loop with SGDR
# ----------------------------
def train_vae(model, train_loader, val_loader, optimizer, device,
              num_epochs=30, beta_max=0.1, alpha=0.5, gamma=0.1,
              vgg=None, imagenet_mean=None, imagenet_std=None, delta=0.01):
    model.to(device)
    feat_dim = model.latent_channels * 32 * 32
    center_loss = CenterLoss(num_classes, feat_dim, device)
    optimizer.add_param_group({'params': center_loss.parameters()})

    # SGDR: CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=75, T_mult=2, eta_min=1e-7)

    for epoch in range(1, num_epochs + 1):
        model.train()
        kl_weight = min(beta_max, beta_max * epoch / num_epochs)
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            x_rec, z, mean, logvar, logits = model(x)
            total, r_loss, k_loss = vae_loss(x, x_rec, mean, logvar, kl_weight)
            cls_loss = F.cross_entropy(logits, y)
            zf = z.view(z.size(0), -1)
            cen_loss = center_loss(zf, y)
            x_norm = (x - imagenet_mean) / imagenet_std
            recon_norm = (x_rec - imagenet_mean) / imagenet_std
            real_feat = vgg(x_norm)
            recon_feat = vgg(recon_norm)
            perc_loss = F.mse_loss(recon_feat, real_feat)
            loss = total + alpha * cls_loss + gamma * cen_loss + delta * perc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # Step scheduler once per epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        val_stats = validate(model, val_loader, device, kl_weight, alpha, gamma, center_loss,
                              vgg, imagenet_mean, imagenet_std, delta)
        print(f"[Epoch {epoch}] lr={current_lr:.2e}  "
              f"val_recon={val_stats['recon']:.6f}  "
              f"val_cls={val_stats['class']:.6f}  "
              f"val_center={val_stats['center']:.6f}  "
              f"val_perc={val_stats['perc']:.6f}")

        # Save checkpoint per epoch
        ckpt_dir = "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4_85in5out/checkpoint"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_name = f"{ckpt_dir}/checkpoint_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_name)
        print(f"Saved {ckpt_name}")

# ----------------------------
# 6. Main
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # VGG feature extractor for perceptual loss
    vgg = models.vgg16(pretrained=False).features[:16].to(device).eval()
    for p in vgg.parameters(): p.requires_grad = False
    imagenet_mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
    imagenet_std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)

    model = ConvVAE(input_channels=3, latent_channels=8, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train_vae(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=1000,
        beta_max=0.0005,
        alpha=0.0029355919056392875,
        gamma=0.004214870341812469,
        vgg=vgg,
        imagenet_mean=imagenet_mean,
        imagenet_std=imagenet_std,
        delta=1
    )
