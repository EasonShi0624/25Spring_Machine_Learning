import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
import pytorch_msssim
import argparse

# ----------------------------------------------------------------
# 0. Helpers for checkpointing
# ----------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


def load_checkpoint(model, optimizer, path, device):
    if os.path.isfile(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt['epoch'] + 1
    else:
        return 1

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true',
                    help='if set, reload from checkpoint before training')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',
                    help='path to checkpoint file')
args = parser.parse_args()

# ----------------------------
# 1. Setup for Reproducibility
# ----------------------------
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# 2. Dataset & Augmentations
# ----------------------------
class CustomDataset(Dataset):
    def __init__(self, npz_file = "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_data_128.npz", transform=None):
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
full_dataset = CustomDataset(transform=None)
all_indices  = list(range(len(full_dataset)))
train_idx, val_idx = train_test_split(
    all_indices,
    test_size=0.1,
    random_state=42,
    shuffle=True
)
# two datasets
train_dataset = CustomDataset(transform=train_transforms)
val_dataset   = CustomDataset(transform=val_transforms)
# subsets
train_subset = Subset(train_dataset, train_idx)
val_subset   = Subset(val_dataset,   val_idx)
# loaders
train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_subset,   batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
num_classes = int(np.unique(full_dataset.labels).shape[0])

# ----------------------------
# 3. Residual Block Definition
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.act   = nn.Mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)

# ----------------------------
# 4. ConvVAE Model with ResidualBlocks
# ----------------------------
class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.SiLU(),
            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.SiLU(), nn.Dropout2d(0.1),
            ResidualBlock(256),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.SiLU(),
            ResidualBlock(256),
        )
        self.quant_conv = nn.Conv2d(256, latent_channels * 2, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.SiLU(),
            ResidualBlock(128),
            nn.Conv2d(128, 64 * (2**2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64), nn.SiLU(),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, input_channels, 4, 2, 1), nn.Sigmoid(),             
        )
        feat_dim = latent_channels * 32 * 32
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(feat_dim, num_classes))

    def preprocess(self, x):
        return 2 * x - 1

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        if self.training:
            logvar = logvar.clamp(-50, 20)
            std = torch.exp(0.5 * logvar).clamp(max=2)
            eps = torch.randn_like(std)
            return mean + eps * std, mean, logvar
        else:
            return mean, mean, logvar

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_rec = self.decode(z)
        logits = self.classifier(z.view(z.size(0), -1))
        return x_rec, z, mean, logvar, logits

# ----------------------------
# 5. Loss & Validation
# ----------------------------

def vae_loss(x, x_recon, mean, logvar, kl_weight):
    recon = 1 - pytorch_msssim.ssim(x, x_recon, data_range=1, size_average=True)
    kl    = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon + kl_weight * kl, recon, kl


def validate(model, loader, device, kl_weight, alpha):
    model.eval()
    stats = {'recon': 0, 'kl': 0, 'class': 0, 'n': 0}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_rec, z, mean, logvar, logits = model(x)
            total, r, k = vae_loss(x, x_rec, mean, logvar, kl_weight)
            c_loss = F.cross_entropy(logits, y)
            bsz    = x.size(0)
            stats['recon'] += r.item() * bsz
            stats['kl']    += k.item() * bsz
            stats['class'] += c_loss.item() * bsz
            stats['n']     += bsz
    return {k: stats[k] / stats['n'] for k in ['recon', 'kl', 'class']}

# ----------------------------
# 6. Training Loop with Validation‐driven LR
# ----------------------------

def train_vae(model, train_loader, val_loader, optimizer, device,
              num_epochs=30, alpha=0.5, resume=False, checkpoint_path='checkpoint.pth'):
    if resume:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 1
    model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        kl_weight = min(0.001, epoch / 20 * 0.001)
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            x_rec, z, mean, logvar, logits = model(x)
            total, r_loss, k_loss = vae_loss(x, x_rec, mean, logvar, kl_weight)
            cls_loss = F.cross_entropy(logits, y)
            loss = total + alpha * cls_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loop.set_postfix(
                loss=loss.item(),
                recon=r_loss.item(),
                kl=k_loss.item(),
                cls=cls_loss.item()
            )
        model.eval()
        val_stats = validate(model, val_loader, device, kl_weight, alpha)
        val_loss = (val_stats['recon'] + kl_weight * val_stats['kl'] + alpha * val_stats['class'])
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch}] lr={current_lr:.2e}  val_recon={val_stats['recon']:.6f}  val_cls={val_stats['class']:.6f}")
        save_checkpoint(model, optimizer, epoch, checkpoint_path)
    
# ----------------------------
# 7. Main
# ----------------------------
if __name__ == "__main__":
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = ConvVAE(input_channels=3, latent_channels=8, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_vae(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=1500,
        alpha=0.002,
        resume=args.resume,
        checkpoint_path=args.checkpoint
    )
