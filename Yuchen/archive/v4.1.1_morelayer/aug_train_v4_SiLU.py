import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

"""
python aug_train_v4_SiLU.py --resume /path/to/checkpoint.pt
"""
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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
        self.images = data['images']   # (N,3,128,128) in [0,1]
        self.labels = data['labels']   # (N,)
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
    transforms.RandomResizedCrop(size=128, scale=(0.8,1.0), ratio=(0.9,1.1), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10, interpolation=InterpolationMode.BILINEAR),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02,0.2), ratio=(0.3,3.3), value='random'),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
val_transforms = transforms.Compose([transforms.ToTensor()])

os.chdir("/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset")
full_dataset = CustomDataset("train_data_128.npz", transform=None)
num_classes = int(np.unique(full_dataset.labels).shape[0])
train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.1, random_state=42, stratify=full_dataset.labels)
train_subset = Subset(full_dataset, train_idx)
val_subset = Subset(full_dataset, val_idx)
train_subset.dataset.transform = train_transforms
val_subset.dataset.transform = val_transforms
train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

# ----------------------------
# 2. MixUp / CutMix utilities
# ----------------------------
def rand_beta(alpha):
    return torch.distributions.Beta(alpha, alpha).sample().item()

def cutmix_data(x, y, alpha=1.0):
    batch_size, _, H, W = x.size()
    lam = rand_beta(alpha)
    cut_rat = torch.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    perm = torch.randperm(batch_size)
    x_perm = x[perm]
    y_perm = y[perm]
    x[:, :, bby1:bby2, bbx1:bbx2] = x_perm[:, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x, y, y_perm, lam


def mixup_data(x, y, alpha=1.0):
    lam = rand_beta(alpha)
    batch_size = x.size(0)
    perm = torch.randperm(batch_size)
    x_perm = x[perm]
    y_perm = y[perm]
    mixed_x = lam * x + (1 - lam) * x_perm
    return mixed_x, y, y_perm, lam

# ----------------------------
# 3. Center Loss
# ----------------------------
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device))
    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        diff = features - batch_centers
        return 0.5 * diff.pow(2).sum() / features.size(0)

# ----------------------------
# 4. ConvVAE
# ----------------------------
class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.PReLU(), nn.Dropout2d(0.3),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.PReLU(),
        )
        self.quant_conv = nn.Conv2d(256, latent_channels * 2, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.ConvTranspose2d(64, input_channels, 4, 2, 1), nn.Sigmoid(),
        )
        feat_dim = latent_channels * 32 * 32
        self.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(feat_dim, num_classes))
    def preprocess(self, x): return 2 * x - 1
    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        if self.training:
            logvar = logvar.clamp(-10, 2)
            std = torch.exp(0.5 * logvar).clamp(max=2)
            eps = torch.randn_like(std)
            return mean + eps * std, mean, logvar
        return mean, mean, logvar
    def decode(self, z): return self.decoder(self.post_quant_conv(z))
    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_rec = self.decode(z)
        logits = self.classifier(z.view(z.size(0), -1))
        return x_rec, z, mean, logvar, logits

# ----------------------------
# 5. Loss & Validation
# ----------------------------
def vae_loss(x, x_rec, mean, logvar, kl_weight):
    recon = F.mse_loss(x, x_rec, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon + kl_weight * kl, recon, kl


def validate(model, loader, device, kl_weight, alpha, gamma, center_loss, vgg, imagenet_mean, imagenet_std, delta):
    model.eval()
    stats = {k: 0 for k in ['recon','kl','class','center','perc']}
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_rec, z, mean, logvar, logits = model(x)
            _, r, k = vae_loss(x, x_rec, mean, logvar, kl_weight)
            c = F.cross_entropy(logits, y)
            zf = z.view(z.size(0), -1)
            cen = center_loss(zf, y)
            xn = (x - imagenet_mean) / imagenet_std
            rn = (x_rec - imagenet_mean) / imagenet_std
            real_f = vgg(xn)
            rec_f = vgg(rn)
            p = F.mse_loss(rec_f, real_f)
            bsz = x.size(0)
            for key, val in zip(stats.keys(), [r,k,c,cen,p]): stats[key] += val.item() * bsz
            n += bsz
    return {k: stats[k]/n for k in stats}

# ----------------------------
# 6. Training Loop with SGDR + MixUp/CutMix
# ----------------------------
def train_vae(model, train_loader, val_loader, optimizer, device,
              num_epochs=30, beta_max=0.1, alpha=0.5, gamma=0.1,
              vgg=None, imagenet_mean=None, imagenet_std=None, delta=0.01, resume=""):
    model.to(device)
    feat_dim = model.latent_channels * 32 * 32
    center_loss = CenterLoss(num_classes, feat_dim, device)
    optimizer.add_param_group({'params': center_loss.parameters()})

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=3, eta_min=1e-8)

    start_epoch = 1
    if resume:
        print(f"Loading checkpoint '{resume}' …")
        ckpt = torch.load(resume, map_location=device)
        # Support raw state_dict or wrapped dict
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print("Training resumed successfully")

    for epoch in range(start_epoch, num_epochs + start_epoch):
        model.train()
        sums = {k:0.0 for k in ['recon','kl','cls','cen','perc','tot']}
        count = 0
        kl_weight = min(beta_max, epoch/20 * beta_max)
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            if random.random() < 0.5:
                x_input, y1, y2, lam = cutmix_data(x.clone(), y, alpha)
            else:
                x_input, y1, y2, lam = mixup_data(x, y, alpha)

            optimizer.zero_grad()
            x_rec, z, mean, logvar, logits = model(x_input)
            total, r, k = vae_loss(x_input, x_rec, mean, logvar, kl_weight)
            cls = lam * F.cross_entropy(logits, y1) + (1-lam) * F.cross_entropy(logits, y2)
            zf = z.view(z.size(0), -1)
            cen = center_loss(zf, y)
            xn = (x_input - imagenet_mean) / imagenet_std
            rn = (x_rec - imagenet_mean) / imagenet_std
            rf = vgg(xn)
            recf = vgg(rn)
            p = F.mse_loss(recf, rf)
            loss = total + alpha*cls + gamma*cen + delta*p
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loop.set_postfix(recon=f"{r:.6f}", kl=f"{k:.6f}", cls=f"{cls.item():.6f}", cen=f"{cen.item():.6f}", perc=f"{p:.6f}", total=f"{loss.item():.6f}")
            for key, val in zip(['recon','kl','cls','cen','perc','tot'], [r,k,cls.item(),cen.item(),p,loss.item()]):
                sums[key] += val
            count += 1

        avg = {k: sums[k]/count for k in sums}
        print(f"Train ep{epoch}: recon={avg['recon']:.4f} kl={avg['kl']:.4f} cls={avg['cls']:.4f} cen={avg['cen']:.4f} perc={avg['perc']:.4f} tot={avg['tot']:.4f}")
        val_stats = validate(model, val_loader, device, kl_weight, alpha, gamma, center_loss, vgg, imagenet_mean, imagenet_std, delta)
        print(f"[Epoch {epoch}] val_recon={val_stats['recon']:.6f} val_cls={val_stats['class']:.6f} val_center={val_stats['center']:.6f} val_perc={val_stats['perc']:.6f}")

        # Save checkpoint
        ckpt_dir = "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4.1.1_mirror/checkpoint"
        torch.save(model.state_dict(), f"{ckpt_dir}/checkpoint_epoch{epoch}.pt")
        print(f"Saved checkpoint_epoch{epoch}.pt")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = models.vgg16(pretrained=False).features[:16].to(device).eval()
    for p in vgg.parameters(): p.requires_grad=False
    imagenet_mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
    imagenet_std = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    model = ConvVAE(input_channels=3, latent_channels=8, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_vae(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=150, beta_max=0.001, alpha=0.004, gamma=0.003,
        vgg=vgg, imagenet_mean=imagenet_mean, imagenet_std=imagenet_std,
        delta=3, resume=args.resume
    )
