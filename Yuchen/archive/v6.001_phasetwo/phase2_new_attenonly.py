import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from tqdm import tqdm

def weighted_mse(x, x_recon, weight=1.5):   #weight of central batch==1.5
    """
    MSE where the central 64×64 patch is 'weight' times more important
    than the rest of the image.

    x, x_recon: [B, C, 128, 128] tensors in [0,1]
    """
    # create a 1‑channel mask broadcastable to [B,C,H,W]
    mask = torch.ones_like(x[:, :1])                  # [B,1,128,128]
    mask[:, :, 16:112, 16:112] = weight                # central patch
    # squared error
    mse = ((x - x_recon).pow(2) * mask).mean()
    return mse

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
        # uint8 for PIL
        img_np = (self.images[idx].transpose(1,2,0) * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# train_transforms = transforms.Compose([ #define transform methods########
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(
#         15,
#         interpolation=transforms.InterpolationMode.BILINEAR
#     ),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
# ])
# val_transforms = transforms.Compose([
#     transforms.ToTensor(),
# ])

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
])
# adjust to your dataset path
os.chdir("/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v6_morelayerv4")
full_dataset = CustomDataset("/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_data_128.npz", transform=None)
num_classes = int(np.unique(full_dataset.labels).shape[0])

train_idx, val_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.1,
    random_state=42,
    stratify=full_dataset.labels
)
train_subset = Subset(full_dataset, train_idx)
val_subset   = Subset(full_dataset, val_idx)
train_subset.dataset.transform = train_transforms
val_subset.dataset.transform   = val_transforms

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True,     ####Hyper parameter: BATCH SIZE
                          num_workers=1, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_subset,   batch_size=128, shuffle=False,
                          num_workers=1, pin_memory=True, persistent_workers=True)


# ----------------------------
# 3. ConvVAE Model
# ----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    A standard residual block with two 3x3 conv layers and Mish activation.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.act   = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return self.act(out)

import math

class SelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads=4):
        super().__init__()
        assert in_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = in_dim // num_heads

        # one proj for all heads, then reshape
        self.to_qkv = nn.Conv2d(in_dim, in_dim * 3, 1, bias=False)
        self.unify_heads = nn.Conv2d(in_dim, in_dim, 1)
        self.norm1 = nn.GroupNorm(1, in_dim)
        self.norm2 = nn.GroupNorm(1, in_dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim*4, 1),
            nn.SiLU(),
            nn.Conv2d(in_dim*4, in_dim, 1),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        # Pre-norm
        x_norm = self.norm1(x)

        # QKV and split heads
        qkv = self.to_qkv(x_norm)                   # B, 3C, H, W
        q, k, v = qkv.chunk(3, dim=1)                # each B, C, H, W
        # reshape to (B, heads, d_k, N)
        def reshape(t):
            t = t.view(B, self.num_heads, self.d_k, H*W)
            return t
        q, k, v = reshape(q), reshape(k), reshape(v)

        # scaled dot-product
        scores = torch.einsum('bhdi,bhdj->bhij', q, k) / math.sqrt(self.d_k)
        attn   = torch.softmax(scores, dim=-1)        # B, heads, N, N

        out = torch.einsum('bhij,bhdj->bhdi', attn, v)  # B, heads, d_k, N
        out = out.contiguous().view(B, C, H, W)         # concat heads

        attn_out = self.unify_heads(out)
        x2 = x + self.gamma * attn_out                  # residual

        # FFN block
        x2_norm = self.norm2(x2)
        ffn_out = self.ffn(x2_norm)
        return x2 + ffn_out


class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: add ResidualBlocks and SelfAttention
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 2, 1),  # 128→64
            nn.BatchNorm2d(64), nn.Mish(),

            nn.Conv2d(64, 128, 3, 2, 1),              # 64→32
            nn.BatchNorm2d(128), nn.Mish(),

            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256), nn.Mish(), nn.Dropout2d(0.1),
            ResBlock(256),
            SelfAttention(256),

            nn.Conv2d(256,256,3,1,1),                 # refine 32×32
            nn.BatchNorm2d(256), nn.Mish(),
            nn.Conv2d(256,256,3,1,1),                 # NEW
            nn.BatchNorm2d(256), nn.Mish()
        )

        # Bottleneck
        self.quant_conv      = nn.Conv2d(256, latent_channels * 2, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)

        # Decoder: add ResidualBlocks and SelfAttention
        self.decoder = nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256), nn.Mish(),
            ResBlock(256),
            SelfAttention(256),

            nn.Conv2d(256,128,3,1,1),
            nn.BatchNorm2d(128), nn.Mish(),
            ResBlock(128),
            SelfAttention(128),

            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128), nn.Mish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),      # 32→64
            nn.BatchNorm2d(64), nn.Mish(),

            nn.ConvTranspose2d(64, input_channels, 4, 2, 1),
            nn.Sigmoid()
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(latent_channels * 32 * 32, num_classes)
        )

    def preprocess(self, x):
        return 2*x - 1  # normalize to [-1,1]

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        if self.training:
            logvar = logvar.clamp(-10.0, 2.0)
            std    = torch.exp(0.5 * logvar).clamp(max=2.0)
            eps    = torch.randn_like(std)
            z      = mean + eps * std
            return z, mean, logvar
        else:
            z = mean
            return z, mean, logvar

    def decode(self, z):
        h = self.post_quant_conv(z)
        return self.decoder(h)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_recon         = self.decode(z)
        logits          = self.classifier(z.view(z.size(0), -1))
        return x_recon, z, mean, logvar, logits


# ----------------------------
# 4. Loss & Validation
# ----------------------------
def vae_loss(x, x_recon):
    recon1 = F.mse_loss(x, x_recon)  # SSIM-based reconstruction loss
    # kl    = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon1

def validate(model, loader, device, kl_weight,center_loss):
    model.eval()
    stats = {'recon': 0, 'kl': 0, 'class': 0, 'center': 0, 'n': 0}

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_recon, z, mean, logvar, logits = model(x)

            total, r, k = vae_loss(x, x_recon, mean, logvar, kl_weight)
            c_loss = F.cross_entropy(logits, y)
            zf = z.view(z.size(0), -1)
            cen = center_loss(zf, y)
            bsz = x.size(0)
            stats['recon'] += r.item() * bsz
            stats['kl']    += k.item() * bsz
            stats['center'] += cen.item() * bsz
            stats['class'] += c_loss.item() * bsz
            stats['n']     += bsz


    avg = {k: stats[k]/stats['n'] for k in ['recon', 'kl', 'class','center']}
    return avg


# ----------------------------
# 5. Training Loop
# ----------------------------
def train_vae_recon_only(model, train_loader, val_loader, optimizer, device,
                         num_epochs=50):
    model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)

    best_recon = float('inf')
    for epoch in range(1, num_epochs+1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for x, _ in loop:               # 不再使用 labels
            x = x.to(device)
            optimizer.zero_grad()

            x_recon, z, mean, logvar, logits = model(x)
            loss = vae_loss(x, x_recon)

            loss.backward()
            optimizer.step()
            loop.set_postfix(recon=loss.item(),
                             lr=optimizer.param_groups[0]['lr'])

        # —— 验证阶段 ——
        model.eval()
        total_recon = 0.0
        n = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_recon, *_ = model(x)
                r = vae_loss(x, x_recon)
                total_recon += r * x.size(0)
                n += x.size(0)
        avg_recon = total_recon / n

        print(f"[Epoch {epoch}] val_recon = {avg_recon:.6f}")
        scheduler.step(avg_recon)
        # Save checkpoint
        ckpt_dir="/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v6_morenew_resatten/checkpoint"
        torch.save(model.state_dict(), f"{ckpt_dir}/atten_only_checkpoint_epoch{epoch}.pt")
        print(f"Saved checkpoint_epoch{epoch}.pt")
        #torch.save(model.state_dict(),"/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v6_morelayerv4/checkpoint")

    print("=== 只优化重建损失训练结束 ===")
# ----------------------------
# 6. Main
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvVAE(input_channels=3, latent_channels=8, num_classes=num_classes)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train_vae_recon_only(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=1000
    )

#### Recon only. No residual block. Recon=combination of two functions.
### Best entry (1280)
### 更新：新的resblock, attention
