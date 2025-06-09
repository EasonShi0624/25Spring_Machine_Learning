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
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_subset,   batch_size=128, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)


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


class ConvAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: same as before
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 2, 1),  # 128→64
            nn.BatchNorm2d(64), nn.Mish(),

            nn.Conv2d(64, 128, 3, 2, 1),              # 64→32
            nn.BatchNorm2d(128), nn.Mish(),

            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256), nn.Mish(), nn.Dropout2d(0.1),

            ResBlock(256),                  # 1st ResBlock at 32×32
            SelfAttention(256),             # 1st SelfAttention at 32×32
            ResBlock(256)

        )

        # ——— Bottleneck (deterministic) ———
        # map 256→latent_channels
        self.bottleneck_conv      = nn.Conv2d(256, latent_channels*2, 1)
        # map latent_channels→256 for decoder
        self.post_bottleneck_conv = nn.Conv2d(latent_channels, 256, 1)
        self.decoder = nn.Sequential(
            # — refine at 32×32, channels 256 → 256
            ResBlock(256),                  # 3rd ResBlock at 32×32
            SelfAttention(256),             # 2nd SelfAttention at 32×32
            ResBlock(256),  

            # — refine + reduce channels at 32×32: 256 → 128
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.Mish(),


            # — up-sample 32→64, channels 128 → 64
            nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(64),  nn.Mish(),


            # — final refine to RGB (or input_channels) at 128×128
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()            # map back to [0,1]
        )
        # Decoder: same as before
        # pooling + classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),  # [B, latent, H, W] → [B, latent, 1, 1]
            nn.Flatten(),                 # → [B, latent]
            nn.Linear(latent_channels, num_classes)
        )

    def preprocess(self, x):
        return 2*x - 1  # if you still want to normalize to [-1,1]

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.bottleneck_conv(h)
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
        h = self.post_bottleneck_conv(z)
        return self.decoder(h)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_recon         = self.decode(z)
        logits          = self.classifier(z)
        return x_recon, z, logits


# ----------------------------
# 4. Loss & Validation
# ----------------------------
'''
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
            x_recon, z, logits = model(x)

            r= vae_loss(x, x_recon)
            c_loss = F.cross_entropy(logits, y)
            zf = z.view(z.size(0), -1)
            cen = center_loss(zf, y)
            bsz = x.size(0)
            stats['recon'] += r.item() * bsz
            stats['class'] += c_loss.item() * bsz
            stats['n']     += bsz


    avg = {k: stats[k]/stats['n'] for k in ['recon', 'kl', 'class','center']}
    return avg


# ----------------------------
# 5. Training Loop
# ----------------------------
def train_vae_recon_only(model, train_loader, val_loader, optimizer, device,
                         num_epochs=50, cls_weight=1.0):
    """
    Train loop that optimizes both reconstruction (MSE) and classification (CE).
    cls_weight scales the cross-entropy loss relative to the MSE.
    """
    model.to(device)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,     # lr <- lr * 0.5
        patience=5,     # 连续 5 个 epoch 无改善再降
        verbose=True
    )

    best_val_loss = float('inf')
    for epoch in range(1, num_epochs+1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # forward pass
            x_recon, z, logits = model(x)

            # compute losses
            recon_loss = F.mse_loss(x_recon, x, reduction='mean')
            class_loss = F.cross_entropy(logits, y)
            loss = torch.log(recon_loss) + cls_weight * torch.log(class_loss)

            # backward & step
            loss.backward()
            optimizer.step()

            loop.set_postfix(
                recon=f"{recon_loss.item():.5f}",
                classification=f"{class_loss.item():.5f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}"
            )

        # —— Validation phase ——
        model.eval()
        total_recon, total_class, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x_recon, z, logits = model(x)

                r = F.mse_loss(x_recon, x, reduction='mean')
                c = F.cross_entropy(logits, y)
                batch_size = x.size(0)
                total_recon += r.item() * batch_size
                total_class += c.item() * batch_size
                n += batch_size

        avg_recon = total_recon / n
        avg_class = total_class / n
        val_loss  = avg_recon + cls_weight * avg_class

        print(f"[Epoch {epoch}] val_recon={avg_recon:.6f}, "
              f"val_class={avg_class:.6f}, val_loss={val_loss:.6f}")

        # step scheduler on the combined loss
        scheduler.step()
        # Save checkpoint
        ckpt_dir="/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v6.2_maxpool_head/checkpoint"
        torch.save(model.state_dict(), f"{ckpt_dir}/cosineLR_newclassi_checkpoint_epoch{epoch}.pt")
        print(f"Saved checkpoint_epoch{epoch}.pt")
        #torch.save(model.state_dict(),"/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v6_morelayerv4/checkpoint")

    print("=== 只优化重建损失训练结束 ===")
    '''
def ae_ce_loss(x, x_recon, logits, labels, alpha=0.1): # <---可调整
    # 重建损失（MSE，也可选 reduction='mean' 或 'sum'）
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    # 分类损失（交叉熵）
    cls_loss   = F.cross_entropy(logits, labels)
    final_loss = recon_loss + alpha * cls_loss
    return final_loss, recon_loss, cls_loss



def train_vae(model, train_loader, val_loader, optimizer, device, num_epochs=1, patience=50, alpha=0.1):

    # 先初始化学习率调度器 & Early‑stop 变量
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,     # lr <- lr * 0.5
        patience=5,     # 连续 5 个 epoch 无改善再降
    )
    best_val_loss     = float('inf')
    epochs_no_improve = 0
    train_losses = []    # ← 用来存放每个 epoch 的train loss
    val_losses   = []    # ← 用来存放每个 epoch 的validate loss


    for epoch in range(1, num_epochs+1):
        # ----- 1) 训练 -----
        model.train()
        train_total = train_recon = train_cls = 0.0

        # ← NEW: 初始化训练准确率统计
        train_correct = 0    # ← NEW
        train_total   = 0    # ← NEW
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            x, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            x_recon, z, logits = model(x)
            total_loss, recon_loss, cls_loss = ae_ce_loss(x, x_recon, logits, labels, alpha)
            total_loss.backward()
            optimizer.step()

            train_total += total_loss.item()
            train_recon += recon_loss.item()
            train_cls   += cls_loss.item()

            # 计算 accuracy
            preds = logits.argmax(dim=1)             # B
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        # 训练集平均
        avg_train_total = train_total / len(train_loader)
        avg_train_recon = train_recon / len(train_loader)
        avg_train_cls   = train_cls   / len(train_loader)
        avg_train_acc   = train_correct / train_total 

        # ----- 2) 验证 -----
        model.eval()
        val_total = 0.0
        val_recon = 0.0
        val_cls   = 0.0
        # ← NEW: 初始化验证准确率统计
        val_correct = 0     # ← NEW
        val_total   = 0     # ← NEW

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                x, y = images.to(device), labels.to(device)
                x_recon, z, logits = model(x)
                total_loss, recon_loss, cls_loss = ae_ce_loss(x, x_recon, logits, y, alpha)
                val_total += total_loss.item()
                val_recon += recon_loss.item()
                val_cls   += cls_loss.item()

                # ← NEW: 计算并累加验证正确数
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total   += y.size(0)

        # 计算平均值
        avg_val_total = val_total / len(val_loader)
        avg_val_recon = val_recon / len(val_loader)
        avg_val_cls   = val_cls   / len(val_loader)
        avg_val_acc   = val_correct   / val_total 

        # ----- 3) 学习率调度器 -----
        # 用 avg_val_total（total loss）来驱动 LR 调度与 early‑stop
        scheduler.step(avg_val_total)
        
        # ----- 4) 打印 -----
        print(
            f"Epoch {epoch:02d}:\n"
            f"  Train → Total={avg_train_total:.8f}, Recon={avg_train_recon:.8f}, Cls={avg_train_cls:.8f},Acc={avg_train_acc:.4f}\n"
            f"  Val   → Total={avg_val_total:.8f}, Recon={avg_val_recon:.8f}, Cls={avg_val_cls:.8f}, Acc={avg_val_acc:.4f}"
        )
        ckpt_dir="/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v10086_ydx/checkpoint"
        torch.save(model.state_dict(), f"{ckpt_dir}/less_layer_checkpoint_epoch{epoch}.pt")
        print(f"Saved checkpoint_epoch{epoch}.pt")
    return train_losses, val_losses

# ----------------------------
# 6. Main
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvAE(input_channels=3, latent_channels=8, num_classes=num_classes)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=1000,
        alpha=0.0003
    )

#### Recon only. No residual block. Recon=combination of two functions.
### Best entry (1280)
### 更新：新的resblock, attention
