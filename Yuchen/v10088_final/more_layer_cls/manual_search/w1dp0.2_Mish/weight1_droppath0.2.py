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
#os.chdir("/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v6_morelayerv4")
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
import math
class DropPath(nn.Module):            # ≈ timm’s implementation
    def __init__(self, p=0.):
        super().__init__()
        self.p = p
    def forward(self, x):
        if self.p == 0. or not self.training:
            return x
        keep = 1 - self.p
        # one mask per sample, broadcast over C,H,W
        shape = (x.size(0),) + (1,)*(x.ndim-1)
        mask  = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

class ResBlock(nn.Module):
    """
    A standard residual block with two 3x3 conv layers and Mish activation.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.act   = nn.Mish()
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



class SelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads=4, drop_path=0.2):
        super().__init__()
        assert in_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = in_dim // num_heads

        self.to_qkv      = nn.Conv2d(in_dim, 3*in_dim, 1, bias=False)
        self.unify_heads = nn.Conv2d(in_dim, in_dim, 1)

        self.norm1 = nn.GroupNorm(1, in_dim)
        self.norm2 = nn.GroupNorm(1, in_dim)

        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim*4, 1), nn.SiLU(),
            nn.Conv2d(in_dim*4, in_dim, 1)
        )

        self.gamma_attn = nn.Parameter(torch.zeros(1))
        self.gamma_ffn  = nn.Parameter(torch.zeros(1))

        # --- new ---
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm1(x)

        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        def reshape(t):  # B,C,H,W -> B,h,d,HW
            return t.view(B, self.num_heads, self.d_k, H*W)
        q, k, v = map(reshape, qkv)

        attn = torch.softmax(
            torch.einsum('bhdi,bhdj->bhij', q, k) / math.sqrt(self.d_k), dim=-1
        )
        out = torch.einsum('bhij,bhdj->bhdi', attn, v).reshape(B, C, H, W)
        # residual + DropPath
        x2 = x + self.drop_path(self.gamma_attn * self.unify_heads(out))

        # FFN branch + DropPath
        ffn_out = self.ffn(self.norm2(x2))
        return x2 + self.drop_path(self.gamma_ffn * ffn_out)



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
            nn.BatchNorm2d(256), nn.Mish(),

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
            # 1) project latent → 128
            nn.Conv2d(latent_channels, 128, kernel_size=1),
            nn.Mish(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Mish(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(), 
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
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

def ae_ce_loss(x, x_recon, logits, labels, alpha=0.1): # <---可调整
    # 重建损失（MSE，也可选 reduction='mean' 或 'sum'）
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    # 分类损失（交叉熵）
    cls_loss   = F.cross_entropy(logits, labels)
    final_loss = torch.log(recon_loss) + alpha * torch.log(cls_loss)
    return final_loss, recon_loss, cls_loss



def train_vae(model, train_loader, val_loader, optimizer, device,
              num_epochs=1, patience=50, alpha=0.1):

    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=0.5, patience=5)
    best_val_total = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        # ─────────── 1) TRAIN ───────────
        model.train()
        tr_loss_sum = tr_recon_sum = tr_cls_sum = 0.0
        tr_correct = tr_samples = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            x_rec, z, logits = model(x)
            tot_loss, rec_loss, cls_loss = ae_ce_loss(x, x_rec, logits, y, alpha)
            tot_loss.backward()
            optimizer.step()

            # running sums
            preds = logits.argmax(dim=1)
            tr_correct   += (preds == y).sum().item()
            tr_samples   += y.size(0)

            tr_loss_sum  += tot_loss.item()
            tr_recon_sum += rec_loss.item()
            tr_cls_sum   += cls_loss.item()

        # per-sample averages
        avg_tr_total = tr_loss_sum  / tr_samples
        avg_tr_recon = tr_recon_sum / tr_samples
        avg_tr_cls   = tr_cls_sum   / tr_samples
        avg_tr_acc   = tr_correct   / tr_samples

        # ─────────── 2) VALIDATION ───────────
        model.eval()
        val_loss_sum = val_recon_sum = val_cls_sum = 0.0
        val_correct = val_samples = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                x_rec, z, logits = model(x)
                tot_loss, rec_loss, cls_loss = ae_ce_loss(x, x_rec, logits, y, alpha)

                preds = logits.argmax(dim=1)
                val_correct  += (preds == y).sum().item()
                val_samples  += y.size(0)

                val_loss_sum  += tot_loss.item()
                val_recon_sum += rec_loss.item()
                val_cls_sum   += cls_loss.item()

        avg_val_total = val_loss_sum  / val_samples
        avg_val_recon = val_recon_sum / val_samples
        avg_val_cls   = val_cls_sum   / val_samples
        avg_val_acc   = val_correct   / val_samples

        # ─────────── 3) SCHEDULER & EARLY-STOP ───────────
        scheduler.step(avg_val_total)

        if avg_val_total < best_val_total - 1e-6:
            best_val_total = avg_val_total
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹ Early-stopping after {epoch} epochs (no impro. for {patience})")
                break

        # ─────────── 4) LOGGING ───────────
        print(
            f"Epoch {epoch:02d}: "
            f"Train→ Tot {avg_tr_total:.8f}  Recon {avg_tr_recon:.8f}  "
            f"Cls {avg_tr_cls:.8f}  Acc {avg_tr_acc:.8f} | "
            f"Val→ Tot {avg_val_total:.8f}  Recon {avg_val_recon:.8f}  "
            f"Cls {avg_val_cls:.8f}  Acc {avg_val_acc:.8f}"
        )

        # keep curves for plotting if you like
        train_losses.append(avg_tr_total)
        val_losses.append(avg_val_total)

        # ─────────── 5) CHECKPOINT ───────────
        ckpt_dir = "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v10088_final/more_layer_cls/manual_search/w1dp0.2_Mish/checkpoint"
        ckpt_path = f"{ckpt_dir}/Mish_weight1_droppath0.2_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"✓ saved {ckpt_path}")

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
        alpha=1
    )

#### Recon only. No residual block. Recon=combination of two functions.
### Best entry (1280)
### 更新：新的resblock, attention
