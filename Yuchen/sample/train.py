# Mount to Google Drive & Switch to the dataset directory
import os
from google.colab import drive
drive.mount('/content/drive')

os.chdir("/content/drive/MyDrive/ml final competition")
os.listdir()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import random
import numpy as np

# Set a fixed random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, npz_path):
        npz_data = np.load(npz_path)
        self.images = npz_data["images"] # (N, 3, 128, 128) in np.uint8
        self.labels = npz_data["labels"] # (N,) in np.int64
        assert self.images.shape[0] == self.labels.shape[0]
        print(f"{npz_path}: images shape {self.images.shape}, "
              f"labels shape {self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]) / 255 # convert to [0, 1] range
        label = torch.tensor(self.labels[idx])
        return image, label

train_dataset = CustomDataset("train.npz")
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)

# sample data batch
images, labels = next(iter(train_loader))
print(f"images shape: {images.shape}")
print(f"labels shape: {labels.shape}")

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import os


class ResidualBlock(nn.Module):
    """简单的残差块，增强特征表达能力并稳定训练。"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


class Model(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8):
        super().__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),  # -> 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),             # -> 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),            # -> 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
        )

        # --- Quantization conv ---
        # 将特征映射到 latent_channels*2，用于 mean 和 logvar
        self.quant_conv = nn.Conv2d(256, latent_channels * 2, kernel_size=1)

        # --- Decoder ---
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),            # -> 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # -> 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),

            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1), # -> 3 x 128 x 128
            nn.Sigmoid()
        )

    def preprocess(self, x):
        # 归一化到 [-1, 1]，有助于模型学习颜色对比度
        return 2 * x - 1

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        if self.training:
            logvar = logvar.clamp(-30.0, 20.0)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
        else:
            z = mean
        return z, mean, logvar

    def decode(self, z):
        h = self.post_quant_conv(z)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z, mean, logvar








# ----- Loss Function -----
import pytorch_msssim

def vae_loss(x, x_recon, mean, logvar, kl_weight=0.1):
    recon_loss = 1 - pytorch_msssim.ssim(x, x_recon, data_range=1, size_average=True)
    # recon_loss = F.l1_loss(x, x_recon, reduction='mean')  # 使用L1损失
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    final_loss = recon_loss + kl_weight * kl_loss
    return final_loss, recon_loss, kl_loss

# ----- Checkpoint Save/Load -----
def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

def load_checkpoint(model, optimizer, path="checkpoint.pth", device="cpu"):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch
    else:
        return 0

def validate_vae(model, dataloader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            x = images.to(device)
            x_recon, z, mean, logvar = model(x)
            loss, _, _ = vae_loss(x, x_recon, mean, logvar)
            val_loss += loss.item()
    return val_loss / len(dataloader)

# ----- Training -----
def train_vae(model, dataloader, optimizer, device, num_epochs=1, scheduler=None, val_loader=None, start_epoch=0, checkpoint_path="checkpoint.pth"):
    model.train()
    for epoch in range(start_epoch, num_epochs):
        print("Current learning rate:", optimizer.param_groups[0]['lr'])
        kl_weight = min(0.001, epoch / 20 * 0.001)
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        for images, labels in loop:
            x, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            x_recon, z, mean, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(x, x_recon, mean, logvar, kl_weight=kl_weight)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item(), recon=recon_loss.item(), kl=kl_loss.item(), klw=kl_weight)
        avg_loss = epoch_loss / len(dataloader)
        print("AVG LOSS: {:.6f}".format(avg_loss))
        # 用训练集loss驱动学习率调度
        if scheduler is not None:
            scheduler.step(avg_loss)
        save_checkpoint(model, optimizer, epoch, path=checkpoint_path)
        torch.save(model.state_dict(), "weights.pt")
        torch.cuda.empty_cache()
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = Model().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 提高学习率
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)  # 更好调度
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3)


# 断点续训
start_epoch = load_checkpoint(model, optimizer, path="checkpoint.pth", device=device)

# Train
train_vae(model, train_loader, optimizer, device, num_epochs=230, scheduler=scheduler, start_epoch=start_epoch, checkpoint_path="checkpoint.pth")

# ----- Visualization -----
def plot_reconstructions(model, dataloader, device, num_images=8):
    model.eval()
    with torch.no_grad():
        x = next(iter(dataloader))[0].to(device)
        x_recon, z, _, _ = model(x)
        x = x.cpu().numpy()
        x_recon = x_recon.cpu().numpy()
        print(f"Latent bottleneck dimension: {z.flatten(start_dim=1).shape[1]}")

        plt.figure(figsize=(16, 4))
        for i in range(num_images):
            # Original
            plt.subplot(2, num_images, i+1)
            plt.imshow(x[i].transpose(1, 2, 0))  # (C, H, W) -> (H, W, C)
            plt.axis('off')

            # Reconstruction
            plt.subplot(2, num_images, i+1+num_images)
            plt.imshow(x_recon[i].transpose(1, 2, 0))
            plt.axis('off')

        plt.tight_layout()
        plt.savefig("reconstructions.png")  # 保存图片
        # plt.show()
    torch.cuda.empty_cache()

plot_reconstructions(model, train_loader, device, num_images=8)