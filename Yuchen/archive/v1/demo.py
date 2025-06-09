import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

# Set deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Dataset Definition ---
class CustomDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.images = data['images']  # shape: (N, 3, 128, 128)
        self.labels = data['labels']  # shape: (N,)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# --- Data Loading ---
os.chdir("/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset")  # adjust as needed
train_dataset = CustomDataset("train_data_128.npz")
num_classes = int(np.unique(train_dataset.labels).shape[0])
train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True
)

# --- Model Definition ---
class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=4, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),  # 64 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),             # 128 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),            # 256 x 32 x 32
            nn.ReLU()
        )
        # Quantization conv
        self.quant_conv = nn.Conv2d(256, latent_channels * 2, kernel_size=1)

        # Decoder
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 64 x 64 x 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),  # 3 x 128 x 128
            nn.Sigmoid()
        )

        # Classifier head on flattened latent
        latent_dim = latent_channels * 32 * 32
        self.classifier = nn.Linear(latent_dim, num_classes)

    def preprocess(self, x):
        return 2 * x - 1  # [0,1] -> [-1,1]

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
        logits = self.classifier(z.view(z.size(0), -1))
        return x_recon, z, mean, logvar, logits

# --- Loss Functions ---
def vae_loss(x, x_recon, mean, logvar, kl_weight=0.1):
    recon_loss = F.mse_loss(x, x_recon, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    total_vae = recon_loss + kl_weight * kl_loss
    return total_vae, recon_loss, kl_loss

# --- Training Loop with beta-Annealing and Joint Classification ---
def train_vae(model, dataloader, optimizer, device, num_epochs=10,
              beta_max=0.1, alpha=0.5):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        # Linear beta-annealing schedule
        kl_weight = min(beta_max, beta_max * (epoch + 1) / num_epochs)
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in loop:
            x, y = images.to(device), labels.to(device)
            optimizer.zero_grad()
            x_recon, z, mean, logvar, logits = model(x)

            # VAE losses with dynamic KL weight
            vae_total, recon_loss, kl_loss = vae_loss(x, x_recon, mean, logvar, kl_weight)
            # Classification loss
            class_loss = F.cross_entropy(logits, y)
            # Combined loss
            loss = vae_total + alpha * class_loss

            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item(), recon=recon_loss.item(),
                              kl=kl_loss.item(), class_loss=class_loss.item(),
                              beta=kl_weight)

# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvVAE(input_channels=3, latent_channels=4, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train
    train_vae(model, train_loader, optimizer, device,
              num_epochs=50, beta_max=0.1, alpha=1000)
    os.chdir("/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v1/checkpoint")
    # Save model weights
    torch.save(model.state_dict(), "checkpoint1.pt")
