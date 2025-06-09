import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torch.optim as optim
from sklearn.model_selection import train_test_split
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

# --- Load and Split Dataset ---
os.chdir("/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset")  # adjust as needed
full_dataset = CustomDataset("train_data_128.npz")
num_classes = int(np.unique(full_dataset.labels).shape[0])

# Train/Validation split (10% validation)
train_idx, val_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.1,
    random_state=42,
    stratify=full_dataset.labels
)
train_subset = Subset(full_dataset, train_idx)
val_subset = Subset(full_dataset, val_idx)

train_loader = DataLoader(
    train_subset, batch_size=128, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True
)
val_loader = DataLoader(
    val_subset, batch_size=128, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True
)

# --- Model Definition with Enhanced Latent Capacity ---
class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=32, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: downsample 128->64->32->16 with BatchNorm
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels,  64, kernel_size=3, stride=2, padding=1),  # ->64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64,             128, kernel_size=3, stride=2, padding=1),  # ->128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128,            256, kernel_size=3, stride=2, padding=1),  # ->256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        # Project to 2 * latent_channels for mean/logvar
        self.quant_conv = nn.Conv2d(256, latent_channels * 2, kernel_size=1)

        # Decoder: upsample 16->32->64->128 with BatchNorm
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256,          128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128,  64,  kernel_size=4, stride=2, padding=1),   # ->64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64,   32,  kernel_size=4, stride=2, padding=1),   # ->32 x 64 x 64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32,   input_channels, kernel_size=4, stride=2, padding=1),  # ->3 x 128 x 128
            nn.Sigmoid(),
        )

        # Classifier on flattened latent: 32 * 16 * 16 = 8192 dims
        latent_dim = latent_channels * 16 * 16
        self.classifier = nn.Linear(latent_dim, num_classes)

    def preprocess(self, x):
        return 2 * x - 1  # [0,1] -> [-1,1]

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)

        if self.training or self.eval:
            logvar = logvar.clamp(-30.0, 20.0)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
            return z, mean, logvar
        else:
            return mean

    def decode(self, z):
        h = self.post_quant_conv(z)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_recon         = self.decode(z)
        logits          = self.classifier(z.view(z.size(0), -1))
        return x_recon, z, mean, logvar, logits


# --- Loss Functions ---
def vae_loss(x, x_recon, mean, logvar, kl_weight=0.1):
    recon_loss = F.mse_loss(x, x_recon, reduction='mean')
    kl_loss    = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss

# --- Validation Function ---
def validate(model, loader, device, kl_weight, alpha):
    model.eval()
    metrics = {'recon': 0.0, 'kl': 0.0, 'class': 0.0, 'n': 0}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_recon, z, mean, logvar, logits = model(x)
            _, recon_loss, kl_loss = vae_loss(x, x_recon, mean, logvar, kl_weight)
            class_loss = F.cross_entropy(logits, y)
            bsz = x.size(0)
            metrics['recon'] += recon_loss.item() * bsz
            metrics['kl']    += kl_loss.item() * bsz
            metrics['class'] += class_loss.item() * bsz
            metrics['n']     += bsz
    return {k: metrics[k]/metrics['n'] for k in ['recon','kl','class']}

# --- Training Loop with β-Annealing, Joint Classification & Validation ---
def train_vae(model, train_loader, val_loader, optimizer, device,
              num_epochs=10, beta_max=0.1, alpha=0.5):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        kl_weight = min(beta_max, beta_max * (epoch + 1) / num_epochs)
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in loop:
            x, y = images.to(device), labels.to(device)
            optimizer.zero_grad()
            x_recon, z, mean, logvar, logits = model(x)

            vae_total, recon_loss, kl_loss = vae_loss(x, x_recon, mean, logvar, kl_weight)
            class_loss = F.cross_entropy(logits, y)
            loss = vae_total + alpha * class_loss

            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item(), recon=recon_loss.item(),
                              kl=kl_loss.item(), class_loss=class_loss.item(),
                              beta=kl_weight)

        # Run validation at epoch end
        val_metrics = validate(model, val_loader, device, kl_weight, alpha)
        print(f"Validation - recon: {val_metrics['recon']:.4f}, kl: {val_metrics['kl']:.4f},"
              f" class: {val_metrics['class']:.4f}")
        ckpt_name = f"/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v2.1/checkpoint/checkpoint_beta0.2_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_name)
        print(f"Saved {ckpt_name}")

# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvVAE(input_channels=3, latent_channels=32, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Train with validation
    train_vae(model, train_loader, val_loader, optimizer, device,
              num_epochs=30, beta_max=0.0005, alpha=0.002)


