import os
import torch
from torchvision import transforms
import random
import numpy as np
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
# Set a fixed random seed for reproducibility
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Mount to dataset directory
os.chdir("/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset")
os.listdir()

if torch.cuda.is_available():
	print(torch.cuda.get_device_name(0))
else:
	print("No CUDA device available")




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
        image = torch.tensor(self.images[idx], dtype=torch.float32) / 255 # convert to [0, 1] range
        label = torch.tensor(self.labels[idx])
        return image, label


train_dataset = CustomDataset("train_data_128.npz")
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

# sample data batch
images, labels = next(iter(train_loader))
print(f"images shape: {images.shape}")
print(f"labels shape: {labels.shape}")



class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=4):
        super().__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),  # 64 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),             # 128 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),            # 256 x 32 x 32
            nn.ReLU()
        )

        # --- Quantization conv ---
        self.quant_conv = nn.Conv2d(256, latent_channels * 2, kernel_size=1)    # 4+4 channels

        # --- Decoder ---
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),            # 128 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # 64 x 64 x 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1), # 3 x 128 x 128
            nn.Sigmoid() # Predict within value range [0, 1]
        )

    def preprocess(self, x):
        return 2 * x - 1 # Value range: [0, 1] => [-1, 1]

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1) # Split along channel (4+4)
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
def vae_loss(x, x_recon, mean, logvar, kl_weight=0.1):
    recon_loss = F.mse_loss(x, x_recon, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    final_loss = recon_loss + kl_weight * kl_loss
    return final_loss, recon_loss, kl_loss

# ----- Training -----
def train_vae(model, dataloader, optimizer, device, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        # loop = dataloader
        for images, labels in loop:
            x, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            x_recon, z, mean, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(x, x_recon, mean, logvar)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item(), recon=recon_loss.item(), kl=kl_loss.item())

# Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = ConvVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train
train_vae(model, train_loader, optimizer, device, num_epochs=5)

# Submission

# 1) Save model weights
torch.save(model.state_dict(), "checkpoint.pt")

# 2) Prepare the 'Model' class for submission
with open("model.py", "r") as f:
    print(f.read())

# 3) Submit the model code & weights online