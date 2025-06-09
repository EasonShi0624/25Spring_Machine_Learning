import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
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
        # convert to H×W×C uint8 for PIL
        img_np = (self.images[idx].transpose(1,2,0) * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        if self.transform:
            img = self.transform(img)    # -> Tensor [0,1] C×H×W
        else:
            img = transforms.ToTensor()(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(
        15,
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])
val_transforms = transforms.Compose([
    transforms.ToTensor(),
])

# adjust path as needed
os.chdir("/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset")
full_dataset = CustomDataset("train_data_128.npz", transform=None)  # assign transforms below
num_classes = int(np.unique(full_dataset.labels).shape[0])

# stratified train/val split
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

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_subset,   batch_size=128, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

# ----------------------------
# 2. Center Loss Definition
# ----------------------------
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        """
        Pure parametric center loss: centers are nn.Parameters
        and updated by gradient descent.
        """
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device))

    def forward(self, features, labels):
        # features: [B, feat_dim], labels: [B]
        batch_centers = self.centers[labels]         # [B, feat_dim]
        diff = features - batch_centers              # [B, feat_dim]
        loss = 0.5 * diff.pow(2).sum() / features.size(0)
        return loss

# ----------------------------
# 3. ConvVAE Model
# ----------------------------
class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=32, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: 128→64→32→16
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels,  64, 3, 2, 1),  nn.BatchNorm2d(64),  nn.LeakyReLU(),
            nn.Conv2d(64,             128, 3, 2, 1),  nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128,            256, 3, 2, 1),  nn.BatchNorm2d(256), nn.LeakyReLU(),
            nn.Dropout2d(p=0.1),
        )
        self.quant_conv = nn.Conv2d(256, latent_channels*2, kernel_size=1)

        # Decoder: 16→32→64→128
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256,          128, 3, 1, 1),    nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.ConvTranspose2d(128,  64, 4, 2, 1),    nn.BatchNorm2d(64),  nn.LeakyReLU(),
            nn.ConvTranspose2d(64,   32, 4, 2, 1),    nn.BatchNorm2d(32),  nn.LeakyReLU(),
            nn.ConvTranspose2d(32,   input_channels, 4, 2, 1),            nn.Sigmoid(),
        )

        # Classifier on flattened latent
        feat_dim = latent_channels * 16 * 16
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feat_dim, num_classes)
        )

    def preprocess(self, x):
        return 2*x - 1  # [0,1]→[-1,1]

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)

        if self.training:
            # stochastic sampling for training
            logvar = logvar.clamp(-10.0, 2.0)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
            return z, mean, logvar
        else:
            # deterministic for eval/inference: z = mean, but still return triple
            z = mean
            logvar = torch.zeros_like(mean)
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
def vae_loss(x, x_recon, mean, logvar, kl_weight=0.1):
    recon_loss = F.mse_loss(x, x_recon, reduction='mean')
    kl_loss    = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_weight*kl_loss, recon_loss, kl_loss

def validate(model, loader, device, kl_weight, alpha, gamma, center_loss):
    model.eval()
    stats = {'recon':0, 'kl':0, 'class':0, 'center':0, 'n':0}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_recon, z, mean, logvar, logits = model(x)
            total, r, k = vae_loss(x, x_recon, mean, logvar, kl_weight)
            c_loss = F.cross_entropy(logits, y)
            z_flat = z.view(z.size(0), -1)
            cen    = center_loss(z_flat, y)
            bsz = x.size(0)
            stats['recon']  += r.item()*bsz
            stats['kl']     += k.item()*bsz
            stats['class']  += c_loss.item()*bsz
            stats['center'] += cen.item()*bsz
            stats['n']      += bsz
    return {k: stats[k]/stats['n'] for k in ['recon','kl','class','center']}

# ----------------------------
# 5. Training Loop
# ----------------------------
def train_vae(model, train_loader, val_loader, optimizer, device,
              num_epochs=30, beta_max=0.1, alpha=0.5, gamma=0.1):
    model.to(device)
    # instantiate CenterLoss updater inside loop scope
    feat_dim = model.latent_channels * 16 * 16
    center_loss = CenterLoss(num_classes, feat_dim, device)
    # extend optimizer to center parameters
    optimizer.add_param_group({'params': center_loss.parameters()})

    best_val = float('inf')
    patience, wait = 5, 0

    for epoch in range(1, num_epochs+1):
        model.train()
        kl_weight = min(beta_max, beta_max * epoch / num_epochs)
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            x_recon, z, mean, logvar, logits = model(x)
            total, r_loss, k_loss = vae_loss(x, x_recon, mean, logvar, kl_weight)
            cls_loss  = F.cross_entropy(logits, y)
            z_flat    = z.view(z.size(0), -1)
            cen_loss  = center_loss(z_flat, y)

            loss = total + alpha*cls_loss + gamma*cen_loss
            loss.backward()
            optimizer.step()

            loop.set_postfix(
                loss=loss.item(),
                recon=r_loss.item(),
                kl=k_loss.item(),
                cls=cls_loss.item(),
                cen=cen_loss.item(),
                beta=kl_weight
            )

        # end epoch: validate & early stop
        val_stats = validate(model, val_loader, device, kl_weight, alpha, gamma, center_loss)
        val_metric = val_stats['recon'] + alpha*val_stats['class'] + gamma*val_stats['center']
        print(f"[{epoch}] val_recon={val_stats['recon']:.4f}, val_cls={val_stats['class']:.4f}, val_center={val_stats['center']:.4f}")

        ckpt_name = f"/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v3/checkpoint_larger_gamma/checkpoint_beta0.2_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_name)
        print(f"Saved {ckpt_name}")
# ----------------------------
# 6. Main
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvVAE(input_channels=3, latent_channels=32, num_classes=num_classes).to(device)
    # Instantiate CenterLoss and include its parameters in the optimizer
    feat_dim = model.latent_channels * 16 * 16
    center_loss = CenterLoss(num_classes, feat_dim, device)
    optimizer = optim.AdamW([
        {'params': model.parameters(),       'lr': 1e-3},
        {'params': center_loss.parameters(), 'lr': 1e-2}
    ], weight_decay=1e-5)


    train_vae(
         model, train_loader, val_loader, optimizer, device,
         num_epochs=100,
         beta_max=0.0005,    # max KL weight
         alpha=0.002,        # classification loss weight
         gamma=0.001         # center loss weight
     )


