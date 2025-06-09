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
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
])
val_transforms = transforms.Compose([transforms.ToTensor()])

os.chdir("/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset")
full_dataset = CustomDataset("train_data_128.npz", transform=None)
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

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_subset,   batch_size=128, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

# ----------------------------
# 2. Center Loss
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
# 3. ConvVAE Model
# ----------------------------
class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=32, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels,  64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64,             128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128,            256, 3, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(),
            nn.Dropout2d(p=0.1),
        )
        self.quant_conv = nn.Conv2d(256, latent_channels*2, 1)

        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256,          128, 3, 1, 1),   nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.ConvTranspose2d(128,  64, 4, 2, 1),   nn.BatchNorm2d(64),  nn.LeakyReLU(),
            nn.ConvTranspose2d(64,   32, 4, 2, 1),   nn.BatchNorm2d(32),  nn.LeakyReLU(),
            nn.ConvTranspose2d(32,   input_channels, 4, 2, 1),           nn.Sigmoid(),
        )

        feat_dim = latent_channels * 16 * 16
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feat_dim, num_classes)
        )

    def preprocess(self, x):
        return 2*x - 1

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        if self.training:
            logvar = logvar.clamp(-10.0, 2.0)
            std = torch.exp(0.5*logvar).clamp(max=2.0)
            z = mean + torch.randn_like(std)*std
            return z, mean, logvar
        else:
            return mean, mean, torch.zeros_like(mean)

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_recon = self.decode(z)
        logits  = self.classifier(z.view(z.size(0), -1))
        return x_recon, z, mean, logvar, logits

# ----------------------------
# 4. Loss & Validation
# ----------------------------
def vae_loss(x, x_recon, mean, logvar, kl_weight):
    recon = F.mse_loss(x, x_recon)
    kl    = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon + kl_weight*kl, recon, kl

def validate(model, loader, device, kl_weight, w_ce, w_cen, w_prc):
    model.eval()
    stats = {'recon':0,'kl':0,'ce':0,'cen':0,'prc':0,'n':0}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            xr, z, m, lv, logits = model(x)
            total, r, k = vae_loss(x, xr, m, lv, kl_weight)
            ce_loss = F.cross_entropy(logits, y)
            zf = z.view(z.size(0), -1)
            cen_loss = center_loss(zf, y)
            # perceptual via pre-extracted features? placeholder MSE on pixels:
            prc_loss = F.mse_loss(xr, x)
            b = x.size(0)
            stats['recon'] += r.item()*b
            stats['kl']    += k.item()*b
            stats['ce']    += ce_loss.item()*b
            stats['cen']   += cen_loss.item()*b
            stats['prc']   += prc_loss.item()*b
            stats['n']     += b
    avg = {k: stats[k]/stats['n'] for k in stats if k!='n'}
    return avg

# ----------------------------
# 5. Training with DWA
# ----------------------------
def train_vae(model, train_loader, val_loader, optimizer, device,
              num_epochs=30, beta_max=0.1, T=2.0):
    model.to(device)
    global center_loss
    feat_dim = model.latent_channels * 16 * 16
    center_loss = CenterLoss(num_classes, feat_dim, device).to(device)
    optimizer.add_param_group({'params': center_loss.parameters()})

    # initialize weights and history
    w_ce, w_cen, w_prc = 0.002, 0.001, 1
    loss_history = {'ce': [], 'cen': [], 'prc': []}

    for epoch in range(1, num_epochs+1):
        # compute DWA weights from history
        if epoch > 2:
            r_ce  = loss_history['ce'][-1]  / loss_history['ce'][-2]
            r_cen = loss_history['cen'][-1] / loss_history['cen'][-2]
            r_prc = loss_history['prc'][-1] / loss_history['prc'][-2]
            w_ce  = np.exp(r_ce  / T)
            w_cen = np.exp(r_cen / T)
            w_prc = np.exp(r_prc / T)
            s = w_ce + w_cen + w_prc
            # normalize such that sum of weights = 3
            w_ce  = 3 * w_ce  / s
            w_cen = 3 * w_cen / s
            w_prc = 3 * w_prc / s

        kl_weight = min(beta_max, beta_max * epoch / num_epochs)
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            xr, z, m, lv, logits = model(x)
            total, r_loss, k_loss = vae_loss(x, xr, m, lv, kl_weight)
            ce_loss  = F.cross_entropy(logits, y)
            zf       = z.view(z.size(0), -1)
            cen_loss = center_loss(zf, y)
            prc_loss = F.mse_loss(xr, x)

            # DWA-weighted sum
            loss = total \
                 + w_ce  * ce_loss \
                 + w_cen * cen_loss \
                 + w_prc * prc_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loop.set_postfix(
                loss=loss.item(),
                recon=r_loss.item(),
                kl=k_loss.item(),
                ce=float(ce_loss.item()),
                cen=float(cen_loss.item()),
                prc=float(prc_loss.item()),
                w_ce=round(w_ce,2),
                w_cen=round(w_cen,2),
                w_prc=round(w_prc,2),
            )

        # validate & record for DWA
        val_stats = validate(model, val_loader, device, kl_weight, w_ce, w_cen, w_prc)
        print(f"[Epoch {epoch}] val_recon={val_stats['recon']:.4f} "
              f"val_ce={val_stats['ce']:.4f} val_cen={val_stats['cen']:.4f} val_prc={val_stats['prc']:.4f}")

        # record losses (use validation CE, center, perc)
        loss_history['ce'].append(val_stats['ce'])
        loss_history['cen'].append(val_stats['cen'])
        loss_history['prc'].append(val_stats['prc'])
        ckpt_name = f"/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4.1_beta/checkpoint/checkpoint_beta0.2_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_name)
        print(f"Saved {ckpt_name}")

# ----------------------------
# 6. Main
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvVAE(3, 32, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train_vae(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=100,
        beta_max=0.1,  # max KL weight
        T=2.0          # temperature for DWA
    )
