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
    transforms.RandomRotation(
        15,
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
])
val_transforms = transforms.Compose([transforms.ToTensor()])

# adjust to your dataset path
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
# 2. Center Loss (parametric)
# ----------------------------
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]        # [B, feat_dim]
        diff = features - batch_centers             # [B, feat_dim]
        loss = 0.5 * diff.pow(2).sum() / features.size(0)
        return loss

# ----------------------------
# 3. ConvVAE Model with Uncertainty Weights
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
        self.quant_conv = nn.Conv2d(256, latent_channels*2, 1)

        # Decoder: 16→32→64→128
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)
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

        # ——— uncertainty weights for CE, center, perceptual losses ———
        # learnable log‐variances
        self.log_var_ce     = nn.Parameter(torch.zeros(()))
        self.log_var_center = nn.Parameter(torch.zeros(()))
        self.log_var_perc   = nn.Parameter(torch.zeros(()))

    def preprocess(self, x):
        return 2*x - 1  # [0,1]→[-1,1]

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)

        logvar = logvar.clamp(-10.0, 2.0)
        std = torch.exp(0.5 * logvar).clamp(max=2.0)
        if self.training:
            eps = torch.randn_like(std)
            z = mean + eps * std
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
def vae_loss(x, x_recon, mean, logvar, kl_weight):
    recon = F.mse_loss(x, x_recon, reduction='mean')
    kl    = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon + kl_weight*kl, recon, kl

def validate(model, loader, device, kl_weight, center_loss, vgg, mean_im, std_im):
    model.eval()
    stats = {'recon':0, 'kl':0, 'class':0, 'center':0, 'perc':0, 'n':0}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_recon, z, mean, logvar, logits = model(x)
            total, r, k = vae_loss(x, x_recon, mean, logvar, kl_weight)
            cls_loss = F.cross_entropy(logits, y)
            zf = z.view(z.size(0), -1)
            cen = center_loss(zf, y)

            # perceptual
            x_norm    = (x - mean_im) / std_im
            recon_norm= (x_recon - mean_im) / std_im
            real_feat = vgg(x_norm)
            recon_feat= vgg(recon_norm)
            p_loss    = F.mse_loss(recon_feat, real_feat)

            bsz = x.size(0)
            stats['recon']  += r.item()*bsz
            stats['kl']     += k.item()*bsz
            stats['class']  += cls_loss.item()*bsz
            stats['center'] += cen.item()*bsz
            stats['perc']   += p_loss.item()*bsz
            stats['n']      += bsz

    avg = {k: stats[k]/stats['n'] for k in ['recon','kl','class','center','perc']}
    return avg

# ----------------------------
# 5. Training Loop
# ----------------------------
def train_vae(model, train_loader, val_loader, optimizer, device,
              num_epochs=30, beta_max=0.1,
              vgg=None, mean_im=None, std_im=None):
    model.to(device)
    # instantiate center loss
    feat_dim = model.latent_channels * 16 * 16
    center_loss = CenterLoss(num_classes, feat_dim, device)
    optimizer.add_param_group({'params': center_loss.parameters()})

    best_val, wait, patience = float('inf'), 0, 5

    for epoch in range(1, num_epochs+1):
        model.train()
        kl_weight = min(beta_max, beta_max * epoch / num_epochs)
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            x_recon, z, mean, logvar, logits = model(x)
            total, r_loss, k_loss = vae_loss(x, x_recon, mean, logvar, kl_weight)
            cls_loss = F.cross_entropy(logits, y)
            zf       = z.view(z.size(0), -1)
            cen_loss = center_loss(zf, y)

            # perceptual
            x_norm     = (x - mean_im) / std_im
            recon_norm = (x_recon - mean_im) / std_im
            real_feat  = vgg(x_norm)
            recon_feat = vgg(recon_norm)
            perc_loss  = F.mse_loss(recon_feat, real_feat)

            # uncertainty‐weighted multi‐task loss
            precision_ce     = torch.exp(-model.log_var_ce)
            precision_center = torch.exp(-model.log_var_center)
            precision_perc   = torch.exp(-model.log_var_perc)

            loss = (
                total
                + 0.5 * precision_ce     * cls_loss
                + 0.5 * precision_center * cen_loss
                + 0.5 * precision_perc   * perc_loss
                + 0.5 * (model.log_var_ce
                         + model.log_var_center
                         + model.log_var_perc)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loop.set_postfix(
                loss=loss.item(),
                recon=r_loss.item(),
                kl=k_loss.item(),
                cls=cls_loss.item(),
                cen=cen_loss.item(),
                perc=perc_loss.item(),
                beta=kl_weight
            )

        # validation & early stop
        val_stats = validate(model, val_loader, device, kl_weight,
                              center_loss, vgg, mean_im, std_im)
        val_metric = (
            val_stats['recon']
            + 0.5 * torch.exp(-model.log_var_ce).item()   * val_stats['class']
            + 0.5 * torch.exp(-model.log_var_center).item() * val_stats['center']
            + 0.5 * torch.exp(-model.log_var_perc).item()   * val_stats['perc']
            + 0.5 * (model.log_var_ce.item()
                     + model.log_var_center.item()
                     + model.log_var_perc.item())
        )
        print(f"[Epoch {epoch}] "
              f"val_recon={val_stats['recon']:.4f}  "
              f"val_cls={val_stats['class']:.4f}  "
              f"val_center={val_stats['center']:.4f}  "
              f"val_perc={val_stats['perc']:.4f}")
        ckpt_name = f"/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4.1/checkpoint/checkpoint_beta0.2_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_name)
        print(f"Saved {ckpt_name}")

# ----------------------------
# 6. Main
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # VGG feature extractor (no download)
    vgg = models.vgg16(pretrained=False).features[:16].to(device).eval()
    for p in vgg.parameters():
        p.requires_grad = False
    mean_im = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
    std_im  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)

    model = ConvVAE(3, 32, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train_vae(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=100,
        beta_max=0.0005,
        vgg=vgg,
        mean_im=mean_im,
        std_im=std_im
    )
