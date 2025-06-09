import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

# ─── Encoder: downsample 128→64→32 and stop (spatial=32×32) ───
        self.encoder = nn.Sequential(
            # 128→64
            nn.Conv2d(input_channels,  64, 3, 2, 1),
            nn.BatchNorm2d(64), nn.SiLU(),

            # 64→32
            nn.Conv2d(64,             128, 3, 2, 1),
            nn.BatchNorm2d(128), nn.SiLU(),

            # refine at 32×32 but KEEP SPATIAL
            nn.Conv2d(128,            256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.SiLU(),
            nn.Dropout2d(p=0.1),

            # (optional) extra blocks at 32×32
            nn.Conv2d(256,            256, 3, 1, 1),
            nn.BatchNorm2d(256), nn.SiLU(),
        )

        # now project from 256→ latent_channels*2  @ 32×32
        self.quant_conv = nn.Conv2d(256, latent_channels*2, 1)

        # ─── Decoder: start at 32×32, upsample twice → 64→128 ───
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)
        self.decoder = nn.Sequential(
            # keep 32×32
            nn.Conv2d(256,          128, 3, 1, 1),
            nn.BatchNorm2d(128), nn.SiLU(),

            # ↑ 32→64
            nn.ConvTranspose2d(128,  64, 4, 2, 1),
            nn.BatchNorm2d(64),  nn.SiLU(),

            # ↑ 64→128
            nn.ConvTranspose2d(64,   input_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

        # ─── Classifier on the full 8192-dim z ───
        feat_dim = latent_channels * 32 * 32  # = 8×32×32 = 8192
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feat_dim, num_classes)
        )

    def preprocess(self, x):
        return 2 * x - 1  # [0,1] -> [-1,1]

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        if self.training:
            # training: sample z and return stats for loss
            logvar = logvar.clamp(-30.0, 20.0)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
            return z, mean, logvar
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