import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_channels=3, latent_channels=32, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 2, 1),
            nn.BatchNorm2d(64), nn.LeakyReLU(),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256), nn.LeakyReLU(),
            nn.Dropout2d(0.1)
        )
        self.quant_conv = nn.Conv2d(256, latent_channels * 2, 1)

        # ---------- Decoder ----------
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.LeakyReLU(),

            nn.ConvTranspose2d(32, input_channels, 4, 2, 1),
            nn.Sigmoid()
        )

        # ---------- Probing head ----------
        feat_dim = latent_channels * 16 * 16
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, num_classes)
        )

    # -------------------------------------------------------------
    def preprocess(self, x):
        return 2 * x - 1  # [0,1] → [-1,1]

    def encode(self, x):
        """Return z, mean, logvar  (always three tensors)."""
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)

        if self.training:
            logvar = logvar.clamp(-30.0, 20.0)
            std = torch.exp(0.5 * logvar)
            z = mean + torch.randn_like(std) * std
        else:
            z      = mean                     # deterministic in eval
            logvar = torch.zeros_like(mean)   # placeholder

        return z, mean, logvar

    def decode(self, z):
        h = self.post_quant_conv(z)
        return self.decoder(h)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_recon = self.decode(z)
        logits  = self.classifier(z.view(z.size(0), -1))
        return x_recon, z, mean, logvar, logits
