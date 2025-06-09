import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

        # ─── Encoder: five conv blocks ───
        #   -- first two downsample to 32×32, next three are stride‐1 refinements
        self.encoder = nn.Sequential(
            # 128→64
            nn.Conv2d(input_channels,  64, 3, 2, 1),
            nn.BatchNorm2d(64), nn.SiLU(),
            # 64→32
            nn.Conv2d(64,            128, 3, 2, 1),
            nn.BatchNorm2d(128), nn.SiLU(),
            # refine @32×32 (3 blocks)
            nn.Conv2d(128,           256, 3, 1, 1),
            nn.BatchNorm2d(256), nn.SiLU(),

            nn.Conv2d(256,           256, 3, 1, 1),
            nn.BatchNorm2d(256), nn.SiLU(),

            nn.Conv2d(256,           256, 3, 1, 1),
            nn.BatchNorm2d(256), nn.SiLU(),
        )

        # ─── Quantization ───
        # projects 256→ latent*2 for mean + logvar
        self.quant_conv = nn.Conv2d(256, latent_channels*2, 1)

        # ─── Decoder: five blocks ───
        #   -- first three are stride‐1 refinements @32×32, then two upsample steps
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)
        self.decoder = nn.Sequential(
            # refine @32×32 (3 blocks)
            nn.Conv2d(256,           256, 3, 1, 1),
            nn.BatchNorm2d(256), nn.SiLU(),

            nn.Conv2d(256,           256, 3, 1, 1),
            nn.BatchNorm2d(256), nn.SiLU(),

            nn.Conv2d(256,           128, 3, 1, 1),
            nn.BatchNorm2d(128), nn.SiLU(),

            # upsample 32→64
            nn.ConvTranspose2d(128,  64, 4, 2, 1),
            nn.BatchNorm2d(64),  nn.SiLU(),
            # upsample 64→128 & output
            nn.ConvTranspose2d(64,   input_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

        # classifier (unchanged)
        feat_dim = latent_channels * 32 * 32  # = 8192
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feat_dim, num_classes)
        )

    def encode(self, x):
        h = self.encoder(2*x - 1)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        if self.training:
            std = (0.5*logvar).exp()
            z   = mean + torch.randn_like(std)*std
            return z, mean, logvar
        return mean

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_recon         = self.decode(z)
        logits          = self.classifier(z.view(z.size(0), -1))
        return x_recon, z, mean, logvar, logits