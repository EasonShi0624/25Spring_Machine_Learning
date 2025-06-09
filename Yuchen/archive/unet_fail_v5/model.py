import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, input_channels=3, latent_channels=32, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

        # ---- Encoder blocks ----
        self.enc1 = nn.Sequential(                       # 128 → 64
            nn.Conv2d(input_channels, 64, 3, 2, 1),
            nn.BatchNorm2d(64), nn.LeakyReLU())
        self.enc2 = nn.Sequential(                       # 64 → 32
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU())
        self.enc3 = nn.Sequential(                       # 32 → 16
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256), nn.LeakyReLU(),
            nn.Dropout2d(p=0.1))

        # project to mean / logvar
        self.quant_conv = nn.Conv2d(256, latent_channels * 2, 1)

        # ---- Decoder blocks (transpose‑convs + skip fusion) ----
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)

        # 16×16 → 32×32
        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up1_fuse = nn.Sequential(                   # cat(128,128)=256
            nn.BatchNorm2d(256), nn.LeakyReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),               # back to 128
            nn.BatchNorm2d(128), nn.LeakyReLU())

        # 32×32 → 64×64
        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up2_fuse = nn.Sequential(                   # cat(64,64)=128
            nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64), nn.LeakyReLU())

        # 64×64 → 128×128
        self.up3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.out_conv = nn.Sequential(
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, input_channels, 3, 1, 1),
            nn.Sigmoid())                                # final output

        # ---- Classifier on flattened latent map ----
        feat_dim = latent_channels * 16 * 16             # 8192
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, num_classes)
        )

    # ----------------------------------------------
    # helpers
    # ----------------------------------------------
    @staticmethod
    def _preprocess(x):                                  # [0,1] → [‑1,1]
        return 2 * x - 1

    # ----------------------------------------------
    # encode() memorises the skips in self._skips
    # ----------------------------------------------
    def encode(self, x):
        x = self._preprocess(x)
        s1 = self.enc1(x)     # 64×64×64
        s2 = self.enc2(s1)    # 128×32×32
        h  = self.enc3(s2)    # 256×16×16
        self._skips = (s1, s2)

        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)

        if self.training:
            logvar = logvar.clamp(-10, 2)
            std = torch.exp(0.5 * logvar).clamp(max=2.0)
            z = mean + torch.randn_like(std) * std
            return z, mean, logvar
        else:  # eval / inference
            z = mean
            return z, mean, logvar

    # ----------------------------------------------
    # decode() fetches the stored skips
    # ----------------------------------------------
    def decode(self, z):
        s1, s2 = self._skips            # 64×64×64  and 128×32×32
        h = self.post_quant_conv(z)     # 256×16×16

        h = self.up1(h)                 # 128×32×32
        h = torch.cat([h, s2], dim=1)   # 256×32×32
        h = self.up1_fuse(h)            # 128×32×32

        h = self.up2(h)                 # 64×64×64
        h = torch.cat([h, s1], dim=1)   # 128×64×64
        h = self.up2_fuse(h)            # 64×64×64

        h = self.up3(h)                 # 32×128×128
        return self.out_conv(h)         # 3×128×128

    # ----------------------------------------------
    # forward() unchanged for outside callers
    # ----------------------------------------------
    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_recon = self.decode(z)
        logits  = self.classifier(z.view(z.size(0), -1))
        return x_recon, z, mean, logvar, logits