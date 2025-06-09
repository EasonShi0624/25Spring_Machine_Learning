import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialSelfAttention(nn.Module):
    """
    Lightweight multi‑head self‑attention that operates on a 16×16 latent map.
    Uses torch.nn.MultiheadAttention under the hood.
    """
    def __init__(self, channels, heads=4):
        super().__init__()
        self.channels = channels
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            batch_first=True
        )

    def forward(self, x):
        # x : [B, C, H, W]  (H=W=16 here)
        B, C, H, W = x.shape
        seq = x.flatten(2).permute(0, 2, 1)      # [B, HW, C]
        out, _ = self.attn(seq, seq, seq)        # self‑attention
        out = out + seq                          # residual
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


class Model(nn.Module):
    def __init__(self, input_channels=3, latent_channels=32, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: 128→64→32→16
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels,  64, 3, 2, 1), nn.BatchNorm2d(64),  nn.PReLU(),
            nn.Conv2d(64,             128, 3, 2, 1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128,            256, 3, 2, 1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Dropout2d(p=0.1),
        )
        self.quant_conv = nn.Conv2d(256, latent_channels * 2, 1)

        # ✦ NEW: self‑attention over the 16×16 bottleneck map ✦
        self.attn = SpatialSelfAttention(latent_channels * 2, heads=4)

        # Decoder: 16→32→64→128
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256,          128, 3, 1, 1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.BatchNorm2d(64),  nn.PReLU(),
            nn.ConvTranspose2d(64,   32, 4, 2, 1), nn.BatchNorm2d(32),  nn.PReLU(),
            nn.ConvTranspose2d(32,   input_channels, 4, 2, 1),         nn.Sigmoid(),
        )

        # Classifier on flattened latent
        feat_dim = latent_channels * 16 * 16
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feat_dim, num_classes)
        )

    # ---------- unchanged helper methods ----------
    def preprocess(self, x):
        return 2 * x - 1                        # [0,1] → [-1,1]

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
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_recon         = self.decode(z)
        logits          = self.classifier(z.reshape(z.size(0), -1))
        return x_recon, z, mean, logvar, logits