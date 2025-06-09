import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),  # 128→64
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.SiLU(),               # 64→32
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.SiLU(), nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.SiLU(),               # refine 32×32
        )
        self.quant_conv = nn.Conv2d(256, latent_channels * 2, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),  # 32→64
            nn.ConvTranspose2d(64, input_channels, 4, 2, 1), nn.Sigmoid(),         # 64→128
        )
        feat_dim = latent_channels * 32 * 32
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(feat_dim, num_classes))

    def preprocess(self, x):
        return 2 * x - 1  # [0,1] → [-1,1]

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        if self.training:
            logvar = logvar.clamp(-10, 2)
            std = torch.exp(0.5 * logvar).clamp(max=2)
            eps = torch.randn_like(std)
            return mean + eps * std, mean, logvar
        else:
            return mean

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_rec = self.decode(z)
        logits = self.classifier(z.view(z.size(0), -1))
        return x_rec, z, mean, logvar, logits