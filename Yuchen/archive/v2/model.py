import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=32, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: downsample 128->64->32->16
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels,  64, kernel_size=3, stride=2, padding=1),  # ->64 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(64,             128, kernel_size=3, stride=2, padding=1),  # ->128 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(128,            256, kernel_size=3, stride=2, padding=1),  # ->256 x 16 x 16
            nn.ReLU(),
        )
        # Project to 2 * latent_channels for mean/logvar
        self.quant_conv = nn.Conv2d(256, latent_channels * 2, kernel_size=1)

        # Decoder: upsample 16->32->64->128
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256,          128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,  64,  kernel_size=4, stride=2, padding=1),   # ->64 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(64,   32,  kernel_size=4, stride=2, padding=1),   # ->32 x 64 x 64
            nn.ReLU(),
            nn.ConvTranspose2d(32,   input_channels, kernel_size=4, stride=2, padding=1),  # ->3 x 128 x 128
            nn.Sigmoid(),
        )

        # Classifier on flattened latent: 32 * 16 * 16 = 8192 dims
        latent_dim = latent_channels * 16 * 16
        self.classifier = nn.Linear(latent_dim, num_classes)

    def preprocess(self, x):
        return 2 * x - 1  # [0,1] -> [-1,1]

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
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
        return self.decoder(h)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_recon = self.decode(z)
        logits  = self.classifier(z.view(z.size(0), -1))
        return x_recon, z, mean, logvar, logits
