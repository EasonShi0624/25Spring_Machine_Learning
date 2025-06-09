import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    """
    A standard residual block with two 3x3 conv layers and SiLU activation.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.act   = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return self.act(out)

class SelfAttention(nn.Module):
    """
    Self-Attention (as in SAGAN) to capture long-range dependencies.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim,     kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, C, W, H = x.size()
        # Project to query, key, value
        proj_query = self.query_conv(x).view(batch, -1, W*H)       # B x C' x N
        proj_key   = self.key_conv(x).view(batch, -1, W*H)         # B x C' x N
        energy     = torch.bmm(proj_query.permute(0,2,1), proj_key) # B x N x N
        attention  = self.softmax(energy)                          # B x N x N
        proj_value = self.value_conv(x).view(batch, -1, W*H)       # B x C x N

        out = torch.bmm(proj_value, attention.permute(0,2,1))      # B x C x N
        out = out.view(batch, C, W, H)
        # Weighted residual connection
        out = self.gamma * out + x
        return out

class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: add ResidualBlocks and SelfAttention
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 2, 1),  # 128→64
            nn.BatchNorm2d(64), nn.SiLU(),

            nn.Conv2d(64, 128, 3, 2, 1),              # 64→32
            nn.BatchNorm2d(128), nn.SiLU(),

            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256), nn.SiLU(), nn.Dropout2d(0.1),
            ResidualBlock(256),
            SelfAttention(256),

            nn.Conv2d(256,256,3,1,1),                 # refine 32×32
            nn.BatchNorm2d(256), nn.SiLU(),
            nn.Conv2d(256,256,3,1,1),                 # NEW
            nn.BatchNorm2d(256), nn.SiLU()
        )

        # Bottleneck
        self.quant_conv      = nn.Conv2d(256, latent_channels * 2, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, 1)

        # Decoder: add ResidualBlocks and SelfAttention
        self.decoder = nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256), nn.SiLU(),
            ResidualBlock(256),
            SelfAttention(256),

            nn.Conv2d(256,128,3,1,1),
            nn.BatchNorm2d(128), nn.SiLU(),
            ResidualBlock(128),
            SelfAttention(128),

            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),      # 32→64
            nn.BatchNorm2d(64), nn.SiLU(),

            nn.ConvTranspose2d(64, input_channels, 4, 2, 1),
            nn.Sigmoid()
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(latent_channels * 32 * 32, num_classes)
        )

    def preprocess(self, x):
        return 2*x - 1  # normalize to [-1,1]

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        if self.training:
            logvar = logvar.clamp(-10.0, 2.0)
            std    = torch.exp(0.5 * logvar).clamp(max=2.0)
            eps    = torch.randn_like(std)
            z      = mean + eps * std
            return z, mean, logvar
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
