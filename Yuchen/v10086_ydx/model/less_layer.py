import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    A standard residual block with two 3x3 conv layers and Mish activation.
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

import math

class SelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads=4):
        super().__init__()
        assert in_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = in_dim // num_heads

        # one proj for all heads, then reshape
        self.to_qkv = nn.Conv2d(in_dim, in_dim * 3, 1, bias=False)
        self.unify_heads = nn.Conv2d(in_dim, in_dim, 1)
        self.norm1 = nn.GroupNorm(1, in_dim)
        self.norm2 = nn.GroupNorm(1, in_dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim*4, 1),
            nn.SiLU(),
            nn.Conv2d(in_dim*4, in_dim, 1),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        # Pre-norm
        x_norm = self.norm1(x)

        # QKV and split heads
        qkv = self.to_qkv(x_norm)                   # B, 3C, H, W
        q, k, v = qkv.chunk(3, dim=1)                # each B, C, H, W
        # reshape to (B, heads, d_k, N)
        def reshape(t):
            t = t.view(B, self.num_heads, self.d_k, H*W)
            return t
        q, k, v = reshape(q), reshape(k), reshape(v)

        # scaled dot-product
        scores = torch.einsum('bhdi,bhdj->bhij', q, k) / math.sqrt(self.d_k)
        attn   = torch.softmax(scores, dim=-1)        # B, heads, N, N

        out = torch.einsum('bhij,bhdj->bhdi', attn, v)  # B, heads, d_k, N
        out = out.contiguous().view(B, C, H, W)         # concat heads

        attn_out = self.unify_heads(out)
        x2 = x + self.gamma * attn_out                  # residual

        # FFN block
        x2_norm = self.norm2(x2)
        ffn_out = self.ffn(x2_norm)
        return x2 + ffn_out


class ConvAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=8, num_classes=170):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: same as before
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 2, 1),  # 128→64
            nn.BatchNorm2d(64), nn.Mish(),

            nn.Conv2d(64, 128, 3, 2, 1),              # 64→32
            nn.BatchNorm2d(128), nn.Mish(),

            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256), nn.Mish(), nn.Dropout2d(0.1),

            ResBlock(256),                  # 1st ResBlock at 32×32
            SelfAttention(256),             # 1st SelfAttention at 32×32
            ResBlock(256)

        )

        # ——— Bottleneck (deterministic) ———
        # map 256→latent_channels
        self.bottleneck_conv      = nn.Conv2d(256, latent_channels*2, 1)
        # map latent_channels→256 for decoder
        self.post_bottleneck_conv = nn.Conv2d(latent_channels, 256, 1)
        self.decoder = nn.Sequential(
            # — refine at 32×32, channels 256 → 256
            ResBlock(256),                  # 3rd ResBlock at 32×32
            SelfAttention(256),             # 2nd SelfAttention at 32×32
            ResBlock(256),  

            # — refine + reduce channels at 32×32: 256 → 128
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.Mish(),


            # — up-sample 32→64, channels 128 → 64
            nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(64),  nn.Mish(),


            # — final refine to RGB (or input_channels) at 128×128
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()            # map back to [0,1]
        )
        # Decoder: same as before
        # pooling + classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),  # [B, latent, H, W] → [B, latent, 1, 1]
            nn.Flatten(),                 # → [B, latent]
            nn.Linear(latent_channels, num_classes)
        )

    def preprocess(self, x):
        return 2*x - 1  # if you still want to normalize to [-1,1]

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.bottleneck_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        if self.training:
            logvar = logvar.clamp(-10.0, 2.0)
            std    = torch.exp(0.5 * logvar).clamp(max=2.0)
            eps    = torch.randn_like(std)
            z      = mean + eps * std
            return z, mean, logvar
        else:
            z = mean
            return z

    def decode(self, z):
        h = self.post_bottleneck_conv(z)
        return self.decoder(h)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_recon         = self.decode(z)
        logits          = self.classifier(z)
        return x_recon, z, logits