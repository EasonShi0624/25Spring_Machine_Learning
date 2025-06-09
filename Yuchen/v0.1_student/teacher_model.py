import argparse, pathlib, os, math, random, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F, torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from tqdm import tqdm

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1); self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1); self.bn2 = nn.BatchNorm2d(c)
        self.act = nn.Mish()
    def forward(self,x):
        out=self.act(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        return self.act(out+x)

class SelfAttn(nn.Module):
    def __init__(self, c, heads=8): super().__init__()
    def forward(self,x):
        B,C,H,W=x.shape; q=x.flatten(2).permute(0,2,1)
        attn = torch.softmax(q @ q.transpose(-1,-2) / math.sqrt(C), -1)
        out  = (attn @ q).permute(0,2,1).view(B,C,H,W)
        return out + x

# ----------------------------
# 3. Teacher VAE
# ----------------------------
class TeacherVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=64, num_classes=170):
        super().__init__()
        self.lat_c=latent_channels
        # encoder 128→64→32→16 with residuals
        self.enc1 = nn.Sequential(nn.Conv2d(input_channels,64,3,2,1), nn.BatchNorm2d(64), nn.Mish(), ResBlock(64))
        self.enc2 = nn.Sequential(nn.Conv2d(64,128,3,2,1), nn.BatchNorm2d(128),nn.Mish(),ResBlock(128))
        self.enc3 = nn.Sequential(nn.Conv2d(128,256,3,2,1),nn.BatchNorm2d(256),nn.Mish(),ResBlock(256))
        self.quant = nn.Conv2d(256, latent_channels*2, 1)
        self.attn  = SelfAttn(latent_channels*2, heads=8)

        # decoder with U‑Net skips
        self.up1 = nn.Sequential(nn.ConvTranspose2d(latent_channels,128,4,2,1),nn.BatchNorm2d(128),nn.Mish(),ResBlock(128))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128+128,64,4,2,1), nn.BatchNorm2d(64), nn.Mish(),ResBlock(64))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(64+64, input_channels,4,2,1), nn.Sigmoid())

        self.cls = nn.Sequential(nn.Dropout(0.3), nn.Linear(latent_channels*16*16, num_classes))

    def encode(self,x):
        h1=self.enc1(x)
        h2=self.enc2(h1)
        h3=self.enc3(h2)
        h = self.attn(self.quant(h3))
        mu,lv=torch.chunk(h,2,1); self._skips=(h1,h2)
        if self.training:
            lv=lv.clamp(-8,2); z=mu+torch.randn_like(mu)*torch.exp(0.5*lv)
            return z,mu,lv
        else: return mu,mu,lv

    def decode(self,z):
        h1,h2=self._skips
        x = self.up1(z)
        x = self.up2(torch.cat([x,h2],1))
        x = self.up3(torch.cat([x,h1],1))
        return x

    def forward(self,x):
        z,mu,lv=self.encode(x)
        xr=self.decode(z)
        logits=self.cls(z.reshape(z.size(0),-1))
        return xr,z,mu,lv,logits