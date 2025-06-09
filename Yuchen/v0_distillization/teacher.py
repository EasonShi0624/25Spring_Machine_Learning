#!/usr/bin/env python
# train_teacher.py  – large teacher VAE for distillation
import argparse, pathlib, os, math, random, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F, torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from tqdm import tqdm

# -------------------------------------------------
# 0. CLI
# -------------------------------------------------
cli = argparse.ArgumentParser()
cli.add_argument("--resume", type=pathlib.Path, default=None,
                 help="checkpoint.pt to resume from")
cli.add_argument("--epochs", type=int, default=500)
args = cli.parse_args()

# ----------------------------
# 1. Dataset & Augmentations
# ----------------------------
class NPZDataset(Dataset):
    def __init__(self, path, tfm=None):
        d = np.load(path)
        self.img, self.lab = d["images"], d["labels"]
        self.tfm = tfm or transforms.ToTensor()
    def __len__(self):  return len(self.img)
    def __getitem__(self, i):
        im = Image.fromarray((self.img[i].transpose(1,2,0)*255).astype(np.uint8))
        return self.tfm(im), torch.tensor(self.lab[i], dtype=torch.long)

train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.25,0.25,0.25,0.1), transforms.ToTensor()
])
val_tfm = transforms.Compose([transforms.ToTensor()])

os.chdir("/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset")
ds_full = NPZDataset("train_data_128.npz")
num_cls = int(np.unique(ds_full.lab).shape[0])

id_train, id_val = train_test_split(range(len(ds_full)), test_size=0.1,
                                    stratify=ds_full.lab, random_state=42)
train_ds = Subset(ds_full, id_train); train_ds.dataset.tfm = train_tfm
val_ds   = Subset(ds_full, id_val);   val_ds.dataset.tfm   = val_tfm

train_loader = DataLoader(train_ds, 256, True,  num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_ds,   256, False, num_workers=8, pin_memory=True)

# ----------------------------
# 2. Blocks
# ----------------------------
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
class Teacher(nn.Module):
    def __init__(self, in_c=3, lat_c=64, num_cls=num_cls):
        super().__init__()
        self.lat_c=lat_c
        # encoder 128→64→32→16 with residuals
        self.enc1 = nn.Sequential(nn.Conv2d(in_c,64,3,2,1), nn.BatchNorm2d(64), nn.Mish(), ResBlock(64))
        self.enc2 = nn.Sequential(nn.Conv2d(64,128,3,2,1), nn.BatchNorm2d(128),nn.Mish(),ResBlock(128))
        self.enc3 = nn.Sequential(nn.Conv2d(128,256,3,2,1),nn.BatchNorm2d(256),nn.Mish(),ResBlock(256))
        self.quant = nn.Conv2d(256, lat_c*2, 1)
        self.attn  = SelfAttn(lat_c*2, heads=8)

        # decoder with U‑Net skips
        self.up1 = nn.Sequential(nn.ConvTranspose2d(lat_c,128,4,2,1),nn.BatchNorm2d(128),nn.Mish(),ResBlock(128))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128+128,64,4,2,1), nn.BatchNorm2d(64), nn.Mish(),ResBlock(64))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(64+64, in_c,4,2,1), nn.Sigmoid())

        self.cls = nn.Sequential(nn.Dropout(0.3), nn.Linear(lat_c*16*16, num_cls))

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

# ----------------------------
# 4. Loss helpers
# ----------------------------
def weighted_mse(x,xr,w=3.0):
    m=torch.ones_like(x[:, :1]); m[:,:,32:96,32:96]=w
    return ((x-xr).pow(2)*m).mean()

def vae_loss(x,xr,mu,lv,beta):
    recon=weighted_mse(x,xr,1.0)
    kl=-0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
    return recon+beta*kl, recon, kl

# perceptual
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg=models.vgg16(pretrained=False).features[:16].to(device).eval()
for p in vgg.parameters(): p.requires_grad=False
m_mean=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
m_std =torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1)

# ----------------------------
# 5. Train/Validate
# ----------------------------
def run():
    model=Teacher().to(device)
    opt=optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    start=1
    if args.resume and args.resume.exists():
        ck=torch.load(args.resume,map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start=ck["epoch"]+1
        for st in opt.state.values():
            for k,v in st.items():
                if isinstance(v,torch.Tensor): st[k]=v.to(device)
        print("Resumed from",args.resume)
    best=float('inf')
    for ep in range(start,args.epochs+1):
        model.train()
        kl_w=0.0002  # small beta
        for x,y in tqdm(train_loader,desc=f"Ep{ep}"):
            x,y=x.to(device),y.to(device)
            xr,z,mu,lv,logits=model(x)
            total,recon,kl=vae_loss(x,xr,mu,lv,kl_w)
            ce = F.cross_entropy(logits,y)
            perc=F.mse_loss(vgg((xr-m_mean)/m_std), vgg((x-m_mean)/m_std))
            loss= total + 5.0*perc + 0.02*ce
            opt.zero_grad(); loss.backward(); opt.step()
        # ---- validation
        model.eval(); stats={'rec':0,'perc':0,'ce':0,'n':0}
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device)
                xr,_,mu,lv,logits=model(x)
                _,recon,_=vae_loss(x,xr,mu,lv,kl_w)
                ce=F.cross_entropy(logits,y)
                perc=F.mse_loss(vgg((xr-m_mean)/m_std), vgg((x-m_mean)/m_std))
                bs=x.size(0)
                stats['rec']+=recon.item()*bs; stats['perc']+=perc.item()*bs
                stats['ce'] +=ce.item()*bs;   stats['n']+=bs
        for k in stats: 
            if k!='n': stats[k]/=stats['n']
        metric=stats['rec']+5*stats['perc']+0.02*stats['ce']
        print(f"VAL Ep{ep}: rec {stats['rec']:.8f} perc {stats['perc']:.8f} ce {stats['ce']:.8f}")
        # checkpoint
        ckpt_name = f"/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v0_distillization/checkpoint_teacher/teacher_epoch{ep}.pt"
        torch.save(model.state_dict(), ckpt_name)
        print(f"Saved {ckpt_name}")


run()
