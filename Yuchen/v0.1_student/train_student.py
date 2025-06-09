'''
python train_student.py \
        --teacher /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v0_distillization/checkpoint_teacher/teacher_epoch207.pt \
        --student_ckpt /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v6_morelayerv4/58_checkpoint_fixed.pt \
        --save_dir /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v0.1_student/checkpoint \
        --epochs 30
'''
import os, argparse, math, pathlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
# ------------- CLI -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--teacher", type=pathlib.Path, required=True)
parser.add_argument("--student_ckpt", type=pathlib.Path, required=True)
parser.add_argument("--save_dir", type=pathlib.Path, default="/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v0.1_student/checkpoint")
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--t", type=float, default=4.0, help="soft‑CE temperature")
args = parser.parse_args()
args.save_dir.mkdir(parents=True, exist_ok=True)

# -------- dataset (same transforms as before) -------------
class NPZDataset(Dataset):
    def __init__(self, npz, tfm=None):
        d = np.load(npz); self.x, self.y = d["images"], d["labels"]; self.tfm = tfm
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        img = Image.fromarray((self.x[i].transpose(1,2,0)*255).astype(np.uint8))
        img = self.tfm(img) if self.tfm else transforms.ToTensor()(img)
        return img, torch.tensor(self.y[i], dtype=torch.long)

train_tfm = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(128, scale=(0.5,1.0), ratio=(1,1)),
    transforms.ToTensor()
])
val_tfm = transforms.ToTensor()

root = pathlib.Path("dataset")
full = NPZDataset("/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_data_128.npz")
n_cls = int(np.unique(full.y).shape[0])
tr_idx, va_idx = train_test_split(range(len(full)), test_size=0.1,
                                  stratify=full.y, random_state=42)
train_ds = Subset(full, tr_idx); train_ds.dataset.tfm = train_tfm
val_ds   = Subset(full, va_idx);  val_ds.dataset.tfm  = val_tfm
dl_tr = DataLoader(train_ds, args.batch, True,  num_workers=4, pin_memory=True)
dl_va = DataLoader(val_ds,   args.batch, False, num_workers=4, pin_memory=True)

# -------- models (reuse your existing classes) -------------
from model import ConvVAE  # small 8 192‑dim student
from teacher_model import TeacherVAE  # big 16 384‑dim teacher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher = TeacherVAE().to(device)
teacher.load_state_dict(torch.load(args.teacher, map_location=device), strict=True)
teacher.eval();                        # freeze weights
for p in teacher.parameters(): p.requires_grad = False

student = ConvVAE(latent_channels=8).to(device)
student.load_state_dict(torch.load(args.student_ckpt, map_location=device), strict=True)

# 1x1 projection so teacher latent (64ch) → 32ch
proj = nn.Sequential(
    nn.Conv2d(64, 8, 1),
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
).to(device)

optimizer = optim.AdamW(list(student.parameters())+list(proj.parameters()),
                        lr=3e-4, weight_decay=1e-5)

# -------- helper losses -------------
def weighted_mse(x, xr, w=3.0):
    mask = torch.ones_like(x[:,:1])
    mask[:,:,32:96,32:96] = w
    return ((x-xr).pow(2)*mask).mean()

vgg = models.vgg16(pretrained=False).features[:16].to(device).eval()
for p in vgg.parameters(): p.requires_grad=False
im_mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
im_std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)

def perceptual(a,b):
    return F.mse_loss(vgg((a-im_mean)/im_std), vgg((b-im_mean)/im_std))

T=args.t
def soft_ce(log_s, log_t):
    return F.kl_div(F.log_softmax(log_s/T,1), F.softmax(log_t/T,1),
                    reduction='batchmean') * (T*T)

# -------- training loop -------------
best = 1e9
for ep in range(1, args.epochs+1):
    student.train(); proj.train()
    loop=tqdm(dl_tr, desc=f"ep{ep}/{args.epochs}")
    for x,y in loop:
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()

        # forward teacher (no grad)
        with torch.no_grad():
            x_t,z_t,_,_,log_t = teacher(x)
        # forward student
        x_s,z_s,mean,logvar,log_s = student(x)

        # losses
        pix = weighted_mse(x_s, x_t)           # ≥ 0
        perc= perceptual(x_s, x_t)             # ≥ 0
        lat = F.mse_loss(z_s, proj(z_t))       # align latent maps
        ce  = soft_ce(log_s, log_t)            # soften logits

        loss = pix + 5*perc + 0.5*lat + 0.1*ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        optimizer.step()
        loop.set_postfix(pix=float(pix),perc=float(perc),
                         lat=float(lat),ce=float(ce))

    # --- quick val recon ---
    student.eval(); proj.eval()
    val_pix=val_perc=0; n=0
    with torch.no_grad():
        for x,_ in dl_va:
            x=x.to(device)
            x_t=teacher(x)[0]; x_s=student(x)[0]
            val_pix += weighted_mse(x_s,x_t).item()*x.size(0)
            val_perc+= perceptual(x_s,x_t).item()*x.size(0)
            n+=x.size(0)
    val_pix/=n; val_perc/=n
    metric = val_pix + 5*val_perc
    print(f"VAL ep{ep}: pix={val_pix:.5f} perc={val_perc:.5f} metric={metric:.5f}")

    if metric < best:
        best=metric
        torch.save(student.state_dict(), args.save_dir/"best_student_distilled.pt")
        torch.save(proj.state_dict(),     args.save_dir/"proj.pt")
