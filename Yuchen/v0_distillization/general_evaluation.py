#!/usr/bin/env python
# test_teacher.py
'''
For techer model
python general_evaluation.py \
       --model_def /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v0_distillization/teacher_model.py \
       --ckpt      /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v0_distillization/checkpoint_teacher/teacher_epoch207.pt \
       --data      /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_data_128.npz \
       --batch     256  \
       --latent     64
'''
'''
For v4_SiLU_8_latent
python general_evaluation.py \
       --model_def /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4_SiLU_8latent/for_general_test.py \
       --ckpt      /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4_SiLU_8latent/checkpoint/8latent_checkpoint_SiLU_epoch269.pt \
       --data      /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_data_128.npz \
       --batch     128  \
       --latent     8
'''
'''
For v4_SiLU_8_latent_heavier
python general_evaluation.py \
       --model_def /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4_SiLU_8latent/for_general_test.py \
       --ckpt      /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4_SiLU_8latent/heavier_recon/checkpoint/heavier_8latent_checkpoint_SiLU_epoch1.pt \
       --data      /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_data_128.npz \
       --batch     128  \
       --latent     8
'''
'''
For v4_SiLU
python general_evaluation.py \
       --model_def /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4_SiLU/for_general_test.py \
       --ckpt      /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4_SiLU/checkpoint/second_finetune_epoch1_acc0.070.pt \
       --data      /gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_data_128.npz \
       --batch     128  \
       --latent     32
'''
import argparse, importlib.machinery as ml, pathlib, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--model_def", type=pathlib.Path, required=True,
               help="path to teacher model .py")
p.add_argument("--ckpt",      type=pathlib.Path, required=True,
               help="path to checkpoint .pt (state_dict or full dict)")
p.add_argument("--data",      type=pathlib.Path, required=True,
               help="npz file with images[ N,3,128,128 ] and labels[ N ]")
p.add_argument("--batch",     type=int, default=256, help="batch size")
p.add_argument("--latent", type=int, default=64,
                    help="latent_channels in the model (teacher=64, student=32/8)")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------
# Dataset
# -----------------------------------------------------------
class PlainDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.imgs, self.lbls = d["images"], d["labels"]
        self.tfm = transforms.ToTensor()  # will re‑normalise to [0,1] float
    def __len__(self):  return len(self.imgs)
    def __getitem__(self, i):
        img = Image.fromarray((self.imgs[i].transpose(1,2,0)*255).astype(np.uint8))
        return self.tfm(img), torch.tensor(self.lbls[i], dtype=torch.long)

data_loader = DataLoader(PlainDataset(args.data), batch_size=args.batch,
                         shuffle=False, num_workers=4, pin_memory=True)

# -----------------------------------------------------------
# Dynamically import teacher model definition
# -----------------------------------------------------------
loader = ml.SourceFileLoader("teacher_mod", str(args.model_def))
teacher_mod = loader.load_module()
TeacherModel = teacher_mod.Model        # assumes class name unchanged

# instantiate with same hyper‑params used during training
model = TeacherModel(input_channels=3, latent_channels=args.latent, num_classes=len(np.unique(np.load(args.data)["labels"])))
model.to(device)

# load checkpoint
ckpt = torch.load(args.ckpt, map_location=device)
state = ckpt["model_state"] if "model_state" in ckpt else ckpt
model.load_state_dict(state, strict=True)
model.eval()

# -----------------------------------------------------------
# Evaluation
# -----------------------------------------------------------
pix_err_sum, n_pix = 0.0, 0
correct, n_sample = 0, 0

with torch.no_grad():
    for x, y in tqdm(data_loader, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        x_rec, _, _, _, logits = model(x)          # same forward signature
        pix_err_sum += F.mse_loss(x_rec, x, reduction="sum").item()
        n_pix      += x.numel()
        correct    += (logits.argmax(1) == y).sum().item()
        n_sample   += y.size(0)

recon_mse  = pix_err_sum / n_sample               # per‑image MSE (matches train code)
acc        = correct / n_sample
score      = recon_mse / acc

print(f"\nReconstruction MSE (per image) : {recon_mse:.6f}")
print(f"Probing accuracy                : {acc*100:.2f}%")
print(f"Competition score (MSE / acc)   : {score:.6f}")
