import torch

# path to your broken checkpoint
orig_ckpt = torch.load('/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v6_morelayerv4/less_augmentation_checkpoint_epoch66.pt', map_location='cpu')
print("Checkpoint keys:", orig_ckpt.keys())
# extract the real model weights
if 'model_state_dict' in orig_ckpt:
    fixed_state = orig_ckpt['model_state_dict']
else:
    raise KeyError("No 'model_state_dict' found in checkpoint")

# (optional) if you also want to preserve optimizer state:
# opt_state = orig_ckpt.get('optimizer_state_dict', None)

# overwrite (or save to a new file) just the weights
torch.save(fixed_state, 'no_augmentation_718_checkpoint.pt')

print("Wrote fixed checkpoint to checkpoint_fixed.pt")
