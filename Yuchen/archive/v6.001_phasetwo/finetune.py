import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from phase2_new_attenonly import ConvVAE, train_loader, val_loader, num_classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. Define a LoRA-style Linear layer
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=16):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.r = r
        self.scaling = alpha / r

        # frozen “base” weight
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        self.weight.requires_grad = False

        # low-rank adapters
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_features, r) * 0.01)

        # init base to the checkpoint weight later

    def forward(self, x):
        delta = (self.lora_B @ self.lora_A) * self.scaling  # [out, in]
        return F.linear(x, self.weight + delta, self.bias)


# 2. Load your existing checkpoint
checkpoint_path = "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v6.001_phasetwo/atten_only_checkpoint_epoch98.pt"
model = ConvVAE(input_channels=3, latent_channels=8, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(checkpoint_path))

# 3. Replace the classifier with a LoRA-augmented head
#    Original: nn.Sequential(Dropout, Linear(in=in_dim, out=num_classes))
in_dim = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.6),
    LoRALinear(in_dim, num_classes, r=8, alpha=32)
).to(device)

# 4. Initialize LoRALinear.weight from the old classifier
with torch.no_grad():
    orig_W = model.classifier[1].weight.data.clone()
    orig_b = model.classifier[1].bias.data.clone()
    model.classifier[1].weight.copy_(orig_W)
    model.classifier[1].bias.copy_(orig_b)

# 5. Freeze all parameters except the LoRA adapters
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# 6. Set up optimizer & scheduler
opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=5e-4, weight_decay=1e-5)
sched = CosineAnnealingLR(opt, T_max=20)

# 7. Fine-tuning loop (classification only)
best_ratio = float("inf")
for epoch in range(1, 101):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        # forward through frozen encoder
        with torch.no_grad():
            z, *_ = model.encode(x)
        logits = model.classifier(z.view(z.size(0), -1))

        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()

    # validation: compute recon & acc to get your score
    model.eval()
    total_recon, total_acc, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            x_recon, z, *_ = model(x)
            # reconstruction in original pixel space
            total_recon += F.mse_loss(x_recon, x, reduction="sum").item()
            preds = model.classifier(z.view(z.size(0), -1)).argmax(dim=1)
            total_acc += (preds == y).sum().item()
            n += x.size(0)

    avg_recon = total_recon / n
    acc       = total_acc / n
    score     = avg_recon / acc
    print(f"Epoch {epoch:02d}: recon={avg_recon:.6f}, acc={acc:.4f}, score={score:.6f}")

    if score < best_ratio:
        best_ratio = score
        torch.save(
            model.state_dict(),
            f"/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v6.001_phasetwo/checkpoint/lora_best_epoch{epoch}.pt"
        )
        print("  → New best score, checkpointed.")

print(f"Finished. Best ratio = {best_ratio:.6f}")
