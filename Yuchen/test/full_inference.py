import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from load_model import load_model
from run_inference import run_inference_AE

"""
Local evaluation script that mimics the public leaderboard metric:
    score = reconstruction_error / probing_acc  (lower is better)

 * reconstruction_error  → mean‑squared‑error (per‑sample, over the whole set)
 * probing_acc           → top‑1 classification accuracy (0‒1)

The script still retains the original `run_inference_AE` call unless you pass
`--skip-run-inference`.
"""

# -----------------------------------------------------------------------------
# Utility: evaluation loop
# -----------------------------------------------------------------------------

def evaluate_ae(model, X_np, y_np, batch_size=128, gpu=0):
    """Return (mse, ce, acc, score).

    mse   : mean‑squared reconstruction error (smaller = better)
    ce    : cross‑entropy on the probing head (diagnostic only)
    acc   : probing accuracy in [0,1]
    score : mse / acc (competition metric)
    """
    device = torch.device(f"cuda:{gpu}" if (gpu >= 0 and torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()

    ds = TensorDataset(
        torch.tensor(X_np, dtype=torch.float32),
        torch.tensor(y_np, dtype=torch.long)
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    recon_sum, cls_sum, correct, n = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            xr, z, mean, logvar, logits = model(x)

            # reconstruction error (sum to later divide by n)
            recon_sum += F.mse_loss(xr, x, reduction="sum").item()
            # classification cross‑entropy
            cls_sum   += F.cross_entropy(logits, y, reduction="sum").item()
            # accuracy
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)

    mse  = recon_sum / n
    ce   = cls_sum   / n
    acc  = correct / n if n else 0.0
    # small epsilon to avoid div‑by‑zero when acc == 0
    score = mse / max(acc, 1e-8)
    return mse, ce, acc, score


# -----------------------------------------------------------------------------
# CLI helper
# -----------------------------------------------------------------------------

def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-npz", default="/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_data_128.npy",
                   help="Path to .npy with images (N,3,128,128)")
    p.add_argument("--label-npz", default="/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_labels_128.npy",
                   help="Path to .npy with labels (N,)")
    p.add_argument("--model-py", default="/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v5/model.py",
                   help="Path to model definition .py")
    p.add_argument("--ckpt", default="/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v5/checkpoint/checkpoint_epoch100.pt",
                   help="Path to checkpoint .pt")
    p.add_argument("--gpu", type=int, default=0, help="GPU index to use; -1 for CPU")
    p.add_argument("--batch", type=int, default=128, help="Batch size for evaluation")
    p.add_argument("--skip-run-inference", action="store_true",
                   help="Skip the original run_inference_AE call (local eval only)")
    return p


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    # 1) load dataset (assumed .npy for convenience)
    X = np.load(args.data_npz)
    y = np.load(args.label_npz)

    # 2) load autoencoder w/ probing head
    ae = load_model(args.model_py, args.ckpt)

    # 3) compute local metric
    mse, ce, acc, score = evaluate_ae(ae, X, y, batch_size=args.batch, gpu=args.gpu)

    print("\n=== Local evaluation ===")
    print(f"Reconstruction MSE  : {mse:.6f}")
    print(f"Probing CrossEntropy: {ce:.6f}")
    print(f"Probing Accuracy    : {acc*100:.2f}%")
    print(f"Competition SCORE   : {score:.6f}  (lower is better)\n")

    # 4) still call the provided benchmarking script
    run_inference_AE(
            X, y,
            num_classes=170,
            model_e=ae,               # encoder
            model_d=ae,               # decoder
            gpu_index=args.gpu if args.gpu >= 0 else 0,
            batch_size=args.batch,
            timeout=50,
            bottleNeckDim=8192,
        )
