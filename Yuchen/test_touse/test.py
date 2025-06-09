import numpy as np
import torch

from load_model import load_model
from run_inference import run_inference_AE

# ──‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
# 1.  Paths & constants  (EDIT these four)
# ─────────────────────────────────────────────
MODEL_CODE =  "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4.2central_mse/model.py"
WEIGHTS     =  "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/v4.2central_mse/checkpoint/checkpoint_beta0.2_epoch122.pt"
X_TEST_NPY  =  "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_data_128.npy"   # shape (N,3,128,128)
Y_TEST_NPY  =  "/gpfsnyu/scratch/ys6132/25spring_machine_learning/Yuchen/dataset/train_labels_128.npy"   # shape (N,)
GPU_ID      =   0               # use -1 for CPU
NUM_CLASSES = 10                # change if needed
# ─────────────────────────────────────────────

# 2.  Load the same model twice (encoder & decoder share weights)
model_e = load_model(MODEL_CODE, WEIGHTS)  # encoder
model_d = load_model(MODEL_CODE, WEIGHTS)  # decoder

# 3.  Load data
X_test = np.load(X_TEST_NPY)
y_test = np.load(Y_TEST_NPY)

# 4.  Run inference
run_inference_AE(
        test_data_numpy = X_test,
        test_label_numpy = y_test,
        num_classes = NUM_CLASSES,
        model_e = model_e,
        model_d = model_d,
        gpu_index = GPU_ID,          # set to -1 if no CUDA
        batch_size = 64,
        timeout = 50,
        bottleNeckDim = 8192
    )
