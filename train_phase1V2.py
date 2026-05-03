"""
Phase 1 V2 — Mel-Spectrogram CNN  (overfitting + imbalance fixes)
Run from project root: python train_phase1V2.py

Changes vs V1:
  [FIX 1] Effective Number of Samples class weights  (Cui et al., CVPR 2019)
  [FIX 2] Label smoothing ε = 0.1  (Müller et al., NeurIPS 2019)
  [FIX 3] SpecAugment params calibrated to paper proportions  (Park et al., 2019)
  [FIX 4] Mixup augmentation α = 0.3  (Zhang et al., ICLR 2018)
  [FIX 5] Linear warmup (2 epochs) + CosineAnnealingLR
  [FIX 6] Early stopping on val_loss, patience = 4  (Prechelt, 1998)
  [KEPT]  Flat 6-class head, same MelCNN architecture as V1
"""

import os
import time
import random
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score

from machine_listener.src.preprocess import AudioPreprocessor, PreprocessConfig, AugmentationConfig
from machine_listener.src.dataset import scan_wav_files, SPLIT_DIR
from machine_listener.src.features.mel_spectrogram import compute_mel_spectrogram
from machine_listener.src.models.cnn_baseline import MelCNN
from machine_listener.src.split_utils import load_clean_split
import machine_listener.src.train_utils as utils

# ── Config ────────────────────────────────────────────────────────────────────

ROOT_DIR   = "Students"
MODELS_DIR = "machine_listener/outputs/saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATS_DIR_MEL = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mel"))
os.makedirs(FEATS_DIR_MEL, exist_ok=True)

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 32
EPOCHS        = 20
NUM_WORKERS   = 4
LR            = 1e-3
WEIGHT_DECAY  = 5e-4
MIXUP_ALPHA   = 0.3
LABEL_SMOOTH  = 0.1
ENS_BETA      = 0.9999
WARMUP_EPOCHS = 2
ES_PATIENCE   = 4

CLASS_NAMES = [
    "Machine1_Normal", "Machine1_Abnormal",
    "Machine2_Normal", "Machine2_Abnormal",
    "Machine3_Normal", "Machine3_Abnormal",
]

print(f"Device: {DEVICE}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def spec_augment(mel, freq_mask=27, time_mask=15, n_freq=2, n_time=2):
    """Paper-calibrated params: freq_mask=27 on 128 bins (~21%), time_mask=15."""
    mel = mel.clone()
    _, F, T = mel.shape
    for _ in range(n_freq):
        f  = random.randint(0, freq_mask)
        f0 = random.randint(0, max(F - f, 1))
        mel[:, f0:f0+f, :] = 0.0
    for _ in range(n_time):
        t  = random.randint(0, time_mask)
        t0 = random.randint(0, max(T - t, 1))
        mel[:, :, t0:t0+t] = 0.0
    return mel


def _precompute_mel_one(args):
    idx, wav_path, feats_dir, preprocessor = args
    out = pathlib.Path(feats_dir) / f"{idx:06d}.npy"
    if out.exists():
        return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        np.save(out, compute_mel_spectrogram(w))
    except Exception:
        np.save(out, np.zeros((1, 128, 84), dtype=np.float32))


def precompute_mel(paths, feats_dir, preprocessor, n_workers=4):
    import tqdm
    feats_dir = pathlib.Path(feats_dir)
    feats_dir.mkdir(parents=True, exist_ok=True)
    already = sum(1 for i in range(len(paths)) if (feats_dir / f"{i:06d}.npy").exists())
    if already == len(paths):
        print(f"All {len(paths)} mel-specs already cached  (skipping)")
        return
    print(f"Pre-computing {len(paths)} mel-specs with {n_workers} threads ...")
    from concurrent.futures import ThreadPoolExecutor
    args = [(i, p, str(feats_dir), preprocessor) for i, p in enumerate(paths)]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(tqdm.tqdm(ex.map(lambda a: _precompute_mel_one(a), args),
                       total=len(args), desc="mel"))
    print("Pre-computation done.")


def effective_num_weights(label_counts, beta=0.9999, num_classes=6):
    eff_num = (1.0 - np.power(beta, label_counts)) / (1.0 - beta)
    weights = 1.0 / eff_num
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def mixup_batch(x, y, alpha=0.3, device="cpu"):
    if alpha <= 0: return x, y, y, 1.0
    lam  = float(np.random.beta(alpha, alpha))
    lam  = max(lam, 1.0 - lam)
    B    = x.size(0)
    perm = torch.randperm(B, device=device)
    return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def make_warmup_scheduler(optimizer, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, feats_dir, labels, indices, augment=False):
        self.feats_dir = pathlib.Path(feats_dir)
        self.labels    = labels
        self.indices   = indices
        self.augment   = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri  = self.indices[idx]
        mel = torch.tensor(np.load(self.feats_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment:
            mel = spec_augment(mel)
        return mel, torch.tensor(self.labels[ri], dtype=torch.long)


def train_epoch_v2(model, loader, optimizer, criterion, device, mixup_alpha):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for mel, y in loader:
        mel, y = mel.to(device), y.to(device)
        mel_mix, y_a, y_b, lam = mixup_batch(mel, y, alpha=mixup_alpha, device=device)
        optimizer.zero_grad()
        out  = model(mel_mix)
        loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == y_a).sum().item()
        total      += y.size(0)
    return total_loss / len(loader), correct / total


# ── Step 1: scan files and load split ────────────────────────────────────────

_infer_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=16000, default_duration_sec=2.75, trim_silence=True, normalize_mode="peak",
    augmentation=AugmentationConfig(enabled=False),
))

ALL_PATHS, ALL_LABELS = scan_wav_files(ROOT_DIR)
print(f"Found {len(ALL_PATHS)} files")

_splits = load_clean_split(SPLIT_DIR)

# ── Step 2: pre-compute mel-specs ─────────────────────────────────────────────

precompute_mel(ALL_PATHS, FEATS_DIR_MEL, _infer_prep, n_workers=NUM_WORKERS)

# ── Step 3: datasets, loaders, class weights ──────────────────────────────────

label_counts  = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
class_weights = effective_num_weights(label_counts, beta=ENS_BETA).to(DEVICE)
print(f"Label counts: {label_counts}")

train_ds = PrecomputedDataset(FEATS_DIR_MEL, ALL_LABELS, _splits["train"], augment=True)
val_ds   = PrecomputedDataset(FEATS_DIR_MEL, ALL_LABELS, _splits["val"],   augment=False)
test_ds  = PrecomputedDataset(FEATS_DIR_MEL, ALL_LABELS, _splits["test"],  augment=False)

print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)

# ── Step 4: model, optimizer, criterion ──────────────────────────────────────

model     = MelCNN(num_classes=6).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

# [FIX 5] Warmup + CosineAnnealingLR
warmup_scheduler  = make_warmup_scheduler(optimizer, warmup_epochs=WARMUP_EPOCHS)
cosine_scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Step 5: training loop ─────────────────────────────────────────────────────

best_val_loss = float("inf")
best_val_acc  = 0.0
es_counter    = 0
history       = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
ckpt_path     = os.path.join(MODELS_DIR, "phase1_v2_best.pth")

print("\n── Training Phase 1 V2 ────────────────────────────────────")
print(f"   ENS β={ENS_BETA} | LabelSmooth ε={LABEL_SMOOTH} | Mixup α={MIXUP_ALPHA}")
print(f"   Warmup={WARMUP_EPOCHS}ep | ES patience={ES_PATIENCE}")
print()

for epoch in range(1, EPOCHS + 1):
    current_lr = optimizer.param_groups[0]["lr"]
    tr_loss, tr_acc       = train_epoch_v2(model, train_loader, optimizer, criterion, DEVICE, MIXUP_ALPHA)
    vl_loss, vl_acc, _, _ = utils.eval_epoch(model, val_loader, criterion, DEVICE)

    if epoch <= WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)

    improved = vl_loss < best_val_loss
    tag = ""
    if improved:
        best_val_loss = vl_loss
        best_val_acc  = vl_acc
        es_counter    = 0
        torch.save({
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "val_loss": vl_loss, "val_acc": vl_acc,
        }, ckpt_path)
        tag = "  ← saved"
    else:
        es_counter += 1
        tag = f"  (patience {es_counter}/{ES_PATIENCE})"

    print(f"Epoch {epoch:3d}/{EPOCHS}  lr={current_lr:.2e}  "
          f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
          f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{tag}")

    if es_counter >= ES_PATIENCE:
        print(f"\n[early stop] Stopping after {ES_PATIENCE} epochs without val_loss improvement.")
        break

print(f"\nBest val loss: {best_val_loss:.4f}  (val acc: {best_val_acc:.4f})")

# ── Step 6: test evaluation ───────────────────────────────────────────────────

ckpt = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_, test_acc, test_preds, test_labels = utils.eval_epoch(model, test_loader, criterion, DEVICE)
t_test = time.time() - t0
n_test = len(test_ds)
ms_per_sample = (t_test / n_test) * 1000

metrics = utils.compute_metrics(test_preds, test_labels)

print(f"\n── Test Results ─────────────────────────────────────────────")
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"Macro F1 : {metrics['macro_f1']:.4f}")
print(f"Epoch    : {ckpt['epoch']}")
print()
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

print(f"\n── Inference Timing ─────────────────────────────────────────")
print(f"Test set size         : {n_test} samples")
print(f"Total inference time  : {t_test:.2f} s")
print(f"Per-sample time       : {ms_per_sample:.3f} ms  →  {1000/ms_per_sample:.0f} samples/sec")
print(f"Estimated   100 files : {ms_per_sample *   100 / 1000:.2f} s")
print(f"Estimated 1 000 files : {ms_per_sample *  1000 / 1000:.2f} s")
print(f"Estimated 10 000 files: {ms_per_sample * 10000 / 1000:.2f} s")

utils.plot_confusion_matrix(test_preds, test_labels, CLASS_NAMES)

# ── Training curves ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="train"); ax1.plot(history["val_loss"], label="val")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
ax2.plot(history["train_acc"],  label="train"); ax2.plot(history["val_acc"],  label="val")
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
plt.suptitle("Phase 1 V2 — ENS + Label Smoothing + Mixup + Warmup")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "phase1_v2_curves.png"))
plt.show()

print(f"Checkpoint: {ckpt_path}")
