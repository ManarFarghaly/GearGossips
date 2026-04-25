"""
Mel-Spectrogram → 2D CNN Baseline
This script:
  1. Builds the dataset (70/15/15 split, saves split_indices.json)
  2. Trains MelCNN for 20 epochs
  3. Saves the best checkpoint to outputs/saved_models/phase1_best.pth
  4. Evaluates on the test set and prints accuracy + confusion matrix
"""

import os
import random
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from machine_listener.src.preprocess import AudioPreprocessor, PreprocessConfig, AugmentationConfig
from machine_listener.src.dataset import MachineDataset
from machine_listener.src.features.mel_spectrogram import compute_mel_spectrogram
from machine_listener.src.models.cnn_baseline import MelCNN
import machine_listener.src.train_utils as utils

# Configurations
ROOT_DIR   = "Students"
MODELS_DIR = "machine_listener/outputs/saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATS_DIR_MEL = os.path.join(MODELS_DIR, "..", "features", "mel")
# normalise to avoid double-slashes:
FEATS_DIR_MEL = os.path.normpath(FEATS_DIR_MEL)
os.makedirs(FEATS_DIR_MEL, exist_ok=True)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 32
EPOCHS       = 20
NUM_WORKERS  = 4
LR           = 1e-3       # AdamW initial learning rate
WEIGHT_DECAY = 1e-4       # L2 regularisation strength

CLASS_NAMES = [
    "Machine1_Normal", "Machine1_Abnormal",
    "Machine2_Normal", "Machine2_Abnormal",
    "Machine3_Normal", "Machine3_Abnormal",
]

print(f"Device: {DEVICE}")

# ─────────────────────────── HELPERS ─────────────────────────────────────────

def spec_augment(mel, freq_mask=30, time_mask=15, n_freq=2, n_time=2):
    """Mask random frequency/time bands on a mel tensor. Replaces audio-domain augmentation."""
    mel = mel.clone()
    _, F, T = mel.shape
    for _ in range(n_freq):
        f  = random.randint(0, freq_mask); f0 = random.randint(0, max(F-f, 1))
        mel[:, f0:f0+f, :] = 0.0
    for _ in range(n_time):
        t  = random.randint(0, time_mask); t0 = random.randint(0, max(T-t, 1))
        mel[:, :, t0:t0+t] = 0.0
    return mel

def _precompute_mel_one(args):
    idx, wav_path, feats_dir, preprocessor = args
    out = pathlib.Path(feats_dir) / f"{idx:06d}.npy"
    if out.exists(): return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        np.save(out, compute_mel_spectrogram(w))
    except Exception:
        np.save(out, np.zeros((1,128,84), dtype=np.float32))

def precompute_mel(paths, feats_dir, preprocessor, n_workers=4):
    import pathlib as _pl, tqdm
    feats_dir = _pl.Path(feats_dir); feats_dir.mkdir(parents=True, exist_ok=True)
    already = sum(1 for i in range(len(paths)) if (feats_dir/f"{i:06d}.npy").exists())
    if already == len(paths):
        print(f"All {len(paths)} mel-specs already cached  (skipping)"); return
    print(f"Pre-computing {len(paths)} mel-specs with {n_workers} threads ...")
    args = [(i, p, str(feats_dir), preprocessor) for i, p in enumerate(paths)]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(tqdm.tqdm(ex.map(lambda a: _precompute_mel_one(a), args), total=len(args), desc="mel"))
    print("Pre-computation done.")

class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, feats_dir, labels, indices, augment=False):
        self.feats_dir = pathlib.Path(feats_dir)
        self.labels = labels; self.indices = indices; self.augment = augment
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        ri = self.indices[idx]
        mel = torch.tensor(np.load(self.feats_dir/f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment: mel = spec_augment(mel)
        return mel, torch.tensor(self.labels[ri], dtype=torch.long)

# ─────────────────────────── DATASET ─────────────────────────────────────────

# Step A: scan files and create/load the split
_scan_preprocessor = AudioPreprocessor(PreprocessConfig(
    target_sr=16000, default_duration_sec=2.75, trim_silence=True, normalize_mode="peak",
    augmentation=AugmentationConfig(enabled=False),
))
_scan_ds   = MachineDataset(ROOT_DIR, _scan_preprocessor, compute_mel_spectrogram, "train")
ALL_PATHS  = _scan_ds.paths
ALL_LABELS = _scan_ds.labels

# Step B: pre-compute mel-specs (runs once, then cached)
precompute_mel(ALL_PATHS, FEATS_DIR_MEL, _scan_preprocessor, n_workers=NUM_WORKERS)

# Step C: load split indices
import json as _json
_split_file = pathlib.Path(MODELS_DIR) / "split_indices.json"
# If split was saved next to root_dir by MachineDataset, also check there
if not _split_file.exists():
    _alt = pathlib.Path(ROOT_DIR).parent / "split_indices.json"
    if _alt.exists(): import shutil; shutil.copy(_alt, _split_file)
_splits = _json.load(open(_split_file))

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

# class weights
# If one class has far fewer samples than others, the model ignores it.
# Weighting inversely to frequency fixes this.
label_counts  = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
class_weights = torch.tensor(1.0 / (label_counts + 1), dtype=torch.float32).to(DEVICE)
print(f"Label counts: {label_counts}")

# MODEL + optimizer
model = MelCNN(num_classes=6).to(DEVICE)

# AdamW: adapts learning rate per parameter + fixes weight-decay interaction.
# Better convergence and generalisation than plain SGD or Adam for audio tasks.
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# CrossEntropyLoss: standard for multi-class classification.
# Internally applies softmax so we do NOT add softmax to the model's forward().
criterion = nn.CrossEntropyLoss(weight=class_weights)

# CosineAnnealingLR: slowly reduces LR from 1e-3 to ~0 over EPOCHS epochs.
# Prevents oscillating around the minimum at the end of training.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

#  TRAINING LOOP
best_val_acc = 0.0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
ckpt_path = os.path.join(MODELS_DIR, "phase1_best.pth")

print("\n── Training Phase 1: Mel-Spectrogram CNN ─────────────────────")
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc         = utils.train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    vl_loss, vl_acc, _, _   = utils.eval_epoch( model, val_loader,   criterion, DEVICE)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)

    saved = ""
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        utils.save_checkpoint(model, optimizer, epoch, vl_acc, ckpt_path)
        saved = "  ← best saved"

    print(f"Epoch {epoch:3d}/{EPOCHS}  "
          f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
          f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{saved}")

print(f"\nBest validation accuracy: {best_val_acc:.4f}")

#  TEST EVALUATION
# Load the best checkpoint (not the last epoch — they can differ)
model, best_epoch, _ = utils.load_checkpoint(ckpt_path, model)

_, test_acc, test_preds, test_labels = utils.eval_epoch(model, test_loader, criterion, DEVICE)
metrics = utils.compute_metrics(test_preds, test_labels)

print(f"\n── Test Results ─────────────────────────────────────────────")
print(f"Test Accuracy : {metrics['accuracy']:.4f}")
print(f"Test Macro F1 : {metrics['macro_f1']:.4f}")
print(f"Best at epoch : {best_epoch}")
print()
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
utils.plot_confusion_matrix(test_preds, test_labels, CLASS_NAMES)

#  TRAINING CURVES
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="train")
ax1.plot(history["val_loss"],   label="val")
ax1.set_title("Loss");     ax1.set_xlabel("Epoch"); ax1.legend()
ax2.plot(history["train_acc"],  label="train")
ax2.plot(history["val_acc"],    label="val")
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
plt.suptitle("Phase 1 — Mel-Spectrogram CNN")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "phase1_curves.png"))
plt.show()
print(f"Checkpoint: {ckpt_path}")
