"""
train_phase2.py — Phase 2: Dual-Stream CNN (Mel-Spectrogram + MFCC)
Run from project root: python train_phase2.py

What's new vs Phase 1:
  - feature_fn now returns a TUPLE (mel, mfcc) instead of a single array
  - MelMFCCCNN has two streams: mel_stream (loaded from Phase 1) + mfcc_stream (new)
  - Optimizer uses different learning rates per group:
      mel_stream  → low LR (1e-4)  — already trained, protect its knowledge
      mfcc_stream → high LR (5e-4) — brand new, needs to learn fast
  - Only 15 epochs because mel_stream already has a head start from Phase 1
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
from machine_listener.src.features.mfcc import compute_mfcc
from machine_listener.src.models.cnn_MelMFCC import MelMFCCCNN
import machine_listener.src.train_utils as utils

# ─────────────────────────── CONFIG ───────────────────────────────────────────
ROOT_DIR   = "Students"
MODELS_DIR = "machine_listener/outputs/saved_models"
PHASE1_CKPT = os.path.join(MODELS_DIR, "phase1_best.pth")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATS_DIR_MEL  = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mel"))
FEATS_DIR_MFCC = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mfcc"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS     = 15   # fewer than Phase 1 because mel_stream weights are already good
NUM_WORKERS = 4

print(f"Device: {DEVICE}")

CLASS_NAMES = [
    "Machine1_Normal", "Machine1_Abnormal",
    "Machine2_Normal", "Machine2_Abnormal",
    "Machine3_Normal", "Machine3_Abnormal",
]

# ─────────────────────────── FEATURE FUNCTION ─────────────────────────────────
# Phase 2 feature_fn returns a TUPLE: (mel_array, mfcc_array)
# The dataset's __getitem__ detects the tuple and returns two tensors.
# The DataLoader then needs a custom collate function to stack them into batches.

def compute_mel_and_mfcc(waveform, sr=16000):
    """
    Returns (mel, mfcc) tuple.
    mel  shape: (1, 128, 84)   — 1 channel, 128 freq bins, 84 time frames
    mfcc shape: (3, 40,  84)   — 3 channels (mfcc, delta, delta2), 40 coefficients
    """
    return compute_mel_spectrogram(waveform, sr), compute_mfcc(waveform, sr)

def collate_dual(batch):
    """
    batch = list of  ((mel_tensor, mfcc_tensor), label_tensor)
    We stack each part separately so DataLoader gives us two batched tensors.
    """
    feats, labels = zip(*batch)
    mel_batch  = torch.stack([f[0] for f in feats])   # (B, 1, 128, 84)
    mfcc_batch = torch.stack([f[1] for f in feats])   # (B, 3, 40,  84)
    return (mel_batch, mfcc_batch), torch.stack(labels)

# ─────────────────────────── HELPERS ─────────────────────────────────────────

def spec_augment(mel, freq_mask=30, time_mask=15, n_freq=2, n_time=2):
    mel = mel.clone(); _, F, T = mel.shape
    for _ in range(n_freq):
        f=random.randint(0,freq_mask); f0=random.randint(0,max(F-f,1))
        mel[:,f0:f0+f,:]=0.0
    for _ in range(n_time):
        t=random.randint(0,time_mask); t0=random.randint(0,max(T-t,1))
        mel[:,:,t0:t0+t]=0.0
    return mel

def _precompute_dual_one(args):
    idx, wav_path, mel_dir, mfcc_dir, preprocessor = args
    mel_out  = pathlib.Path(mel_dir)  / f"{idx:06d}.npy"
    mfcc_out = pathlib.Path(mfcc_dir) / f"{idx:06d}.npy"
    if mel_out.exists() and mfcc_out.exists(): return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        np.save(mel_out,  compute_mel_spectrogram(w))
        np.save(mfcc_out, compute_mfcc(w))
    except Exception:
        np.save(mel_out,  np.zeros((1,128,84), dtype=np.float32))
        np.save(mfcc_out, np.zeros((3,40,84),  dtype=np.float32))

def precompute_dual(paths, mel_dir, mfcc_dir, preprocessor, n_workers=4):
    import pathlib as _pl, tqdm
    mel_dir  = _pl.Path(mel_dir);  mel_dir.mkdir(parents=True,  exist_ok=True)
    mfcc_dir = _pl.Path(mfcc_dir); mfcc_dir.mkdir(parents=True, exist_ok=True)
    already = sum(1 for i in range(len(paths))
                  if (mel_dir/f"{i:06d}.npy").exists() and (mfcc_dir/f"{i:06d}.npy").exists())
    if already == len(paths):
        print(f"All {len(paths)} mel+mfcc already cached  (skipping)"); return
    print(f"Pre-computing mel + mfcc for {len(paths)} files with {n_workers} threads ...")
    args = [(i,p,str(mel_dir),str(mfcc_dir),preprocessor) for i,p in enumerate(paths)]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(tqdm.tqdm(ex.map(lambda a:_precompute_dual_one(a), args), total=len(args), desc="mel+mfcc"))
    print("Pre-computation done.")

class PrecomputedDataset2(torch.utils.data.Dataset):
    def __init__(self, mel_dir, mfcc_dir, labels, indices, augment=False):
        self.mel_dir=pathlib.Path(mel_dir); self.mfcc_dir=pathlib.Path(mfcc_dir)
        self.labels=labels; self.indices=indices; self.augment=augment
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        ri=self.indices[idx]
        mel =torch.tensor(np.load(self.mel_dir /f"{ri:06d}.npy"), dtype=torch.float32)
        mfcc=torch.tensor(np.load(self.mfcc_dir/f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment: mel=spec_augment(mel)
        return (mel,mfcc), torch.tensor(self.labels[ri], dtype=torch.long)

# ─────────────────────────── DATASET ──────────────────────────────────────────
# IMPORTANT: we load from the SAME split_indices.json created in Phase 1.
# This guarantees no data leakage between phases.

# Step A: scan + create/load split
_infer_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=16000, default_duration_sec=2.75,
    augmentation=AugmentationConfig(enabled=False),
))
_scan_ds  = MachineDataset(ROOT_DIR, _infer_prep, compute_mel_spectrogram, "train")
ALL_PATHS  = _scan_ds.paths
ALL_LABELS = _scan_ds.labels

# Step B: pre-compute mel + mfcc (runs once, then cached)
precompute_dual(ALL_PATHS, FEATS_DIR_MEL, FEATS_DIR_MFCC, _infer_prep, n_workers=NUM_WORKERS)

# Step C: load split
import json as _json
_split_file = pathlib.Path(MODELS_DIR) / "split_indices.json"
if not _split_file.exists():
    _alt = pathlib.Path(ROOT_DIR).parent / "split_indices.json"
    if _alt.exists(): import shutil; shutil.copy(_alt, _split_file)
_splits = _json.load(open(_split_file))

train_ds = PrecomputedDataset2(FEATS_DIR_MEL, FEATS_DIR_MFCC, ALL_LABELS, _splits["train"], augment=True)
val_ds   = PrecomputedDataset2(FEATS_DIR_MEL, FEATS_DIR_MFCC, ALL_LABELS, _splits["val"],   augment=False)
test_ds  = PrecomputedDataset2(FEATS_DIR_MEL, FEATS_DIR_MFCC, ALL_LABELS, _splits["test"],  augment=False)
print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_dual,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_dual,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_dual,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)

# ─────────────────────────── MODEL ────────────────────────────────────────────
model = MelMFCCCNN(num_classes=6).to(DEVICE)

# Load Phase 1 weights into mel_stream ONLY
# mel_stream.state_dict() keys look like: "block1.0.weight", "fc1.weight" ...
# phase1 checkpoint state_dict keys are the same (MelCNN has same architecture)
if not os.path.exists(PHASE1_CKPT):
    raise FileNotFoundError(
        f"Phase 1 checkpoint not found at {PHASE1_CKPT}. "
        "Run train_phase1.py first."
    )
p1_ckpt = torch.load(PHASE1_CKPT, map_location=DEVICE)
model.mel_stream.load_state_dict(p1_ckpt["model_state_dict"])
print(f"Loaded Phase 1 weights into mel_stream  (best epoch: {p1_ckpt['epoch']}, val_acc: {p1_ckpt['val_acc']:.4f})")

# ─────────────────────────── OPTIMISER (PARAMETER GROUPS) ────────────────────
# WHY DIFFERENT LEARNING RATES:
#
# mel_stream: already trained for 20 epochs. Its weights encode good audio features.
#   Using a HIGH learning rate would overwrite that knowledge too aggressively.
#   We use a LOW rate (1e-4) to nudge it gently toward the new combined task.
#
# mfcc_stream: fresh random weights. It has learned NOTHING yet.
#   Using a low rate would make it learn so slowly it barely contributes.
#   We use a HIGH rate (5e-4) so it catches up quickly.
#
# fc1, fc2 (fusion head): also brand new — high rate too.
#
# The syntax is: a LIST of DICTS, each dict has "params" and "lr".
# "params" must be model.SUBMODULE.parameters(), NOT torch.optim.SUBMODULE

optimizer = torch.optim.AdamW([
    {"params": model.mel_stream.parameters(),  "lr": 1e-4},   # protect learned weights
    {"params": model.mfcc_stream.parameters(), "lr": 5e-4},   # new stream: learn fast
    {"params": model.fc1.parameters(),         "lr": 5e-4},   # new fusion head
    {"params": model.fc2.parameters(),         "lr": 5e-4},
], weight_decay=1e-4)

label_counts  = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
class_weights = torch.tensor(1.0 / (label_counts + 1), dtype=torch.float32).to(DEVICE)
criterion     = nn.CrossEntropyLoss(weight=class_weights)
scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ─────────────────────────── TRAINING / EVAL HELPERS (dual input) ─────────────
# train_utils.train_epoch expects a single tensor x.
# Phase 2 gives (mel, mfcc) tuple. We define specialised versions here.

def train_epoch_dual(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for (mel, mfcc), y in loader:
        mel, mfcc, y = mel.to(device), mfcc.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(mel, mfcc)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / len(loader), correct / total

def eval_epoch_dual(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for (mel, mfcc), y in loader:
            mel, mfcc, y = mel.to(device), mfcc.to(device), y.to(device)
            out   = model(mel, mfcc)
            loss  = criterion(out, y)
            preds = out.argmax(1)
            total_loss += loss.item()
            correct    += (preds == y).sum().item()
            total      += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels

# ─────────────────────────── TRAINING LOOP ────────────────────────────────────
best_val_acc = 0.0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
ckpt_path = os.path.join(MODELS_DIR, "phase2_best.pth")

print("\n── Training Phase 2: Mel + MFCC Dual-Stream CNN ──────────────")
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc       = train_epoch_dual(model, train_loader, optimizer, criterion, DEVICE)
    vl_loss, vl_acc, _, _ = eval_epoch_dual( model, val_loader,   criterion, DEVICE)
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
          f"train_acc={tr_acc:.4f}  val_acc={vl_acc:.4f}{saved}")

print(f"\nBest validation accuracy: {best_val_acc:.4f}")

# ─────────────────────────── TEST EVALUATION ──────────────────────────────────
ckpt = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

_, test_acc, test_preds, test_labels = eval_epoch_dual(model, test_loader, criterion, DEVICE)
metrics = utils.compute_metrics(test_preds, test_labels)

print(f"\n── Test Results ─────────────────────────────────────────────")
print(f"Test Accuracy : {metrics['accuracy']:.4f}")
print(f"Test Macro F1 : {metrics['macro_f1']:.4f}")
print()
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
utils.plot_confusion_matrix(test_preds, test_labels, CLASS_NAMES)

# ─────────────────────────── TRAINING CURVES ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="train"); ax1.plot(history["val_loss"], label="val")
ax1.set_title("Loss"); ax1.legend()
ax2.plot(history["train_acc"],  label="train"); ax2.plot(history["val_acc"],  label="val")
ax2.set_title("Accuracy"); ax2.legend()
plt.suptitle("Phase 2 — Mel + MFCC Dual-Stream CNN")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "phase2_curves.png"))
plt.show()
print(f"Checkpoint: {ckpt_path}")
