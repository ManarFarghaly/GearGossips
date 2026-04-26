"""
train_phase4b.py — Phase 4b: Mel + MFCC + [rms, zcr, rolloff, bandwidth, kurtosis]
Run from project root: python train_phase4b.py

Same idea as Phase 3, two changes:
  - Centroid removed (Phase 3 ablation showed it hurt when added alone)
  - Kurtosis added in its place — impulsive faults show up clearly as high kurtosis

No ablation here. Phase 3 already mapped out which features help.
We go straight to a single training run on the fixed 5-feature set.

Weight transfer from phase3_best.pth:
  _phase2 (mel + mfcc CNN)  ← fully transferred
  fc1, fc2 (fusion head)    ← transferred (same shapes: 416→256, 256→6)
  stat_branch               ← re-initialized (different feature semantics)
"""

import os, time, pickle, json, random
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

from machine_listener.src.preprocess import AudioPreprocessor, PreprocessConfig, AugmentationConfig
from machine_listener.src.dataset import MachineDataset
from machine_listener.src.features.mel_spectrogram import compute_mel_spectrogram
from machine_listener.src.features.MFCC import compute_mfcc
from machine_listener.src.features.statistical import (
    compute_stat_features_v2,
    ALL_FEATURE_NAMES_V2,
    STAT_COL_V2,
)
from machine_listener.src.models.cnn_statistical import MelMFCCStatCNN
import machine_listener.src.train_utils as utils

# ---------- config ---------------------------------------------------------------

ROOT_DIR    = "Students"
MODELS_DIR  = "machine_listener/outputs/saved_models"
PHASE3_CKPT = os.path.join(MODELS_DIR, "phase3_best.pth")
os.makedirs(MODELS_DIR, exist_ok=True)

# feats_mel and feats_mfcc are reused from Phase 3 if they already exist
FEATS_MEL     = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mel"))
FEATS_MFCC    = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mfcc"))
FEATS_STAT_V2 = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "stat_v2"))

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 32
TRAIN_EPOCHS = 25   # single run, differential LRs handle the new branch warmup
NUM_WORKERS  = 2   # 2 is enough for .npy loads; more workers cause lock contention on 4-CPU machines

# Fixed feature set — no ablation, we already know what works from Phase 3
STAT_FEATURES = ["rms", "zcr", "rolloff", "bandwidth", "kurtosis"]

CLASS_NAMES = [
    "Machine1_Normal", "Machine1_Abnormal",
    "Machine2_Normal", "Machine2_Abnormal",
    "Machine3_Normal", "Machine3_Abnormal",
]

print(f"Device: {DEVICE}")
if not os.path.exists(PHASE3_CKPT):
    raise FileNotFoundError(f"Phase 3 checkpoint not found at {PHASE3_CKPT}. Run train_phase3.py first.")

# ---------- spec augment ---------------------------------------------------------

def spec_augment(mel, freq_mask=30, time_mask=15, n_freq=2, n_time=2):
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

# ---------- precompute -----------------------------------------------------------
# stat_v2 has 5 features: [rms, zcr, rolloff, bandwidth, kurtosis] — no centroid
# feats_mel and feats_mfcc skip per-file if already cached from Phase 3.

def _compute_one(args):
    idx, wav_path, mel_dir, mfcc_dir, stat_dir, prep = args
    mel_out  = pathlib.Path(mel_dir)  / f"{idx:06d}.npy"
    mfcc_out = pathlib.Path(mfcc_dir) / f"{idx:06d}.npy"
    stat_out = pathlib.Path(stat_dir) / f"{idx:06d}.npy"
    if mel_out.exists() and mfcc_out.exists() and stat_out.exists():
        return
    try:
        w = prep.preprocess(str(wav_path), mode="inference")
        if not mel_out.exists():  np.save(mel_out,  compute_mel_spectrogram(w))
        if not mfcc_out.exists(): np.save(mfcc_out, compute_mfcc(w))
        if not stat_out.exists(): np.save(stat_out, compute_stat_features_v2(w))
    except Exception:
        if not mel_out.exists():  np.save(mel_out,  np.zeros((1, 128, 84), dtype=np.float32))
        if not mfcc_out.exists(): np.save(mfcc_out, np.zeros((3, 40, 84),  dtype=np.float32))
        if not stat_out.exists(): np.save(stat_out, np.zeros(6,             dtype=np.float32))

def precompute_features(paths, mel_dir, mfcc_dir, stat_dir, prep, n_workers=4):
    import tqdm
    for d in [mel_dir, mfcc_dir, stat_dir]:
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)
    done = sum(1 for i in range(len(paths))
               if all((pathlib.Path(d) / f"{i:06d}.npy").exists()
                      for d in [mel_dir, mfcc_dir, stat_dir]))
    if done == len(paths):
        print(f"All {len(paths)} files already cached — skipping precompute")
        return
    print(f"Precomputing features for {len(paths)} files ({n_workers} threads)...")
    print("mel+mfcc skip if they exist from Phase 3; only stat_v2 needs new computation")
    args = [(i, p, mel_dir, mfcc_dir, stat_dir, prep) for i, p in enumerate(paths)]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(tqdm.tqdm(ex.map(lambda a: _compute_one(a), args),
                       total=len(args), desc="features"))
    print("Done.")

# ---------- dataset --------------------------------------------------------------

class CachedDataset(Dataset):
    """Loads from .npy cache — no wav reads during training."""

    def __init__(self, mel_dir, mfcc_dir, stat_dir, labels, indices, stat_cols, augment=False):
        self.mel_dir   = pathlib.Path(mel_dir)
        self.mfcc_dir  = pathlib.Path(mfcc_dir)
        self.stat_dir  = pathlib.Path(stat_dir)
        self.labels    = labels
        self.indices   = indices
        self.stat_cols = stat_cols   # which columns to slice from 6-element stat_v2
        self.augment   = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        ri   = self.indices[i]
        mel  = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        mfcc = torch.tensor(np.load(self.mfcc_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment:
            mel = spec_augment(mel)
        stat = np.load(self.stat_dir / f"{ri:06d}.npy")[self.stat_cols].astype(np.float32)
        return (mel, mfcc, torch.tensor(stat)), torch.tensor(self.labels[ri], dtype=torch.long)

def collate3(batch):
    feats, labels = zip(*batch)
    return (torch.stack([f[0] for f in feats]),
            torch.stack([f[1] for f in feats]),
            torch.stack([f[2] for f in feats])), torch.stack(labels)

def fit_scaler(stat_dir, indices, stat_cols):
    """Fit scaler directly from .npy files — fast, no dataset iteration."""
    arr = np.stack([np.load(pathlib.Path(stat_dir) / f"{i:06d}.npy")[stat_cols]
                    for i in indices])
    return arr.mean(0), arr.std(0) + 1e-8

# ---------- training helpers -----------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, s_mean, s_std):
    model.train()
    sm = torch.tensor(s_mean, dtype=torch.float32, device=device)
    ss = torch.tensor(s_std,  dtype=torch.float32, device=device)
    total_loss, correct, total = 0.0, 0, 0
    for (mel, mfcc, stat), y in loader:
        mel, mfcc, stat, y = mel.to(device), mfcc.to(device), stat.to(device), y.to(device)
        stat = (stat - sm) / ss
        optimizer.zero_grad()
        out  = model(mel, mfcc, stat)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / len(loader), correct / total

def eval_epoch(model, loader, criterion, device, s_mean, s_std):
    model.eval()
    sm = torch.tensor(s_mean, dtype=torch.float32, device=device)
    ss = torch.tensor(s_std,  dtype=torch.float32, device=device)
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for (mel, mfcc, stat), y in loader:
            mel, mfcc, stat, y = mel.to(device), mfcc.to(device), stat.to(device), y.to(device)
            stat  = (stat - sm) / ss
            out   = model(mel, mfcc, stat)
            loss  = criterion(out, y)
            preds = out.argmax(1)
            total_loss += loss.item()
            correct    += (preds == y).sum().item()
            total      += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels

def load_phase3_weights(model, ckpt_path, device):
    """Transfer everything from Phase 3 except stat_branch.

    stat_branch weights aren't transferred because Phase 3 learned a different
    feature set — better to start fresh for the new features.
    Everything else (CNN + fusion head) transfers fine."""
    sd = torch.load(ckpt_path, map_location=device)["model_state_dict"]
    # Load all keys except stat_branch — strict=False handles missing keys cleanly
    filtered = {k: v for k, v in sd.items() if not k.startswith("stat_branch.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    # missing should only be stat_branch keys — anything else is worth flagging
    stat_keys   = [k for k in missing if k.startswith("stat_branch.")]
    other_missing = [k for k in missing if not k.startswith("stat_branch.")]
    print(f"Phase 3 weights loaded — {len(filtered)} keys transferred")
    print(f"  stat_branch re-initialized ({len(stat_keys)} keys, expected)")
    if other_missing:
        print(f"  WARNING: unexpected missing keys: {other_missing}")

# ---------- scan + precompute ----------------------------------------------------

_infer_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=16000, default_duration_sec=2.75,
    augmentation=AugmentationConfig(enabled=False)))

# Scan paths + labels using MachineDataset (feature_fn is never called here — we only need .paths/.labels)
_scan_ds   = MachineDataset(ROOT_DIR, _infer_prep, lambda w: w, split="train")
ALL_PATHS  = _scan_ds.paths
ALL_LABELS = _scan_ds.labels
print(f"Found {len(ALL_PATHS)} files")

precompute_features(ALL_PATHS, FEATS_MEL, FEATS_MFCC, FEATS_STAT_V2, _infer_prep, NUM_WORKERS)

# Reuse the split from Phase 3 — same random seed so test set is identical
split_file = pathlib.Path(MODELS_DIR) / "split_indices.json"
if not split_file.exists():
    raise FileNotFoundError(
        f"split_indices.json not found at {split_file}. Run train_phase3.py first "
        "to generate the split, then re-run this script.")
_splits = json.load(open(split_file))

stat_cols = [STAT_COL_V2[f] for f in STAT_FEATURES]   # [0, 1, 3, 4, 5]

# ---------- build datasets -------------------------------------------------------

train_ds = CachedDataset(FEATS_MEL, FEATS_MFCC, FEATS_STAT_V2,
                         ALL_LABELS, _splits["train"], stat_cols, augment=True)
val_ds   = CachedDataset(FEATS_MEL, FEATS_MFCC, FEATS_STAT_V2,
                         ALL_LABELS, _splits["val"],   stat_cols, augment=False)
test_ds  = CachedDataset(FEATS_MEL, FEATS_MFCC, FEATS_STAT_V2,
                         ALL_LABELS, _splits["test"],  stat_cols, augment=False)

print("Fitting scaler...")
s_mean, s_std = fit_scaler(FEATS_STAT_V2, _splits["train"], stat_cols)

tr_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate3,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False, prefetch_factor=2)
vl_ldr = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate3,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False, prefetch_factor=2)
te_ldr = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate3,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False, prefetch_factor=2)

# ---------- model + optimizer ----------------------------------------------------

model = MelMFCCStatCNN(num_classes=6, stat_dim=len(STAT_FEATURES)).to(DEVICE)
load_phase3_weights(model, PHASE3_CKPT, DEVICE)

# CNN and fusion head came from Phase 3 (well-trained) — keep their LR very low.
# stat_branch is starting from scratch — give it room to learn.
optimizer = torch.optim.AdamW([
    {"params": model._phase2.mel_stream.parameters(),   "lr": 5e-5},
    {"params": model._phase2.mfcc_stream.parameters(),  "lr": 5e-5},
    {"params": model._phase2.fc1.parameters(),          "lr": 5e-5},
    {"params": model._phase2.fc2.parameters(),          "lr": 5e-5},
    {"params": model.stat_branch.parameters(),          "lr": 5e-4},
    {"params": model.fc1.parameters(),                  "lr": 2e-4},
    {"params": model.fc2.parameters(),                  "lr": 2e-4},
], weight_decay=1e-4)

label_counts = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
criterion    = nn.CrossEntropyLoss(
    weight=torch.tensor(1.0 / (label_counts + 1), dtype=torch.float32).to(DEVICE))
scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)

# ---------- training loop --------------------------------------------------------

print(f"\nTraining {TRAIN_EPOCHS} epochs on: {STAT_FEATURES}")
print(f"Weights from: {PHASE3_CKPT}\n")

best_val  = 0.0
ckpt_path = os.path.join(MODELS_DIR, "phase4b_best.pth")

for epoch in range(1, TRAIN_EPOCHS + 1):
    tr_loss, tr_acc = train_epoch(model, tr_ldr, optimizer, criterion, DEVICE, s_mean, s_std)
    vl_loss, vl_acc, _, _ = eval_epoch(model, vl_ldr, criterion, DEVICE, s_mean, s_std)
    scheduler.step()
    saved = ""
    if vl_acc > best_val:
        best_val = vl_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch":            epoch,
            "val_acc":          vl_acc,
            "stat_features":    STAT_FEATURES,
            "stat_dim":         len(STAT_FEATURES),
        }, ckpt_path)
        saved = "  ← saved"
    print(f"Epoch {epoch:3d}/{TRAIN_EPOCHS}  train={tr_acc:.4f}  val={vl_acc:.4f}{saved}")

print(f"\nBest val accuracy: {best_val:.4f}")

# Save scaler — needed at inference time alongside the .pth
pickle.dump({"mean": s_mean, "std": s_std, "features": STAT_FEATURES},
            open(os.path.join(MODELS_DIR, "stat_scaler_4b.pkl"), "wb"))
print(f"Saved: {ckpt_path}")
print(f"Saved: {os.path.join(MODELS_DIR, 'stat_scaler_4b.pkl')}")

# ---------- test evaluation ------------------------------------------------------

ckpt = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_, test_acc, preds, labels = eval_epoch(model, te_ldr, criterion, DEVICE, s_mean, s_std)
t_test = time.time() - t0
n_test = len(test_ds)
ms     = (t_test / n_test) * 1000

print(f"\nTest accuracy : {test_acc:.4f}")
print(f"Features      : {STAT_FEATURES}")
print("\n", classification_report(labels, preds, target_names=CLASS_NAMES))

print(f"Inference: {ms:.3f} ms/sample ({1000/ms:.0f} samples/sec)")
print(f"  100 files → {ms * 100 / 1000:.2f}s   |   1k files → {ms * 1000 / 1000:.2f}s   |   10k files → {ms * 10000 / 1000:.2f}s")

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.ylabel("True"); plt.xlabel("Predicted")
plt.title("Phase 4b — rms + zcr + rolloff + bandwidth + kurtosis")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "phase4b_confusion.png"), dpi=150)
plt.show()
print("Confusion matrix saved to phase4b_confusion.png")
