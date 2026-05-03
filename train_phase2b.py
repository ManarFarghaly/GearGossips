"""
Phase 2b — Mel-Spectrogram + Statistical Features (no MFCC)
Run from project root: python train_phase2b.py

Loads Phase 1 (V1) checkpoint. Fine-tunes MelStatCNN with a flat 6-class head,
global stat normalization, differential LRs, and ENS class weights.

  Phase 2  : Mel + MFCC  →  ~0.49 ms/sample
  Phase 2b : Mel + Stat  →  ~0.25 ms/sample  (this script measures it)

Features: rms, zcr, rolloff, bandwidth, kurtosis
  Centroid dropped — redundant with mel CNN's frequency representations.
  Kurtosis added — standard vibration fault indicator (impulsive signal bursts).

For the V2 version (hierarchical heads, focal loss, RLROP), use train_phase2bV2.py.
"""

import os
import time
import pickle
import random
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score

from machine_listener.src.preprocess import AudioPreprocessor, PreprocessConfig, AugmentationConfig
from machine_listener.src.dataset import scan_wav_files, SPLIT_DIR
from machine_listener.src.features.mel_spectrogram import compute_mel_spectrogram
from machine_listener.src.features.statistical import (
    compute_stat_features_v2,
    ALL_FEATURE_NAMES_V2,
    STAT_COL_V2,
)
from machine_listener.src.models.cnn_mel_stat import MelStatCNN
from machine_listener.src.split_utils import load_clean_split
import machine_listener.src.train_utils as utils

# ── Config ────────────────────────────────────────────────────────────────────

ROOT_DIR    = "Students"
MODELS_DIR  = "machine_listener/outputs/saved_models"
PHASE1_CKPT = os.path.join(MODELS_DIR, "phase1_best.pth")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATS_DIR_MEL  = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mel"))
FEATS_DIR_STAT = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "stat_v2"))
os.makedirs(FEATS_DIR_MEL,  exist_ok=True)
os.makedirs(FEATS_DIR_STAT, exist_ok=True)

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 32
TRAIN_EPOCHS  = 25
NUM_WORKERS   = 2
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.1
ENS_BETA      = 0.9999
ES_PATIENCE   = 5

STAT_FEATURES = ALL_FEATURE_NAMES_V2   # ["rms", "zcr", "rolloff", "bandwidth", "kurtosis"]

CLASS_NAMES = [
    "Machine1_Normal", "Machine1_Abnormal",
    "Machine2_Normal", "Machine2_Abnormal",
    "Machine3_Normal", "Machine3_Abnormal",
]

print(f"Device: {DEVICE}")
print(f"Stat features: {STAT_FEATURES}")

if not os.path.exists(PHASE1_CKPT):
    raise FileNotFoundError(
        f"Phase 1 checkpoint not found at {PHASE1_CKPT}. Run train_phase1.py first.")

# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Pre-computation ───────────────────────────────────────────────────────────

def _precompute_mel_one(args):
    idx, wav_path, mel_dir, preprocessor = args
    out = pathlib.Path(mel_dir) / f"{idx:06d}.npy"
    if out.exists():
        return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        np.save(out, compute_mel_spectrogram(w))
    except Exception:
        np.save(out, np.zeros((1, 128, 84), dtype=np.float32))


def _precompute_stat_one(args):
    idx, wav_path, stat_dir, preprocessor = args
    out = pathlib.Path(stat_dir) / f"{idx:06d}.npy"
    if out.exists():
        return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        np.save(out, compute_stat_features_v2(w))
    except Exception:
        np.save(out, np.zeros(5, dtype=np.float32))


def precompute_mel_stat(paths, mel_dir, stat_dir, preprocessor, n_workers=4):
    import tqdm
    mel_dir  = pathlib.Path(mel_dir);  mel_dir.mkdir(parents=True, exist_ok=True)
    stat_dir = pathlib.Path(stat_dir); stat_dir.mkdir(parents=True, exist_ok=True)

    mel_done  = sum(1 for i in range(len(paths)) if (mel_dir  / f"{i:06d}.npy").exists())
    stat_done = sum(1 for i in range(len(paths)) if (stat_dir / f"{i:06d}.npy").exists())

    if mel_done < len(paths):
        print(f"Pre-computing mel-specs for {len(paths) - mel_done} files ...")
        args = [(i, p, str(mel_dir), preprocessor) for i, p in enumerate(paths)]
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(tqdm.tqdm(ex.map(lambda a: _precompute_mel_one(a), args),
                           total=len(args), desc="mel"))

    if stat_done < len(paths):
        print(f"Pre-computing stat features for {len(paths) - stat_done} files ...")
        args = [(i, p, str(stat_dir), preprocessor) for i, p in enumerate(paths)]
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(tqdm.tqdm(ex.map(lambda a: _precompute_stat_one(a), args),
                           total=len(args), desc="stat"))

    print("Pre-computation done.")


# ── Dataset ───────────────────────────────────────────────────────────────────

class PrecomputedDatasetMelStat(Dataset):
    def __init__(self, mel_dir, stat_dir, labels, indices, augment=False):
        self.mel_dir  = pathlib.Path(mel_dir)
        self.stat_dir = pathlib.Path(stat_dir)
        self.labels   = labels
        self.indices  = indices
        self.augment  = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ri   = self.indices[idx]
        mel  = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        stat = torch.tensor(np.load(self.stat_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment:
            mel = spec_augment(mel)
        return (mel, stat), torch.tensor(self.labels[ri], dtype=torch.long)


def collate_mel_stat(batch):
    feats, labels = zip(*batch)
    return (torch.stack([f[0] for f in feats]),
            torch.stack([f[1] for f in feats])), torch.stack(labels)


def fit_scaler(stat_dir, indices):
    arr = np.stack([np.load(pathlib.Path(stat_dir) / f"{i:06d}.npy") for i in indices])
    return arr.mean(0), arr.std(0) + 1e-8


# ── Train / eval ──────────────────────────────────────────────────────────────

def train_epoch_ms(model, loader, optimizer, criterion, device, s_mean, s_std):
    model.train()
    sm = torch.tensor(s_mean, dtype=torch.float32).to(device)
    ss = torch.tensor(s_std,  dtype=torch.float32).to(device)
    total_loss, correct, total = 0.0, 0, 0
    for (mel, stat), y in loader:
        mel, stat, y = mel.to(device), stat.to(device), y.to(device)
        stat = (stat - sm) / ss
        optimizer.zero_grad()
        out  = model(mel, stat)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / len(loader), correct / total


def eval_epoch_ms(model, loader, criterion, device, s_mean, s_std):
    model.eval()
    sm = torch.tensor(s_mean, dtype=torch.float32).to(device)
    ss = torch.tensor(s_std,  dtype=torch.float32).to(device)
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for (mel, stat), y in loader:
            mel, stat, y = mel.to(device), stat.to(device), y.to(device)
            stat  = (stat - sm) / ss
            out   = model(mel, stat)
            loss  = criterion(out, y)
            preds = out.argmax(1)
            total_loss += loss.item()
            correct    += (preds == y).sum().item()
            total      += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels


# ── Step 1: scan files and load split ────────────────────────────────────────

_infer_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=16000, default_duration_sec=2.75,
    augmentation=AugmentationConfig(enabled=False),
))

ALL_PATHS, ALL_LABELS = scan_wav_files(ROOT_DIR)
print(f"Found {len(ALL_PATHS)} files")

_splits = load_clean_split(SPLIT_DIR)

# ── Step 2: pre-compute mel + stat (runs once, then cached) ──────────────────

precompute_mel_stat(ALL_PATHS, FEATS_DIR_MEL, FEATS_DIR_STAT, _infer_prep, n_workers=NUM_WORKERS)

# ── Step 3: build datasets and fit scaler ────────────────────────────────────

label_counts  = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
# ENS weights — more principled than 1/(n+1) for imbalanced datasets
eff_num       = (1.0 - np.power(ENS_BETA, label_counts)) / (1.0 - ENS_BETA)
ens_w         = 1.0 / eff_num
ens_w         = ens_w / ens_w.sum() * 6
class_weights = torch.tensor(ens_w, dtype=torch.float32).to(DEVICE)
print(f"Label counts : {label_counts}")
print(f"ENS weights  : {ens_w.round(4)}")

tr_ds = PrecomputedDatasetMelStat(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["train"], augment=True)
vl_ds = PrecomputedDatasetMelStat(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["val"],   augment=False)
te_ds = PrecomputedDatasetMelStat(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["test"],  augment=False)

print("Fitting scaler from training split .npy files ...")
sm, ss = fit_scaler(FEATS_DIR_STAT, _splits["train"])

tr_ldr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_mel_stat,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False, prefetch_factor=2)
vl_ldr = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_mel_stat,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False, prefetch_factor=2)
te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_mel_stat,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False, prefetch_factor=2)

# ── Step 4: model — load Phase 1 mel_stream weights ──────────────────────────

model = MelStatCNN(num_classes=6, stat_dim=len(STAT_FEATURES)).to(DEVICE)

p1_ckpt = torch.load(PHASE1_CKPT, map_location=DEVICE)
model.mel_stream.load_state_dict(p1_ckpt["model_state_dict"])

# mel_stream from Phase 1 → low LR to preserve learned representations
# stat_branch + head are new → larger LR to train from scratch
optimizer = torch.optim.AdamW([
    {"params": model.mel_stream.parameters(),  "lr": 1e-4},
    {"params": model.stat_branch.parameters(), "lr": 5e-4},
    {"params": model.fc1.parameters(),         "lr": 5e-4},
    {"params": model.fc2.parameters(),         "lr": 5e-4},
], weight_decay=1e-4)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
)

# ── Step 5: training loop ─────────────────────────────────────────────────────

print(f"\n══════════════════════════════════════════════════════════════")
print(f"  Phase 2b — Mel + Stat  ({TRAIN_EPOCHS} epochs, differential LRs)")
print(f"  Features: {STAT_FEATURES}")
print(f"══════════════════════════════════════════════════════════════\n")

best_val_loss = float("inf")
best_val_acc  = 0.0
es_counter    = 0
history       = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
ckpt_path     = os.path.join(MODELS_DIR, "phase2b_best.pth")

print(f"\n── Training Phase 2b — Mel + Stat (flat head, global norm) ──")
print(f"   ENS β={ENS_BETA} | LabelSmooth ε={LABEL_SMOOTH} | ES patience={ES_PATIENCE}")
print()

for epoch in range(1, TRAIN_EPOCHS + 1):
    tr_l, tr_a       = train_epoch_ms(model, tr_ldr, optimizer, criterion, DEVICE, sm, ss)
    vl_l, vl_a, _, _ = eval_epoch_ms( model, vl_ldr, criterion, DEVICE, sm, ss)
    scheduler.step(vl_l)

    history["train_loss"].append(tr_l); history["train_acc"].append(tr_a)
    history["val_loss"].append(vl_l);   history["val_acc"].append(vl_a)

    improved = vl_l < best_val_loss
    tag = ""
    if improved:
        best_val_loss = vl_l
        best_val_acc  = vl_a
        es_counter    = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch":            epoch,
            "val_loss":         vl_l,
            "val_acc":          vl_a,
            "stat_features":    STAT_FEATURES,
            "stat_dim":         len(STAT_FEATURES),
            "scaler_mean":      sm,
            "scaler_std":       ss,
        }, ckpt_path)
        tag = "  ← saved"
    else:
        es_counter += 1
        tag = f"  (patience {es_counter}/{ES_PATIENCE})"

    print(f"Epoch {epoch:3d}/{TRAIN_EPOCHS}  "
          f"train_loss={tr_l:.4f}  train_acc={tr_a:.4f}  "
          f"val_loss={vl_l:.4f}  val_acc={vl_a:.4f}{tag}")

    if es_counter >= ES_PATIENCE:
        print(f"\n[early stop] Stopping after {ES_PATIENCE} epochs without improvement.")
        break

print(f"\nBest val loss: {best_val_loss:.4f}  (val acc: {best_val_acc:.4f})")

# ── Step 6: save scaler + test evaluation ─────────────────────────────────────
# Checkpoint keys: model_state_dict · epoch · val_acc · stat_features · stat_dim
# Loaded by train_phase4b.py as PHASE2B_CKPT.

scaler_path = os.path.join(MODELS_DIR, "stat_scaler_2b.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump({"mean": sm, "std": ss, "features": STAT_FEATURES}, f)
print(f"Scaler saved → {scaler_path}")

ckpt = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_, ta, preds, labels = eval_epoch_ms(model, te_ldr, criterion, DEVICE, sm, ss)
t_test = time.time() - t0
n_test = len(te_ds)
ms_per_sample = (t_test / n_test) * 1000

print(f"\n── Test Results ──────────────────────────────────────────────")
print(f"Test Accuracy : {ta:.4f}")
print(f"Macro F1      : {f1_score(labels, preds, average='macro'):.4f}")
print(f"Stat features : {STAT_FEATURES}")
print()
print(classification_report(labels, preds, target_names=CLASS_NAMES))

print(f"\n── Inference Timing ─────────────────────────────────────────")
print(f"Test set size         : {n_test} samples")
print(f"Total inference time  : {t_test:.2f} s")
print(f"Per-sample time       : {ms_per_sample:.3f} ms  →  {1000/ms_per_sample:.0f} samples/sec")
print(f"Estimated   100 files : {ms_per_sample *   100 / 1000:.2f} s")
print(f"Estimated 1 000 files : {ms_per_sample *  1000 / 1000:.2f} s")
print(f"Estimated 10 000 files: {ms_per_sample * 10000 / 1000:.2f} s")

print(f"\n── Phase Comparison ─────────────────────────────────────────")
print(f"Phase 1  (Mel only)   : ~0.24 ms/sample  99.80%  baseline")
print(f"Phase 2  (Mel+MFCC)  : ~0.49 ms/sample  99.91%  +0.11% acc, 2× slower")
print(f"Phase 2b (Mel+Stat)  : {ms_per_sample:.3f} ms/sample  {ta:.2%}  this run")

utils.plot_confusion_matrix(preds, labels, CLASS_NAMES)

# ── Training curves ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history["train_acc"], label="train")
ax.plot(history["val_acc"],   label="val")
ax.set_title("Phase 2b — Mel + Statistical CNN")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "phase2b_curves.png"))
plt.show()

print(f"\nSaved: {ckpt_path}")
print(f"Saved: {scaler_path}")
