"""
train_phase2b.py — Phase 2b: Mel-Spectrogram + Statistical Features (no MFCC)
Run from project root:  python train_phase2b.py

Purpose:
  Test whether replacing the MFCC stream (Phase 2) with cheap global statistical
  features gives comparable accuracy at ~2× lower inference cost.

  Phase 2  : Mel + MFCC   →  0.485 ms/sample   (99.91% test acc)
  Phase 2b : Mel + Stat   →  ~0.25 ms/sample   (TBD — this script measures it)

Features: rms, zcr, rolloff, bandwidth, kurtosis  (centroid dropped — redundant
with mel CNN; kurtosis added — standard vibration fault indicator, ISO 13373).

No ablation here. Phase 3 already validated which features matter.
Single 25-epoch fine-tune with differential LRs from a Phase 1 checkpoint.
"""

import os, time, pickle, json, random
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from machine_listener.src.preprocess import AudioPreprocessor, PreprocessConfig, AugmentationConfig
from machine_listener.src.dataset import MachineDataset
from machine_listener.src.features.mel_spectrogram import compute_mel_spectrogram
from machine_listener.src.features.statistical import (
    compute_stat_features_v2,
    ALL_FEATURE_NAMES_V2,
    STAT_COL_V2,
)
from machine_listener.src.models.cnn_mel_stat import MelStatCNN
import machine_listener.src.train_utils as utils

# ─────────────────────────── CONFIG ───────────────────────────────────────────
ROOT_DIR    = "Students"
MODELS_DIR  = "machine_listener/outputs/saved_models"
PHASE1_CKPT = os.path.join(MODELS_DIR, "phase1_best.pth")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATS_DIR_MEL  = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mel"))
FEATS_DIR_STAT = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "stat_v2"))
os.makedirs(FEATS_DIR_MEL,  exist_ok=True)
os.makedirs(FEATS_DIR_STAT, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 32
TRAIN_EPOCHS = 25   # single run — differential LRs warm up stat branch naturally
NUM_WORKERS  = 4

# Fixed stat features — no ablation needed
STAT_FEATURES = ALL_FEATURE_NAMES_V2   # ["rms", "zcr", "rolloff", "bandwidth", "kurtosis"]

print(f"Device: {DEVICE}")
print(f"Stat features: {STAT_FEATURES}")

CLASS_NAMES = [
    "Machine1_Normal", "Machine1_Abnormal",
    "Machine2_Normal", "Machine2_Abnormal",
    "Machine3_Normal", "Machine3_Abnormal",
]

# ─────────────────────────── HELPERS ──────────────────────────────────────────

def spec_augment(mel, freq_mask=30, time_mask=15, n_freq=2, n_time=2):
    mel = mel.clone(); _, F, T = mel.shape
    for _ in range(n_freq):
        f  = random.randint(0, freq_mask); f0 = random.randint(0, max(F-f, 1))
        mel[:, f0:f0+f, :] = 0.0
    for _ in range(n_time):
        t  = random.randint(0, time_mask); t0 = random.randint(0, max(T-t, 1))
        mel[:, :, t0:t0+t] = 0.0
    return mel

# ─────────────────────────── PRE-COMPUTATION ──────────────────────────────────

def _precompute_mel_one(args):
    idx, wav_path, mel_dir, preprocessor = args
    out = pathlib.Path(mel_dir) / f"{idx:06d}.npy"
    if out.exists(): return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        np.save(out, compute_mel_spectrogram(w))
    except Exception:
        np.save(out, np.zeros((1, 128, 84), dtype=np.float32))

def _precompute_stat_one(args):
    idx, wav_path, stat_dir, preprocessor = args
    out = pathlib.Path(stat_dir) / f"{idx:06d}.npy"
    if out.exists(): return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        # compute_stat_features_v2 returns [rms, zcr, rolloff, bandwidth, kurtosis]
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

# ─────────────────────────── DATASET ──────────────────────────────────────────

class PrecomputedDatasetMelStat(Dataset):
    """Loads mel + stat_v2 from .npy files. Stat is the full 5-element vector — no slicing."""
    def __init__(self, mel_dir, stat_dir, labels, indices, augment=False):
        self.mel_dir  = pathlib.Path(mel_dir)
        self.stat_dir = pathlib.Path(stat_dir)
        self.labels   = labels
        self.indices  = indices
        self.augment  = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri   = self.indices[idx]
        mel  = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        stat = torch.tensor(np.load(self.stat_dir / f"{ri:06d}.npy"), dtype=torch.float32)  # (5,)
        if self.augment:
            mel = spec_augment(mel)
        return (mel, stat), torch.tensor(self.labels[ri], dtype=torch.long)

def collate_mel_stat(batch):
    feats, labels = zip(*batch)
    return (torch.stack([f[0] for f in feats]),
            torch.stack([f[1] for f in feats])), torch.stack(labels)

# ─────────────────────────── STAT SCALER ──────────────────────────────────────

def fit_scaler(stat_dir, indices):
    """Load stat .npy files directly — no wav reads, finishes in seconds."""
    arr = np.stack([np.load(pathlib.Path(stat_dir) / f"{i:06d}.npy") for i in indices])
    return arr.mean(0), arr.std(0) + 1e-8

# ─────────────────────────── TRAIN / EVAL ─────────────────────────────────────

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
        loss.backward(); optimizer.step()
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
            stat = (stat - sm) / ss
            out  = model(mel, stat)
            loss = criterion(out, y)
            preds = out.argmax(1)
            total_loss += loss.item()
            correct    += (preds == y).sum().item()
            total      += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels

# ─────────────────────────── SETUP ────────────────────────────────────────────

if not os.path.exists(PHASE1_CKPT):
    raise FileNotFoundError(
        f"Phase 1 checkpoint not found at {PHASE1_CKPT}. Run train_phase1.py first."
    )

_infer_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=16000, default_duration_sec=2.75,
    augmentation=AugmentationConfig(enabled=False),
))

# Scan all wav files and create/load the same stratified split as Phase 1.
# feature_fn arg is required by MachineDataset but never called in scan-only mode.
_scan_ds   = MachineDataset(ROOT_DIR, _infer_prep, lambda w: w, "train")
ALL_PATHS  = _scan_ds.paths
ALL_LABELS = _scan_ds.labels

# Pre-compute mel + stat_v2 once (mel reused from Phase 1 cache if already there)
precompute_mel_stat(ALL_PATHS, FEATS_DIR_MEL, FEATS_DIR_STAT, _infer_prep, n_workers=NUM_WORKERS)

# Load split indices (created by Phase 1 or by _scan_ds above)
_split_file = pathlib.Path(MODELS_DIR) / "split_indices.json"
if not _split_file.exists():
    _alt = pathlib.Path(ROOT_DIR).parent / "split_indices.json"
    if _alt.exists():
        import shutil; shutil.copy(_alt, _split_file)
_splits = json.load(open(_split_file))

label_counts  = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
class_weights = torch.tensor(1.0 / (label_counts + 1), dtype=torch.float32).to(DEVICE)
print(f"Label counts: {label_counts}")

# ─────────────────────────── TRAINING ─────────────────────────────────────────
print(f"\n══════════════════════════════════════════════════════════════")
print(f"  Phase 2b — Mel + Stat  ({TRAIN_EPOCHS} epochs, differential LRs)")
print(f"  Features: {STAT_FEATURES}")
print(f"══════════════════════════════════════════════════════════════\n")

tr_ds = PrecomputedDatasetMelStat(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["train"], augment=True)
vl_ds = PrecomputedDatasetMelStat(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["val"],   augment=False)
te_ds = PrecomputedDatasetMelStat(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["test"],  augment=False)

print("Fitting scaler from training split .npy files ...")
sm, ss = fit_scaler(FEATS_DIR_STAT, _splits["train"])

tr_ldr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_mel_stat,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
vl_ldr = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_mel_stat,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_mel_stat,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)

model = MelStatCNN(num_classes=6, stat_dim=len(STAT_FEATURES)).to(DEVICE)

# Load Phase 1 mel_stream weights
p1_ckpt = torch.load(PHASE1_CKPT, map_location=DEVICE)
model.mel_stream.load_state_dict(p1_ckpt["model_state_dict"])

# mel_stream came from Phase 1 → small LR to preserve learned representations
# stat_branch + head are new → larger LR to train from scratch
optimizer = torch.optim.AdamW([
    {"params": model.mel_stream.parameters(),  "lr": 1e-4},
    {"params": model.stat_branch.parameters(), "lr": 5e-4},
    {"params": model.fc1.parameters(),         "lr": 5e-4},
    {"params": model.fc2.parameters(),         "lr": 5e-4},
], weight_decay=1e-4)

criterion = nn.CrossEntropyLoss(weight=class_weights)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)

best_val  = 0.0
history   = {"train_acc": [], "val_acc": []}
ckpt_path = os.path.join(MODELS_DIR, "phase2b_best.pth")

for epoch in range(1, TRAIN_EPOCHS + 1):
    tr_l, tr_a       = train_epoch_ms(model, tr_ldr, optimizer, criterion, DEVICE, sm, ss)
    vl_l, vl_a, _, _ = eval_epoch_ms( model, vl_ldr, criterion, DEVICE, sm, ss)
    scheduler.step()

    history["train_acc"].append(tr_a)
    history["val_acc"].append(vl_a)

    saved = ""
    if vl_a > best_val:
        best_val = vl_a
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_acc": vl_a,
            "stat_features": STAT_FEATURES,
            "stat_dim": len(STAT_FEATURES),
        }, ckpt_path)
        saved = "  ← saved"

    print(f"Epoch {epoch:3d}/{TRAIN_EPOCHS}  train={tr_a:.4f}  val={vl_a:.4f}{saved}")

print(f"\nBest val accuracy: {best_val:.4f}")

# ─────────────────────────── TEST EVALUATION ──────────────────────────────────
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  CHECKPOINTS SAVED TO:                                                       │
# │    machine_listener/outputs/saved_models/phase2b_best.pth                    │
# │      Keys: model_state_dict · epoch · val_acc · stat_features · stat_dim     │
# │    machine_listener/outputs/saved_models/stat_scaler_2b.pkl                  │
# │      Keys: mean · std · features  (needed at inference time)                 │
# └──────────────────────────────────────────────────────────────────────────────┘

# Save scaler so inference scripts don't need to refit it
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
print(f"Phase 3  (All three) : run train_phase3.py for the full ensemble")

utils.plot_confusion_matrix(preds, labels, CLASS_NAMES)

# ─────────────────────────── TRAINING CURVES ──────────────────────────────────
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
