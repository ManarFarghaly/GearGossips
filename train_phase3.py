"""
Phase 3 — Mel + MFCC + Statistical Features (Ablation + Fine-tune)
Run from project root: python train_phase3.py

What's new vs Phase 2:
  - feature_fn returns a 3-TUPLE: (mel, mfcc, stat)
  - stat: 5 global scalars (RMS, ZCR, Centroid, Rolloff, BW)
  - MelMFCCStatCNN adds a small FC branch for stat
  - Ablation: 5 experiments with CNN frozen, adding one stat feature at a time
  - Final fine-tune: unfreeze everything with the best config
"""

import os
import pickle
import time
import random
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from machine_listener.src.preprocess import AudioPreprocessor, PreprocessConfig, AugmentationConfig
from machine_listener.src.dataset import scan_wav_files, SPLIT_DIR
from machine_listener.src.features.mel_spectrogram import compute_mel_spectrogram
from machine_listener.src.features.MFCC import compute_mfcc
from machine_listener.src.features.statistical import compute_statistical_features
from machine_listener.src.models.cnn_statistical import MelMFCCStatCNN
from machine_listener.src.split_utils import load_clean_split
import machine_listener.src.train_utils as utils

# ── Config ────────────────────────────────────────────────────────────────────

ROOT_DIR    = "Students"
MODELS_DIR  = "machine_listener/outputs/saved_models"
PHASE2_CKPT = os.path.join(MODELS_DIR, "phase2_best.pth")
os.makedirs(MODELS_DIR, exist_ok=True)

NUM_WORKERS    = 4
FEATS_DIR_MEL  = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mel"))
FEATS_DIR_MFCC = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mfcc"))
FEATS_DIR_STAT = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "stat"))

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE      = 32
ABLATION_EPOCHS = 10
FINETUNE_EPOCHS = 20

CLASS_NAMES = [
    "Machine1_Normal", "Machine1_Abnormal",
    "Machine2_Normal", "Machine2_Abnormal",
    "Machine3_Normal", "Machine3_Abnormal",
]

# Run each config from the previous result + one new feature, so we can see each contribution
ABLATION_CONFIGS = [
    {"name": "rms_only",                    "features": ["rms"]},
    {"name": "rms_zcr",                     "features": ["rms", "zcr"]},
    {"name": "rms_zcr_centroid",            "features": ["rms", "zcr", "centroid"]},
    {"name": "rms_zcr_centroid_rolloff",    "features": ["rms", "zcr", "centroid", "rolloff"]},
    {"name": "all_five",                    "features": ["rms", "zcr", "centroid", "rolloff", "bandwidth"]},
]

_ALL_STAT_NAMES = ["rms", "zcr", "centroid", "rolloff", "bandwidth"]

print(f"Device: {DEVICE}")

if not os.path.exists(PHASE2_CKPT):
    raise FileNotFoundError(
        f"Phase 2 checkpoint not found at {PHASE2_CKPT}. Run train_phase2.py first.")

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

def _precompute_triple_one(args):
    idx, wav_path, mel_dir, mfcc_dir, stat_dir, preprocessor = args
    mel_out  = pathlib.Path(mel_dir)  / f"{idx:06d}.npy"
    mfcc_out = pathlib.Path(mfcc_dir) / f"{idx:06d}.npy"
    stat_out = pathlib.Path(stat_dir) / f"{idx:06d}.npy"
    if mel_out.exists() and mfcc_out.exists() and stat_out.exists():
        return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        if not mel_out.exists():  np.save(mel_out,  compute_mel_spectrogram(w))
        if not mfcc_out.exists(): np.save(mfcc_out, compute_mfcc(w))
        if not stat_out.exists(): np.save(stat_out, compute_statistical_features(w, feature_names=_ALL_STAT_NAMES))
    except Exception:
        if not mel_out.exists():  np.save(mel_out,  np.zeros((1, 128, 84), dtype=np.float32))
        if not mfcc_out.exists(): np.save(mfcc_out, np.zeros((3, 40,  84), dtype=np.float32))
        if not stat_out.exists(): np.save(stat_out, np.zeros(5,             dtype=np.float32))


def precompute_triple(paths, mel_dir, mfcc_dir, stat_dir, preprocessor, n_workers=4):
    import tqdm
    for d in [mel_dir, mfcc_dir, stat_dir]:
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)
    already = sum(1 for i in range(len(paths))
                  if all((pathlib.Path(d) / f"{i:06d}.npy").exists()
                         for d in [mel_dir, mfcc_dir, stat_dir]))
    if already == len(paths):
        print(f"All {len(paths)} mel+mfcc+stat already cached  (skipping)")
        return
    print(f"Pre-computing mel+mfcc+stat for {len(paths)} files with {n_workers} threads ...")
    args = [(i, p, str(mel_dir), str(mfcc_dir), str(stat_dir), preprocessor) for i, p in enumerate(paths)]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(tqdm.tqdm(ex.map(lambda a: _precompute_triple_one(a), args),
                       total=len(args), desc="features"))
    print("Pre-computation done.")


# ── Dataset ───────────────────────────────────────────────────────────────────

class PrecomputedDataset3(Dataset):
    STAT_COL = {"rms": 0, "zcr": 1, "centroid": 2, "rolloff": 3, "bandwidth": 4}

    def __init__(self, mel_dir, mfcc_dir, stat_dir, labels, indices, stat_feature_names, augment=False):
        self.mel_dir   = pathlib.Path(mel_dir)
        self.mfcc_dir  = pathlib.Path(mfcc_dir)
        self.stat_dir  = pathlib.Path(stat_dir)
        self.labels    = labels
        self.indices   = indices
        self.stat_cols = [self.STAT_COL[n] for n in stat_feature_names]
        self.augment   = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ri       = self.indices[idx]
        mel      = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        mfcc     = torch.tensor(np.load(self.mfcc_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        stat_all = np.load(self.stat_dir / f"{ri:06d}.npy")
        stat     = torch.tensor(stat_all[self.stat_cols], dtype=torch.float32)
        if self.augment:
            mel = spec_augment(mel)
        return (mel, mfcc, stat), torch.tensor(self.labels[ri], dtype=torch.long)


def collate3(batch):
    feats, labels = zip(*batch)
    return (torch.stack([f[0] for f in feats]),
            torch.stack([f[1] for f in feats]),
            torch.stack([f[2] for f in feats])), torch.stack(labels)


def fit_scaler(stat_dir, indices, stat_cols):
    arr = np.stack([np.load(pathlib.Path(stat_dir) / f"{i:06d}.npy")[stat_cols] for i in indices])
    return arr.mean(0), arr.std(0) + 1e-8


# ── Train / eval ──────────────────────────────────────────────────────────────

def train_epoch3(model, loader, optimizer, criterion, device, s_mean, s_std):
    model.train()
    sm = torch.tensor(s_mean, dtype=torch.float32).to(device)
    ss = torch.tensor(s_std,  dtype=torch.float32).to(device)
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


def eval_epoch3(model, loader, criterion, device, s_mean, s_std):
    model.eval()
    sm = torch.tensor(s_mean, dtype=torch.float32).to(device)
    ss = torch.tensor(s_std,  dtype=torch.float32).to(device)
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


def load_phase2_weights(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)["model_state_dict"]
    model.mel_stream.load_state_dict(
        {k[len("mel_stream."):]: v for k, v in sd.items() if k.startswith("mel_stream.")})
    model.mfcc_stream.load_state_dict(
        {k[len("mfcc_stream."):]: v for k, v in sd.items() if k.startswith("mfcc_stream.")})
    print("Loaded Phase 2 CNN weights")


# ── Step 1: scan files, load split, pre-compute ───────────────────────────────

_infer_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=16000, default_duration_sec=2.75,
    augmentation=AugmentationConfig(enabled=False),
))

ALL_PATHS, ALL_LABELS = scan_wav_files(ROOT_DIR)
print(f"Found {len(ALL_PATHS)} files")

_splits = load_clean_split(SPLIT_DIR)

precompute_triple(ALL_PATHS, FEATS_DIR_MEL, FEATS_DIR_MFCC, FEATS_DIR_STAT,
                  _infer_prep, n_workers=NUM_WORKERS)

# ── Ablation study ────────────────────────────────────────────────────────────

print("\n══════════════════════════════════════════════════════════════")
print("  ABLATION STUDY — CNN frozen, one stat feature added at a time")
print("══════════════════════════════════════════════════════════════\n")

ablation_results = {}

for cfg_ab in ABLATION_CONFIGS:
    name     = cfg_ab["name"]
    feat     = cfg_ab["features"]
    stat_dim = len(feat)
    stat_cols_ab = [PrecomputedDataset3.STAT_COL[f] for f in feat]

    print(f"── {name}  (features: {feat}) ──────────────────────────────")

    tr_ds_ab = PrecomputedDataset3(FEATS_DIR_MEL, FEATS_DIR_MFCC, FEATS_DIR_STAT,
                                   ALL_LABELS, _splits["train"], feat, augment=True)
    vl_ds_ab = PrecomputedDataset3(FEATS_DIR_MEL, FEATS_DIR_MFCC, FEATS_DIR_STAT,
                                   ALL_LABELS, _splits["val"],   feat, augment=False)

    sm_ab, ss_ab = fit_scaler(FEATS_DIR_STAT, _splits["train"], stat_cols_ab)

    tr_ldr_ab = DataLoader(tr_ds_ab, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate3,
                           num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    vl_ldr_ab = DataLoader(vl_ds_ab, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate3,
                           num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    model_ab = MelMFCCStatCNN(num_classes=6, stat_dim=stat_dim).to(DEVICE)
    load_phase2_weights(model_ab, PHASE2_CKPT, DEVICE)
    model_ab.freeze_cnn()

    trainable = [p for p in model_ab.parameters() if p.requires_grad]
    opt_ab    = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=1e-4)

    lc      = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
    crit_ab = nn.CrossEntropyLoss(weight=torch.tensor(1.0 / (lc + 1), dtype=torch.float32).to(DEVICE))

    best_vl_ab = 0.0
    for epoch in range(1, ABLATION_EPOCHS + 1):
        tr_l, tr_a    = train_epoch3(model_ab, tr_ldr_ab, opt_ab, crit_ab, DEVICE, sm_ab, ss_ab)
        vl_l, vl_a, _, _ = eval_epoch3(model_ab, vl_ldr_ab, crit_ab, DEVICE, sm_ab, ss_ab)
        if vl_a > best_vl_ab:
            best_vl_ab = vl_a
        print(f"  Epoch {epoch:2d}/{ABLATION_EPOCHS}  train_acc={tr_a:.4f}  val_acc={vl_a:.4f}")

    ablation_results[name] = {"val_acc": best_vl_ab, "features": feat, "stat_dim": stat_dim}
    print(f"  → Best val_acc: {best_vl_ab:.4f}\n")

print("\n── Ablation Comparison ──────────────────────────────────────────")
print(f"{'Config':<35} {'N_features':>10} {'Val Acc':>8}")
print("-" * 57)
for k, v in ablation_results.items():
    print(f"{k:<35} {v['stat_dim']:>10} {v['val_acc']:>8.4f}")

best_name = max(ablation_results, key=lambda k: ablation_results[k]["val_acc"])
best_feat = ablation_results[best_name]["features"]
print(f"\nWinner: '{best_name}'  features={best_feat}")

# ── Final fine-tune ───────────────────────────────────────────────────────────

print(f"\n══════════════════════════════════════════════════════════════")
print(f"  FINAL FINE-TUNE — all layers unfrozen, {FINETUNE_EPOCHS} epochs")
print(f"══════════════════════════════════════════════════════════════\n")

stat_dim_f    = len(best_feat)
stat_cols_f   = [PrecomputedDataset3.STAT_COL[f] for f in best_feat]

tr_ds_f = PrecomputedDataset3(FEATS_DIR_MEL, FEATS_DIR_MFCC, FEATS_DIR_STAT,
                               ALL_LABELS, _splits["train"], best_feat, augment=True)
vl_ds_f = PrecomputedDataset3(FEATS_DIR_MEL, FEATS_DIR_MFCC, FEATS_DIR_STAT,
                               ALL_LABELS, _splits["val"],   best_feat, augment=False)
te_ds_f = PrecomputedDataset3(FEATS_DIR_MEL, FEATS_DIR_MFCC, FEATS_DIR_STAT,
                               ALL_LABELS, _splits["test"],  best_feat, augment=False)

sm_f, ss_f = fit_scaler(FEATS_DIR_STAT, _splits["train"], stat_cols_f)

pickle.dump({"mean": sm_f, "std": ss_f, "features": best_feat},
            open(os.path.join(MODELS_DIR, "stat_scaler.pkl"), "wb"))
print("Scaler saved to stat_scaler.pkl")

tr_ldr_f = DataLoader(tr_ds_f, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate3,
                      num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
vl_ldr_f = DataLoader(vl_ds_f, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate3,
                      num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
te_ldr_f = DataLoader(te_ds_f, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate3,
                      num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)

model_f = MelMFCCStatCNN(num_classes=6, stat_dim=stat_dim_f).to(DEVICE)
load_phase2_weights(model_f, PHASE2_CKPT, DEVICE)
model_f.unfreeze_cnn()

# CNN has been trained twice → very low LR; new stat branch → medium LR
optimizer_f = torch.optim.AdamW([
    {"params": model_f.mel_stream.parameters(),   "lr": 5e-5},
    {"params": model_f.mfcc_stream.parameters(),  "lr": 5e-5},
    {"params": model_f.stat_branch.parameters(),  "lr": 2e-4},
    {"params": model_f.fc1.parameters(),          "lr": 2e-4},
    {"params": model_f.fc2.parameters(),          "lr": 2e-4},
], weight_decay=1e-4)

label_counts = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
crit_f       = nn.CrossEntropyLoss(
    weight=torch.tensor(1.0 / (label_counts + 1), dtype=torch.float32).to(DEVICE))
sched_f      = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_f, T_max=FINETUNE_EPOCHS)

best_vl_f = 0.0
history   = {"train_acc": [], "val_acc": []}
ckpt_path = os.path.join(MODELS_DIR, "phase3_best.pth")

for epoch in range(1, FINETUNE_EPOCHS + 1):
    tr_l, tr_a       = train_epoch3(model_f, tr_ldr_f, optimizer_f, crit_f, DEVICE, sm_f, ss_f)
    vl_l, vl_a, _, _ = eval_epoch3( model_f, vl_ldr_f, crit_f, DEVICE, sm_f, ss_f)
    sched_f.step()

    history["train_acc"].append(tr_a)
    history["val_acc"].append(vl_a)

    saved = ""
    if vl_a > best_vl_f:
        best_vl_f = vl_a
        torch.save({
            "model_state_dict": model_f.state_dict(),
            "epoch":            epoch,
            "val_acc":          vl_a,
            "stat_features":    best_feat,
            "stat_dim":         stat_dim_f,
        }, ckpt_path)
        saved = "  ← best saved"

    print(f"Epoch {epoch:3d}/{FINETUNE_EPOCHS}  train_acc={tr_a:.4f}  val_acc={vl_a:.4f}{saved}")

print(f"\nBest validation accuracy: {best_vl_f:.4f}")

# ── Test evaluation ───────────────────────────────────────────────────────────
# Checkpoint keys: model_state_dict · epoch · val_acc · stat_features · stat_dim

ckpt = torch.load(ckpt_path, map_location=DEVICE)
model_f.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_, ta, preds, labels = eval_epoch3(model_f, te_ldr_f, crit_f, DEVICE, sm_f, ss_f)
t_test = time.time() - t0
n_test = len(te_ds_f)
ms_per_sample = (t_test / n_test) * 1000

metrics = utils.compute_metrics(preds, labels)

print(f"\n── Test Results ──────────────────────────────────────────────")
print(f"Test Accuracy : {metrics['accuracy']:.4f}")
print(f"Test Macro F1 : {metrics['macro_f1']:.4f}")
print(f"Stat features : {best_feat}")
print()
print(classification_report(labels, preds, target_names=CLASS_NAMES))

print(f"\n── Inference Timing ─────────────────────────────────────────")
print(f"Test set size         : {n_test} samples")
print(f"Total inference time  : {t_test:.2f} s")
print(f"Per-sample time       : {ms_per_sample:.3f} ms  →  {1000/ms_per_sample:.0f} samples/sec")
print(f"Estimated   100 files : {ms_per_sample *   100 / 1000:.2f} s")
print(f"Estimated 1 000 files : {ms_per_sample *  1000 / 1000:.2f} s")
print(f"Estimated 10 000 files: {ms_per_sample * 10000 / 1000:.2f} s")

utils.plot_confusion_matrix(preds, labels, CLASS_NAMES)

print(f"\nSaved: {ckpt_path}")
print(f"Saved: {os.path.join(MODELS_DIR, 'stat_scaler.pkl')}")
