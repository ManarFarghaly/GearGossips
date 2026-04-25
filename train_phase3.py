"""
train_phase3.py — Phase 3: Mel + MFCC + Statistical Features (Ablation + Fine-tune)
Run from project root: python train_phase3.py

What's new vs Phase 2:
  - feature_fn now returns a 3-TUPLE: (mel, mfcc, stat)
  - stat is a small vector of 5 global scalars (RMS, ZCR, Centroid, Rolloff, BW)
  - MelMFCCStatCNN adds a small FC branch for stat alongside the CNN branch
  - Ablation: run 5 experiments with CNN FROZEN, adding one stat feature at a time
  - Final run: unfreeze everything and fine-tune end-to-end with the best config
  - StandardScaler normalises stat features (each feature has a very different scale)
"""

import os, pickle, time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import pathlib, json
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from machine_listener.src.preprocess import AudioPreprocessor, PreprocessConfig, AugmentationConfig
from machine_listener.src.features.mel_spectrogram import compute_mel_spectrogram
from machine_listener.src.features.mfcc import compute_mfcc
from machine_listener.src.features.statistical import compute_statistical_features
from machine_listener.src.models.cnn_statistical import MelMFCCStatCNN
import machine_listener.src.train_utils as utils
from sklearn.model_selection import train_test_split

# ─────────────────────────── CONFIG ───────────────────────────────────────────
ROOT_DIR   = "Students"
MODELS_DIR = "machine_listener/outputs/saved_models"
PHASE2_CKPT = os.path.join(MODELS_DIR, "phase2_best.pth")
os.makedirs(MODELS_DIR, exist_ok=True)

NUM_WORKERS    = 4
FEATS_DIR_MEL  = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mel"))
FEATS_DIR_MFCC = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mfcc"))
FEATS_DIR_STAT = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "stat"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE       = 32
ABLATION_EPOCHS  = 10   # fast: CNN is frozen, only the stat branch trains
FINETUNE_EPOCHS  = 20   # full end-to-end fine-tune after picking best config

print(f"Device: {DEVICE}")

CLASS_NAMES = [
    "Machine1_Normal", "Machine1_Abnormal",
    "Machine2_Normal", "Machine2_Abnormal",
    "Machine3_Normal", "Machine3_Abnormal",
]

# Ablation plan: add one feature at a time → see how much each one helps
# Running order matters: start with the most commonly impactful feature (RMS)
ABLATION_CONFIGS = [
    {"name": "rms_only",                    "features": ["rms"]},
    {"name": "rms_zcr",                     "features": ["rms", "zcr"]},
    {"name": "rms_zcr_centroid",            "features": ["rms", "zcr", "centroid"]},
    {"name": "rms_zcr_centroid_rolloff",    "features": ["rms", "zcr", "centroid", "rolloff"]},
    {"name": "all_five",                    "features": ["rms", "zcr", "centroid", "rolloff", "bandwidth"]},
]

# ─────────────────────────── PRE-COMPUTE HELPERS ──────────────────────────────

def spec_augment(mel, freq_mask=30, time_mask=15, n_freq=2, n_time=2):
    mel=mel.clone(); _,F,T=mel.shape
    for _ in range(n_freq):
        f=random.randint(0,freq_mask); f0=random.randint(0,max(F-f,1)); mel[:,f0:f0+f,:]=0.
    for _ in range(n_time):
        t=random.randint(0,time_mask); t0=random.randint(0,max(T-t,1)); mel[:,:,t0:t0+t]=0.
    return mel

_ALL_STAT_NAMES = ["rms","zcr","centroid","rolloff","bandwidth"]

def _precompute_triple_one(args):
    """Save mel (1,128,84), mfcc (3,40,84), and all-5 stat features (5,) for one wav."""
    idx, wav_path, mel_dir, mfcc_dir, stat_dir, preprocessor = args
    mel_out  = pathlib.Path(mel_dir)  / f"{idx:06d}.npy"
    mfcc_out = pathlib.Path(mfcc_dir) / f"{idx:06d}.npy"
    stat_out = pathlib.Path(stat_dir) / f"{idx:06d}.npy"
    if mel_out.exists() and mfcc_out.exists() and stat_out.exists(): return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        np.save(mel_out,  compute_mel_spectrogram(w))
        np.save(mfcc_out, compute_mfcc(w))
        np.save(stat_out, compute_statistical_features(w, feature_names=_ALL_STAT_NAMES))
    except Exception:
        np.save(mel_out,  np.zeros((1,128,84),dtype=np.float32))
        np.save(mfcc_out, np.zeros((3,40,84), dtype=np.float32))
        np.save(stat_out, np.zeros(5,          dtype=np.float32))

def precompute_triple(paths, mel_dir, mfcc_dir, stat_dir, preprocessor, n_workers=4):
    import pathlib as _pl, tqdm
    for d in [mel_dir, mfcc_dir, stat_dir]: _pl.Path(d).mkdir(parents=True, exist_ok=True)
    already = sum(1 for i in range(len(paths))
                  if all((_pl.Path(d)/f"{i:06d}.npy").exists()
                         for d in [mel_dir,mfcc_dir,stat_dir]))
    if already == len(paths):
        print(f"All {len(paths)} mel+mfcc+stat already cached  (skipping)"); return
    print(f"Pre-computing mel+mfcc+stat for {len(paths)} files with {n_workers} threads ...")
    args=[(i,p,str(mel_dir),str(mfcc_dir),str(stat_dir),preprocessor) for i,p in enumerate(paths)]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(tqdm.tqdm(ex.map(lambda a:_precompute_triple_one(a),args),total=len(args),desc="features"))
    print("Pre-computation done.")

# ─────────────────────────── PHASE 3 DATASET ──────────────────────────────────
# We need a dataset that returns (mel, mfcc, stat_raw) triples.
# We build it here rather than modifying MachineDataset,
# so MachineDataset stays generic.

class MachineDataset3(Dataset):
    """Returns ((mel_t, mfcc_t, stat_t), label_t). stat is unscaled — scaled in train loop."""

    LABEL_MAP = {
        ("machine1","Normal"):0, ("machine1","Abnormal"):1,
        ("machine2","Normal"):2, ("machine2","Abnormal"):3,
        ("machine3","Normal"):4, ("machine3","Abnormal"):5,
    }

    def __init__(self, root_dir, preprocessor, stat_features, split, augment=False):
        self.root_dir     = pathlib.Path(root_dir)
        self.preprocessor = preprocessor
        self.stat_features = stat_features
        self.split        = split
        self.augment      = augment
        self.paths, self.labels = self._scan()
        self.indices             = self._load_split()

    def _scan(self):
        paths, labels = [], []
        for f in self.root_dir.rglob("*.wav"):
            lbl = self.LABEL_MAP.get((f.parent.parent.name, f.parent.name))
            if lbl is not None:
                paths.append(f); labels.append(lbl)
        if not paths:
            raise RuntimeError(f"No .wav files under {self.root_dir}")
        return paths, labels

    def _load_split(self):
        # Reuse the same split_indices.json from Phase 1/2 — critical for fair comparison
        sf_path = pathlib.Path(MODELS_DIR) / "split_indices.json"
        if sf_path.exists():
            return json.load(open(sf_path))[self.split]
        # Fallback: create if doesn't exist (first run)
        idx = list(range(len(self.paths)))
        tr, tmp, _, tl = train_test_split(idx, self.labels, test_size=0.30,
                                          stratify=self.labels, random_state=42)
        va, te = train_test_split(tmp, test_size=0.50, stratify=tl, random_state=42)
        json.dump({"train":tr,"val":va,"test":te}, open(sf_path,"w"))
        return {"train":tr,"val":va,"test":te}[self.split]

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri  = self.indices[idx]
        lbl = self.labels[ri]
        mode = "train" if (self.split == "train" and self.augment) else "inference"
        w = self.preprocessor.preprocess(self.paths[ri], mode=mode)

        mel  = compute_mel_spectrogram(w)                                 # (1,128,84)
        mfcc = compute_mfcc(w)                                            # (3,40,84)
        stat = compute_statistical_features(w, feature_names=self.stat_features)  # (N,)

        return (
            torch.tensor(mel,  dtype=torch.float32),
            torch.tensor(mfcc, dtype=torch.float32),
            torch.tensor(stat, dtype=torch.float32),
        ), torch.tensor(lbl, dtype=torch.long)

def collate3(batch):
    feats, labels = zip(*batch)
    return (
        torch.stack([f[0] for f in feats]),   # mel  (B,1,128,84)
        torch.stack([f[1] for f in feats]),   # mfcc (B,3,40,84)
        torch.stack([f[2] for f in feats]),   # stat (B,N)
    ), torch.stack(labels)

# ─────────────────────────── PRECOMPUTED DATASET ──────────────────────────────

class PrecomputedDataset3(torch.utils.data.Dataset):
    """Loads precomputed mel+mfcc, computes stat by slicing from precomputed all-5 array."""
    def __init__(self, mel_dir, mfcc_dir, stat_dir, labels, indices, stat_feature_names, augment=False):
        self.mel_dir  = pathlib.Path(mel_dir)
        self.mfcc_dir = pathlib.Path(mfcc_dir)
        self.stat_dir = pathlib.Path(stat_dir)
        self.labels   = labels; self.indices = indices; self.augment = augment
        # Indices into the 5-element stat array for the features we want
        self.stat_cols = [_ALL_STAT_NAMES.index(n) for n in stat_feature_names]
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        ri   = self.indices[idx]
        mel  = torch.tensor(np.load(self.mel_dir /f"{ri:06d}.npy"), dtype=torch.float32)
        mfcc = torch.tensor(np.load(self.mfcc_dir/f"{ri:06d}.npy"), dtype=torch.float32)
        stat_all = np.load(self.stat_dir/f"{ri:06d}.npy")  # (5,)
        stat = torch.tensor(stat_all[self.stat_cols],         dtype=torch.float32)
        if self.augment: mel = spec_augment(mel)
        return (mel, mfcc, stat), torch.tensor(self.labels[ri], dtype=torch.long)

# ─────────────────────────── STAT SCALER ──────────────────────────────────────
def fit_scaler(dataset):
    """Compute mean & std of stat features from the PrecomputedDataset3 training split."""
    all_stat = [dataset[i][0][2].numpy() for i in range(len(dataset))]
    arr = np.stack(all_stat, axis=0)
    return arr.mean(axis=0), arr.std(axis=0) + 1e-8

# ─────────────────────────── TRAIN / EVAL HELPERS (3-input) ───────────────────
def train_epoch3(model, loader, optimizer, criterion, device, s_mean, s_std):
    model.train()
    sm = torch.tensor(s_mean, dtype=torch.float32).to(device)
    ss = torch.tensor(s_std,  dtype=torch.float32).to(device)
    total_loss, correct, total = 0.0, 0, 0
    for (mel, mfcc, stat), y in loader:
        mel, mfcc, stat, y = mel.to(device), mfcc.to(device), stat.to(device), y.to(device)
        stat = (stat - sm) / ss      # StandardScaler normalisation (on GPU)
        optimizer.zero_grad()
        out  = model(mel, mfcc, stat)
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
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
            stat = (stat - sm) / ss
            out  = model(mel, mfcc, stat)
            loss = criterion(out, y)
            preds = out.argmax(1)
            total_loss += loss.item()
            correct    += (preds == y).sum().item()
            total      += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels

def load_phase2_weights(model, ckpt_path, device):
    """Copy mel_stream and mfcc_stream weights from Phase 2 checkpoint."""
    sd = torch.load(ckpt_path, map_location=device)["model_state_dict"]
    # Keys in sd are like "mel_stream.block1.0.weight" — strip prefix to match sub-module
    model.mel_stream.load_state_dict(
        {k[len("mel_stream."):]: v for k, v in sd.items() if k.startswith("mel_stream.")}
    )
    model.mfcc_stream.load_state_dict(
        {k[len("mfcc_stream."):]: v for k, v in sd.items() if k.startswith("mfcc_stream.")}
    )
    print("Loaded Phase 2 CNN weights")

if not os.path.exists(PHASE2_CKPT):
    raise FileNotFoundError(f"Phase 2 checkpoint not found at {PHASE2_CKPT}. Run train_phase2.py first.")

# ─── SCAN + PRE-COMPUTE ────────────────────────────────────────────────────────
_infer_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=16000, default_duration_sec=2.75,
    augmentation=AugmentationConfig(enabled=False),
))
_scan_ds  = MachineDataset3(ROOT_DIR, _infer_prep, _ALL_STAT_NAMES, "train")
ALL_PATHS  = _scan_ds.paths
ALL_LABELS = _scan_ds.labels
precompute_triple(ALL_PATHS, FEATS_DIR_MEL, FEATS_DIR_MFCC, FEATS_DIR_STAT,
                  _infer_prep, n_workers=NUM_WORKERS)

import json as _json
_split_file = pathlib.Path(MODELS_DIR) / "split_indices.json"
if not _split_file.exists():
    _alt = pathlib.Path(ROOT_DIR).parent / "split_indices.json"
    if _alt.exists(): import shutil; shutil.copy(_alt, _split_file)
_splits = _json.load(open(_split_file))

# ─────────────────────────── ABLATION STUDY ───────────────────────────────────
# For each config: load Phase 2 CNN weights, FREEZE the CNN, train ONLY stat branch + head.
# This is fast (~10 epochs) and tells you exactly how much each feature contributes.
# After the ablation we pick the best config and do a full fine-tune.

print("\n══════════════════════════════════════════════════════════════")
print("  ABLATION STUDY — CNN frozen, one stat feature added at a time")
print("══════════════════════════════════════════════════════════════\n")

ablation_results = {}

for cfg_ab in ABLATION_CONFIGS:
    name     = cfg_ab["name"]
    feat     = cfg_ab["features"]
    stat_dim = len(feat)

    print(f"── {name}  (features: {feat}) ──────────────────────────────")

    train_ds = PrecomputedDataset3(FEATS_DIR_MEL, FEATS_DIR_MFCC, FEATS_DIR_STAT,
                                   ALL_LABELS, _splits["train"], feat, augment=True)
    val_ds   = PrecomputedDataset3(FEATS_DIR_MEL, FEATS_DIR_MFCC, FEATS_DIR_STAT,
                                   ALL_LABELS, _splits["val"],   feat, augment=False)
    s_mean, s_std = fit_scaler(train_ds)
    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate3,
                           num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    vl_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate3,
                           num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    model = MelMFCCStatCNN(num_classes=6, stat_dim=stat_dim).to(DEVICE)
    load_phase2_weights(model, PHASE2_CKPT, DEVICE)
    model.freeze_cnn()   # freeze mel_stream + mfcc_stream — only stat_branch + fc1/fc2 update

    # Only optimise the unfrozen parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=1e-4)

    lc = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(1.0/(lc+1), dtype=torch.float32).to(DEVICE))

    best_vl = 0.0
    for epoch in range(1, ABLATION_EPOCHS + 1):
        tr_l, tr_a = train_epoch3(model, tr_loader, opt, crit, DEVICE, s_mean, s_std)
        vl_l, vl_a, _, _ = eval_epoch3(model, vl_loader, crit, DEVICE, s_mean, s_std)
        if vl_a > best_vl: best_vl = vl_a
        print(f"  Epoch {epoch:2d}/{ABLATION_EPOCHS}  train_acc={tr_a:.4f}  val_acc={vl_a:.4f}")

    ablation_results[name] = {"val_acc": best_vl, "features": feat, "stat_dim": stat_dim}
    print(f"  → Best val_acc: {best_vl:.4f}\n")

# Print comparison table — this goes in your report
print("\n── Ablation Comparison ──────────────────────────────────────────")
print(f"{'Config':<35} {'N_features':>10} {'Val Acc':>8}")
print("-" * 57)
for k, v in ablation_results.items():
    print(f"{k:<35} {v['stat_dim']:>10} {v['val_acc']:>8.4f}")

best_name = max(ablation_results, key=lambda k: ablation_results[k]["val_acc"])
best_feat = ablation_results[best_name]["features"]
print(f"\nWinner: '{best_name}'  features={best_feat}")

# ─────────────────────────── FINAL FINE-TUNE ──────────────────────────────────
# Now unfreeze everything and train end-to-end with the winning stat config.
# Use very low LR on CNN (already trained twice), higher on new branches.

print(f"\n══════════════════════════════════════════════════════════════")
print(f"  FINAL FINE-TUNE — all layers unfrozen, {FINETUNE_EPOCHS} epochs")
print(f"══════════════════════════════════════════════════════════════\n")

stat_dim_f = len(best_feat)

train_ds_f = PrecomputedDataset3(FEATS_DIR_MEL, FEATS_DIR_MFCC, FEATS_DIR_STAT,
                                 ALL_LABELS, _splits["train"], best_feat, augment=True)
val_ds_f   = PrecomputedDataset3(FEATS_DIR_MEL, FEATS_DIR_MFCC, FEATS_DIR_STAT,
                                 ALL_LABELS, _splits["val"],   best_feat, augment=False)
test_ds_f  = PrecomputedDataset3(FEATS_DIR_MEL, FEATS_DIR_MFCC, FEATS_DIR_STAT,
                                 ALL_LABELS, _splits["test"],  best_feat, augment=False)

s_mean_f, s_std_f = fit_scaler(train_ds_f)

# Save scaler — infer.py needs it to normalise stat features at inference time
pickle.dump({
    "mean": s_mean_f, "std": s_std_f, "features": best_feat,
}, open(os.path.join(MODELS_DIR, "stat_scaler.pkl"), "wb"))
print("Scaler saved to stat_scaler.pkl")

tr_ldr = DataLoader(train_ds_f, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate3,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
vl_ldr = DataLoader(val_ds_f,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate3,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
te_ldr = DataLoader(test_ds_f,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate3,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)

model_f = MelMFCCStatCNN(num_classes=6, stat_dim=stat_dim_f).to(DEVICE)
load_phase2_weights(model_f, PHASE2_CKPT, DEVICE)
model_f.unfreeze_cnn()

# Parameter groups: the CNN has been trained twice already — extremely low LR
# New stat branch and fusion head still need to learn — medium LR
optimizer_f = torch.optim.AdamW([
    {"params": model_f.mel_stream.parameters(),   "lr": 5e-5},  # trained in Phase 1+2
    {"params": model_f.mfcc_stream.parameters(),  "lr": 5e-5},  # trained in Phase 2
    {"params": model_f.stat_branch.parameters(),  "lr": 2e-4},  # still relatively new
    {"params": model_f.fc1.parameters(),          "lr": 2e-4},
    {"params": model_f.fc2.parameters(),          "lr": 2e-4},
], weight_decay=1e-4)

label_counts = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
crit_f = nn.CrossEntropyLoss(weight=torch.tensor(1.0/(label_counts+1), dtype=torch.float32).to(DEVICE))
sched_f = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_f, T_max=FINETUNE_EPOCHS)

best_vl_f = 0.0
history   = {"train_acc": [], "val_acc": []}
ckpt_path = os.path.join(MODELS_DIR, "phase3_best.pth")

for epoch in range(1, FINETUNE_EPOCHS + 1):
    tr_l, tr_a = train_epoch3(model_f, tr_ldr, optimizer_f, crit_f, DEVICE, s_mean_f, s_std_f)
    vl_l, vl_a, _, _ = eval_epoch3(model_f, vl_ldr, crit_f, DEVICE, s_mean_f, s_std_f)
    sched_f.step()

    history["train_acc"].append(tr_a)
    history["val_acc"].append(vl_a)

    saved = ""
    if vl_a > best_vl_f:
        best_vl_f = vl_a
        torch.save({
            "model_state_dict": model_f.state_dict(),
            "epoch": epoch, "val_acc": vl_a,
            "stat_features": best_feat, "stat_dim": stat_dim_f,
        }, ckpt_path)
        saved = "  ← best saved"

    print(f"Epoch {epoch:3d}/{FINETUNE_EPOCHS}  train_acc={tr_a:.4f}  val_acc={vl_a:.4f}{saved}")

print(f"\nBest validation accuracy: {best_vl_f:.4f}")

# ─────────────────────────── TEST EVALUATION ──────────────────────────────────
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  CHECKPOINTS SAVED TO:                                                       │
# │    machine_listener/outputs/saved_models/phase3_best.pth                     │
# │      Keys: model_state_dict · epoch · val_acc · stat_features · stat_dim     │
# │    machine_listener/outputs/saved_models/stat_scaler.pkl                     │
# │      Keys: mean · std · features  (needed at inference time for stat branch) │
# └──────────────────────────────────────────────────────────────────────────────┘
ckpt = torch.load(ckpt_path, map_location=DEVICE)
model_f.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_, ta, preds, labels = eval_epoch3(model_f, te_ldr, crit_f, DEVICE, s_mean_f, s_std_f)
t_test = time.time() - t0
n_test = len(test_ds_f)
ms_per_sample = (t_test / n_test) * 1000   # milliseconds per sample

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
