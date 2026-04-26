"""
train_phase2b.py — Phase 2b: Mel-Spectrogram + Statistical Features (no MFCC)
Run from project root:  python train_phase2b.py

Purpose:
  Test whether replacing the MFCC stream (Phase 2) with cheap global statistical
  features gives comparable accuracy at ~2× lower inference cost.

  Phase 2  : Mel + MFCC   →  0.485 ms/sample   (99.91 % test acc)
  Phase 2b : Mel + Stat   →  ~0.25 ms/sample   (TBD — this script measures it)

What this script does:
  1. Reuses mel features already cached by Phase 1 (or re-computes them).
  2. Pre-computes all-5 statistical features once and saves them as .npy.
  3. Ablation: 5 configs (CNN frozen, 10 epochs each) to find best stat subset.
  4. Final fine-tune: unfreeze everything, best stat config, 20 epochs.
  5. Saves phase2b_best.pth + stat_scaler_2b.pkl.
"""

import os, time, pickle, json, random
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from machine_listener.src.preprocess import AudioPreprocessor, PreprocessConfig, AugmentationConfig
from machine_listener.src.dataset import MachineDataset
from machine_listener.src.features.mel_spectrogram import compute_mel_spectrogram
from machine_listener.src.features.statistical import compute_statistical_features, ALL_FEATURE_NAMES
from machine_listener.src.models.cnn_mel_stat import MelStatCNN
import machine_listener.src.train_utils as utils

# ─────────────────────────── CONFIG ───────────────────────────────────────────
ROOT_DIR    = "Students"
MODELS_DIR  = "machine_listener/outputs/saved_models"
PHASE1_CKPT = os.path.join(MODELS_DIR, "phase1_best.pth")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATS_DIR_MEL  = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mel"))
FEATS_DIR_STAT = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "stat"))
os.makedirs(FEATS_DIR_MEL,  exist_ok=True)
os.makedirs(FEATS_DIR_STAT, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE      = 32
ABLATION_EPOCHS = 10   # fast: CNN frozen, only stat branch + head train
FINETUNE_EPOCHS = 20   # full end-to-end fine-tune with best stat config
NUM_WORKERS     = 4

print(f"Device: {DEVICE}")

CLASS_NAMES = [
    "Machine1_Normal", "Machine1_Abnormal",
    "Machine2_Normal", "Machine2_Abnormal",
    "Machine3_Normal", "Machine3_Abnormal",
]

# Ablation plan: start minimal, add features one at a time
ABLATION_CONFIGS = [
    {"name": "rms_only",                    "features": ["rms"]},
    {"name": "rms_zcr",                     "features": ["rms", "zcr"]},
    {"name": "rms_zcr_centroid",            "features": ["rms", "zcr", "centroid"]},
    {"name": "rms_zcr_centroid_rolloff",    "features": ["rms", "zcr", "centroid", "rolloff"]},
    {"name": "all_five",                    "features": ["rms", "zcr", "centroid", "rolloff", "bandwidth"]},
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
        np.save(out, compute_statistical_features(w, feature_names=ALL_FEATURE_NAMES))
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
    """Loads pre-computed mel (.npy) and slices stat features from the all-5 stat array."""
    def __init__(self, mel_dir, stat_dir, labels, indices, stat_feature_names, augment=False):
        self.mel_dir   = pathlib.Path(mel_dir)
        self.stat_dir  = pathlib.Path(stat_dir)
        self.labels    = labels
        self.indices   = indices
        self.augment   = augment
        # Index positions into the 5-element stat array for selected features
        self.stat_cols = [ALL_FEATURE_NAMES.index(n) for n in stat_feature_names]

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri   = self.indices[idx]
        mel  = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        stat_all = np.load(self.stat_dir / f"{ri:06d}.npy")           # (5,)
        stat = torch.tensor(stat_all[self.stat_cols], dtype=torch.float32)
        if self.augment:
            mel = spec_augment(mel)
        return (mel, stat), torch.tensor(self.labels[ri], dtype=torch.long)

def collate_mel_stat(batch):
    """Stack (mel, stat) tuples separately for DataLoader."""
    feats, labels = zip(*batch)
    return (torch.stack([f[0] for f in feats]),
            torch.stack([f[1] for f in feats])), torch.stack(labels)

# ─────────────────────────── STAT SCALER ──────────────────────────────────────

def fit_scaler(dataset):
    """Compute mean & std from training split stat features."""
    all_stat = [dataset[i][0][1].numpy() for i in range(len(dataset))]
    arr = np.stack(all_stat, axis=0)
    return arr.mean(axis=0), arr.std(axis=0) + 1e-8

# ─────────────────────────── TRAIN / EVAL HELPERS ─────────────────────────────

def train_epoch_ms(model, loader, optimizer, criterion, device, s_mean, s_std):
    model.train()
    sm = torch.tensor(s_mean, dtype=torch.float32).to(device)
    ss = torch.tensor(s_std,  dtype=torch.float32).to(device)
    total_loss, correct, total = 0.0, 0, 0
    for (mel, stat), y in loader:
        mel, stat, y = mel.to(device), stat.to(device), y.to(device)
        stat = (stat - sm) / ss   # StandardScaler on GPU
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

# Scan files + create/load the same stratified split as Phase 1
_scan_ds   = MachineDataset(ROOT_DIR, _infer_prep, compute_mel_spectrogram, "train")
ALL_PATHS  = _scan_ds.paths
ALL_LABELS = _scan_ds.labels

# Pre-compute mel + stat (mel reused from Phase 1 cache if already there)
precompute_mel_stat(ALL_PATHS, FEATS_DIR_MEL, FEATS_DIR_STAT, _infer_prep, n_workers=NUM_WORKERS)

# Load split indices
_split_file = pathlib.Path(MODELS_DIR) / "split_indices.json"
if not _split_file.exists():
    _alt = pathlib.Path(ROOT_DIR).parent / "split_indices.json"
    if _alt.exists():
        import shutil; shutil.copy(_alt, _split_file)
_splits = json.load(open(_split_file))

label_counts  = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
class_weights = torch.tensor(1.0 / (label_counts + 1), dtype=torch.float32).to(DEVICE)
print(f"Label counts: {label_counts}")

# ─────────────────────────── ABLATION STUDY ───────────────────────────────────
print("\n══════════════════════════════════════════════════════════════")
print("  ABLATION STUDY — CNN frozen, one stat feature added at a time")
print("══════════════════════════════════════════════════════════════\n")

ablation_results = {}

for cfg_ab in ABLATION_CONFIGS:
    name     = cfg_ab["name"]
    feat     = cfg_ab["features"]
    stat_dim = len(feat)
    print(f"── {name}  (features: {feat}) ─────────────────────────────")

    train_ds = PrecomputedDatasetMelStat(FEATS_DIR_MEL, FEATS_DIR_STAT,
                                         ALL_LABELS, _splits["train"], feat, augment=True)
    val_ds   = PrecomputedDatasetMelStat(FEATS_DIR_MEL, FEATS_DIR_STAT,
                                         ALL_LABELS, _splits["val"],   feat, augment=False)
    s_mean, s_std = fit_scaler(train_ds)

    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_mel_stat,
                           num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    vl_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_mel_stat,
                           num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    model = MelStatCNN(num_classes=6, stat_dim=stat_dim).to(DEVICE)

    # Load Phase 1 weights into mel_stream
    p1_ckpt = torch.load(PHASE1_CKPT, map_location=DEVICE)
    model.mel_stream.load_state_dict(p1_ckpt["model_state_dict"])

    # Freeze CNN — only stat_branch + fc1 + fc2 train
    for p in model.mel_stream.parameters():
        p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt  = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    best_vl = 0.0
    for epoch in range(1, ABLATION_EPOCHS + 1):
        tr_l, tr_a    = train_epoch_ms(model, tr_loader, opt, crit, DEVICE, s_mean, s_std)
        vl_l, vl_a, _, _ = eval_epoch_ms(model, vl_loader, crit, DEVICE, s_mean, s_std)
        if vl_a > best_vl: best_vl = vl_a
        print(f"  Epoch {epoch:2d}/{ABLATION_EPOCHS}  train_acc={tr_a:.4f}  val_acc={vl_a:.4f}")

    ablation_results[name] = {"val_acc": best_vl, "features": feat, "stat_dim": stat_dim}
    print(f"  → Best val_acc: {best_vl:.4f}\n")

# Print comparison table
print("\n── Ablation Comparison ──────────────────────────────────────────")
print(f"{'Config':<35} {'N_features':>10} {'Val Acc':>8}")
print("-" * 57)
for k, v in ablation_results.items():
    print(f"{k:<35} {v['stat_dim']:>10} {v['val_acc']:>8.4f}")

best_name = max(ablation_results, key=lambda k: ablation_results[k]["val_acc"])
best_feat = ablation_results[best_name]["features"]
print(f"\nWinner: '{best_name}'  features={best_feat}")

# ─────────────────────────── FINAL FINE-TUNE ──────────────────────────────────
print(f"\n══════════════════════════════════════════════════════════════")
print(f"  FINAL FINE-TUNE — all layers unfrozen, {FINETUNE_EPOCHS} epochs")
print(f"══════════════════════════════════════════════════════════════\n")

stat_dim_f = len(best_feat)

train_ds_f = PrecomputedDatasetMelStat(FEATS_DIR_MEL, FEATS_DIR_STAT,
                                        ALL_LABELS, _splits["train"], best_feat, augment=True)
val_ds_f   = PrecomputedDatasetMelStat(FEATS_DIR_MEL, FEATS_DIR_STAT,
                                        ALL_LABELS, _splits["val"],   best_feat, augment=False)
test_ds_f  = PrecomputedDatasetMelStat(FEATS_DIR_MEL, FEATS_DIR_STAT,
                                        ALL_LABELS, _splits["test"],  best_feat, augment=False)

s_mean_f, s_std_f = fit_scaler(train_ds_f)

# Save scaler alongside checkpoint
scaler_path = os.path.join(MODELS_DIR, "stat_scaler_2b.pkl")
pickle.dump({"mean": s_mean_f, "std": s_std_f, "features": best_feat},
            open(scaler_path, "wb"))
print(f"Scaler saved → {scaler_path}")

tr_ldr = DataLoader(train_ds_f, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_mel_stat,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
vl_ldr = DataLoader(val_ds_f,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_mel_stat,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
te_ldr = DataLoader(test_ds_f,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_mel_stat,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)

model_f = MelStatCNN(num_classes=6, stat_dim=stat_dim_f).to(DEVICE)

# Load Phase 1 weights into mel_stream — protect them with a low LR
p1_ckpt = torch.load(PHASE1_CKPT, map_location=DEVICE)
model_f.mel_stream.load_state_dict(p1_ckpt["model_state_dict"])
model_f.unfreeze_cnn if hasattr(model_f, "unfreeze_cnn") else None
for p in model_f.parameters():
    p.requires_grad = True

# Differential LRs: mel_stream was trained in Phase 1 → use small LR to protect its weights
optimizer_f = torch.optim.AdamW([
    {"params": model_f.mel_stream.parameters(),   "lr": 1e-4},   # already trained
    {"params": model_f.stat_branch.parameters(),  "lr": 5e-4},   # new, learn fast
    {"params": model_f.fc1.parameters(),          "lr": 5e-4},
    {"params": model_f.fc2.parameters(),          "lr": 5e-4},
], weight_decay=1e-4)

crit_f  = nn.CrossEntropyLoss(weight=class_weights)
sched_f = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_f, T_max=FINETUNE_EPOCHS)

best_vl_f = 0.0
history   = {"train_acc": [], "val_acc": []}
ckpt_path = os.path.join(MODELS_DIR, "phase2b_best.pth")

for epoch in range(1, FINETUNE_EPOCHS + 1):
    tr_l, tr_a    = train_epoch_ms(model_f, tr_ldr, optimizer_f, crit_f, DEVICE, s_mean_f, s_std_f)
    vl_l, vl_a, _, _ = eval_epoch_ms(model_f, vl_ldr, crit_f, DEVICE, s_mean_f, s_std_f)
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
# │    machine_listener/outputs/saved_models/phase2b_best.pth                    │
# │      Keys: model_state_dict · epoch · val_acc · stat_features · stat_dim     │
# │    machine_listener/outputs/saved_models/stat_scaler_2b.pkl                  │
# │      Keys: mean · std · features  (needed at inference time)                 │
# └──────────────────────────────────────────────────────────────────────────────┘
ckpt = torch.load(ckpt_path, map_location=DEVICE)
model_f.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_, ta, preds, labels = eval_epoch_ms(model_f, te_ldr, crit_f, DEVICE, s_mean_f, s_std_f)
t_test = time.time() - t0
n_test = len(test_ds_f)
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
print(f"\n── Phase comparison note ───────────────────────────────────")
print(f"Phase 1 (Mel only)    : ~0.24 ms/sample  — baseline")
print(f"Phase 2 (Mel+MFCC)   : ~0.49 ms/sample  — 2× slower, +0.11% acc")
print(f"Phase 2b (Mel+Stat)  : {ms_per_sample:.3f} ms/sample  — this run")

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
