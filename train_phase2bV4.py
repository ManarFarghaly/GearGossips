"""
Phase 2b V4 — Mel-spectrogram CNN + Statistical features (19-d).
Run from project root: python train_phase2bV4.py

This is the best-performing model.
Requires: machine_listener/outputs/saved_models/phase1_best.pth
          (produced by train_phase1.py or train_phase1V3.py)

Key differences from V3:
  - Lower LR for new layers (5e-5 vs 3e-4) — more conservative fine-tuning
  - Earlier backbone unfreeze (epoch 2 vs epoch 7)
  - Heavier dropout in stat branch (0.5 vs 0.3)
  - Same 19 stat features: rms, zcr, rolloff, bandwidth, spectral_flux, kurtosis, mfcc_1..13
  - Per-machine stat normalisation with three separate scalers

Checkpoint saves: model weights, per-machine scalers, feature list, stat_dim.
infer.py loads this checkpoint directly — no separate scaler file needed.
"""

import os
import json
import time
import pickle
import random
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns

from machine_listener.src.preprocess import AudioPreprocessor, PreprocessConfig, AugmentationConfig
from machine_listener.src.dataset import scan_wav_files, SPLIT_DIR
from machine_listener.src.features.mel_spectrogram import compute_mel_spectrogram
from machine_listener.src.features.statistical import compute_stat_features_v4
from machine_listener.src.models.cnn_mel_stat_v4 import MelStatCNNV4, AttentionPool2d, MelCNNAttn
from machine_listener.src.split_utils import load_clean_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT_DIR   = "Students"
MODELS_DIR = "machine_listener/outputs/saved_models"
PHASE1_CKPT = os.path.join(MODELS_DIR, "phase1_best.pth")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATS_DIR_MEL  = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mel"))
FEATS_DIR_STAT = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "stat_v4"))
os.makedirs(FEATS_DIR_MEL,  exist_ok=True)
os.makedirs(FEATS_DIR_STAT, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE    = 32
TRAIN_EPOCHS  = 40
NUM_WORKERS   = 2
WEIGHT_DECAY  = 1e-3
LABEL_SMOOTH  = 0.1
ENS_BETA      = 0.99
MIXUP_ALPHA   = 0.4
FOCAL_GAMMA   = 1.0
HIER_ALPHA    = 0.3
WARMUP_EPOCHS = 0
RLROP_PATIENCE = 6
RLROP_FACTOR  = 0.5
ES_PATIENCE   = 15
UNFREEZE_EPOCH = 2

LR_MEL_STREAM = 1e-6
LR_NEW_LAYERS = 5e-5

STAT_FEATURES = (
    ["rms", "zcr", "rolloff", "bandwidth", "spectral_flux", "kurtosis"]
    + [f"mfcc_{i}" for i in range(1, 14)]
)
STAT_DIM = len(STAT_FEATURES)   # 19

_MACHINE_FROM_CLASS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
_FAULT_FROM_CLASS   = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}

CLASS_NAMES = [
    "Machine1_Normal", "Machine1_Abnormal",
    "Machine2_Normal", "Machine2_Abnormal",
    "Machine3_Normal", "Machine3_Abnormal",
]

print(f"Device: {DEVICE}")
print(f"Stat features ({STAT_DIM}): {STAT_FEATURES}")

if not os.path.exists(PHASE1_CKPT):
    raise FileNotFoundError(
        f"Phase 1 checkpoint not found at {PHASE1_CKPT}. "
        "Run train_phase1.py or train_phase1V3.py first."
    )

# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------

_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=16000, default_duration_sec=2.75,
    augmentation=AugmentationConfig(enabled=False),
))

# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------

def _precompute_one(args):
    idx, wav_path, mel_dir, stat_dir = args
    mel_out = pathlib.Path(mel_dir)  / f"{idx:06d}.npy"
    sta_out = pathlib.Path(stat_dir) / f"{idx:06d}.npy"
    if mel_out.exists() and sta_out.exists():
        return
    try:
        w = _prep.preprocess(str(wav_path), mode="inference")
        if not mel_out.exists():
            np.save(mel_out, compute_mel_spectrogram(w))
        if not sta_out.exists():
            np.save(sta_out, compute_stat_features_v4(w))
    except Exception:
        if not mel_out.exists():
            np.save(mel_out, np.zeros((1, 128, 84), dtype=np.float32))
        if not sta_out.exists():
            np.save(sta_out, np.zeros(STAT_DIM, dtype=np.float32))


def precompute_all(paths, mel_dir, stat_dir):
    import tqdm
    from concurrent.futures import ThreadPoolExecutor
    mel_dir  = pathlib.Path(mel_dir);  mel_dir.mkdir(parents=True, exist_ok=True)
    stat_dir = pathlib.Path(stat_dir); stat_dir.mkdir(parents=True, exist_ok=True)
    already  = sum(1 for i in range(len(paths))
                   if (mel_dir / f"{i:06d}.npy").exists() and (stat_dir / f"{i:06d}.npy").exists())
    if already == len(paths):
        print(f"All {len(paths)} mel+stat features cached — skipping")
        return
    print(f"Pre-computing features for {len(paths) - already} files ...")
    args = [(i, p, str(mel_dir), str(stat_dir)) for i, p in enumerate(paths)]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        list(tqdm.tqdm(ex.map(_precompute_one, args), total=len(paths), desc="features"))
    print("Pre-computation done.")


# ---------------------------------------------------------------------------
# Per-machine scalers
# ---------------------------------------------------------------------------

def fit_per_machine_scalers(stat_dir, labels, train_indices):
    stat_dir = pathlib.Path(stat_dir)
    buckets  = {m: [] for m in range(3)}
    for i in train_indices:
        buckets[_MACHINE_FROM_CLASS[labels[i]]].append(np.load(stat_dir / f"{i:06d}.npy"))
    scalers = []
    for m in range(3):
        arr  = np.stack(buckets[m])
        mean = arr.mean(0)
        std  = arr.std(0) + 1e-8
        scalers.append((mean, std))
        print(f"  Machine{m + 1}: n={len(buckets[m])}")
    return scalers


def normalise_per_machine(stat, machine_ids, scalers, device):
    out = stat.clone()
    for m, (mean, std) in enumerate(scalers):
        mask = machine_ids == m
        if mask.any():
            m_t = torch.tensor(mean, dtype=torch.float32, device=device)
            s_t = torch.tensor(std,  dtype=torch.float32, device=device)
            out[mask] = (stat[mask] - m_t) / s_t
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _spec_augment(mel, freq_mask=27, time_mask=15, n_freq=2, n_time=2):
    mel = mel.clone(); _, F, T = mel.shape
    for _ in range(n_freq):
        f  = random.randint(0, freq_mask); f0 = random.randint(0, max(F - f, 1))
        mel[:, f0:f0 + f, :] = 0.0
    for _ in range(n_time):
        t  = random.randint(0, time_mask); t0 = random.randint(0, max(T - t, 1))
        mel[:, :, t0:t0 + t] = 0.0
    return mel


class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, mel_dir, stat_dir, labels, indices, augment=False):
        self.mel_dir  = pathlib.Path(mel_dir)
        self.stat_dir = pathlib.Path(stat_dir)
        self.labels   = labels
        self.indices  = indices
        self.augment  = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri  = self.indices[idx]; lbl = self.labels[ri]
        mel  = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        stat = torch.tensor(np.load(self.stat_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment: mel = _spec_augment(mel)
        return (mel, stat,
                torch.tensor(lbl,                         dtype=torch.long),
                torch.tensor(_MACHINE_FROM_CLASS[lbl],   dtype=torch.long),
                torch.tensor(_FAULT_FROM_CLASS[lbl],     dtype=torch.long))


def _collate(batch):
    mel, stat, lbl, mach, fault = zip(*batch)
    return torch.stack(mel), torch.stack(stat), torch.stack(lbl), torch.stack(mach), torch.stack(fault)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float().unsqueeze(1),
            pos_weight=self.pos_weight, reduction="none"
        )
        return ((1.0 - torch.exp(-bce)) ** self.gamma * bce).mean()


def effective_num_weights(label_counts, beta=0.9999, num_classes=6):
    eff = (1.0 - np.power(beta, label_counts)) / (1.0 - beta)
    w   = 1.0 / eff
    return torch.tensor(w / w.sum() * num_classes, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def _mixup_criterion(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1.0 - lam) * crit(pred, y_b)


def train_epoch(model, loader, optimizer, crit_main, crit_machine, crit_fault,
                scalers, mixup_alpha=0.4, hier_alpha=0.3):
    model.train(); total_loss = correct = total = 0
    for mel, stat, y_main, y_machine, y_fault in loader:
        mel, stat = mel.to(DEVICE), stat.to(DEVICE)
        y_main    = y_main.to(DEVICE)
        y_machine = y_machine.to(DEVICE)
        y_fault   = y_fault.to(DEVICE)

        stat = normalise_per_machine(stat, y_machine, scalers, DEVICE)

        lam  = float(np.random.beta(mixup_alpha, mixup_alpha))
        lam  = max(lam, 1.0 - lam)
        perm = torch.randperm(mel.size(0), device=DEVICE)
        mel_m  = lam * mel  + (1.0 - lam) * mel[perm]
        stat_m = lam * stat + (1.0 - lam) * stat[perm]
        y_a, y_b = y_main, y_main[perm]

        optimizer.zero_grad()
        out_main, out_machine, out_fault = model(mel_m, stat_m)
        loss = (
            _mixup_criterion(crit_main, out_main, y_a, y_b, lam)
            + hier_alpha * crit_machine(out_machine, y_machine)
            + (1.0 - hier_alpha) * crit_fault(out_fault, y_fault)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (out_main.argmax(1) == y_a).sum().item()
        total      += y_main.size(0)
    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, crit_main, scalers):
    model.eval(); total_loss = correct = total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, stat, y_main, y_machine, y_fault in loader:
            mel, stat = mel.to(DEVICE), stat.to(DEVICE)
            y_main    = y_main.to(DEVICE)
            y_machine = y_machine.to(DEVICE)
            stat = normalise_per_machine(stat, y_machine, scalers, DEVICE)
            out_main, _, _ = model(mel, stat)
            preds = out_main.argmax(1)
            total_loss += crit_main(out_main, y_main).item()
            correct    += (preds == y_main).sum().item()
            total      += y_main.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_main.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, epoch, val_loss, val_acc, scalers, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch":            epoch,
        "val_loss":         val_loss,
        "val_acc":          val_acc,
        "stat_features":    STAT_FEATURES,
        "stat_dim":         STAT_DIM,
        "scalers":          scalers,
    }, path)


def load_phase1_weights(model, ckpt_path):
    ckpt     = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    remapped = {"mel_stream." + k: v for k, v in ckpt["model_state_dict"].items()}
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    loaded   = len(remapped) - len(unexpected)
    print(f"[ckpt] Loaded {loaded}/{len(remapped)} Phase 1 keys into mel_stream.")
    new_keys = [k for k in missing if not k.startswith("mel_stream.")]
    if new_keys:
        print(f"[ckpt] New layers (random init): {new_keys[:6]}"
              f"{'...' if len(new_keys) > 6 else ''}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_PATHS, ALL_LABELS = scan_wav_files(ROOT_DIR)
print(f"Found {len(ALL_PATHS)} files")

_splits = load_clean_split(SPLIT_DIR)
precompute_all(ALL_PATHS, FEATS_DIR_MEL, FEATS_DIR_STAT)

print("\nFitting per-machine stat scalers ...")
scalers = fit_per_machine_scalers(FEATS_DIR_STAT, ALL_LABELS, _splits["train"])

label_counts  = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
class_weights = effective_num_weights(label_counts, beta=ENS_BETA).to(DEVICE)

machine_counts = np.array([
    label_counts[0] + label_counts[1],
    label_counts[2] + label_counts[3],
    label_counts[4] + label_counts[5],
], dtype=np.float64)
machine_ens = (1.0 - np.power(ENS_BETA, machine_counts)) / (1.0 - ENS_BETA)
machine_w   = torch.tensor(
    1.0 / machine_ens / (1.0 / machine_ens).sum() * 3, dtype=torch.float32
).to(DEVICE)

fault_counts = np.array([
    sum(label_counts[i] for i in [0, 2, 4]),
    sum(label_counts[i] for i in [1, 3, 5]),
], dtype=np.float64)
fault_pos_w = torch.tensor(
    [fault_counts[0] / (fault_counts[1] + 1e-6)], dtype=torch.float32
).to(DEVICE)

print("\nClass counts (training split):")
for i, (name, cnt, w) in enumerate(zip(CLASS_NAMES, label_counts, class_weights.cpu())):
    print(f"  [{i}] {name:<25s}  n={cnt:>5d}  weight={w:.4f}")

kw = dict(collate_fn=_collate, num_workers=NUM_WORKERS, pin_memory=True,
          persistent_workers=False, prefetch_factor=2)
tr_ds = PrecomputedDataset(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["train"], augment=True)
vl_ds = PrecomputedDataset(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["val"],   augment=False)
te_ds = PrecomputedDataset(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["test"],  augment=False)
print(f"\nTrain: {len(tr_ds)}  Val: {len(vl_ds)}  Test: {len(te_ds)}")

tr_ldr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw)
vl_ldr = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, **kw)
te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, **kw)

model = MelStatCNNV4(num_classes=6, stat_dim=STAT_DIM).to(DEVICE)
load_phase1_weights(model, PHASE1_CKPT)

for param in model.mel_stream.parameters():
    param.requires_grad = False
print(f"Mel backbone frozen — unfreezes at epoch {UNFREEZE_EPOCH}")

optimizer = torch.optim.AdamW([
    {"params": model.mel_stream.parameters(),  "lr": LR_MEL_STREAM},
    {"params": model.stat_branch.parameters(), "lr": LR_NEW_LAYERS},
    {"params": model.fc1.parameters(),         "lr": LR_NEW_LAYERS},
    {"params": model.head_main.parameters(),   "lr": LR_NEW_LAYERS},
    {"params": model.head_machine.parameters(),"lr": LR_NEW_LAYERS},
    {"params": model.head_fault.parameters(),  "lr": LR_NEW_LAYERS},
], weight_decay=WEIGHT_DECAY)

crit_main    = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
crit_machine = nn.CrossEntropyLoss(weight=machine_w)
crit_fault   = BinaryFocalLoss(gamma=FOCAL_GAMMA, pos_weight=fault_pos_w)

plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=RLROP_FACTOR, patience=RLROP_PATIENCE, min_lr=1e-7
)

best_val_acc  = 0.0
best_val_loss = float("inf")
es_counter    = 0
history       = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
ckpt_path     = os.path.join(MODELS_DIR, "phase2b_v4_best.pth")

print(f"\nTraining Phase 2b V4 — Mel+Stat+MFCC | AttentionPool")
print(f"  Mixup a={MIXUP_ALPHA} | HierAlpha={HIER_ALPHA} | LabelSmooth={LABEL_SMOOTH}")
print(f"  ENS_beta={ENS_BETA} | Unfreeze@ep{UNFREEZE_EPOCH} | ES patience={ES_PATIENCE}")
print()

for epoch in range(1, TRAIN_EPOCHS + 1):
    current_lr = optimizer.param_groups[1]["lr"]

    if epoch == UNFREEZE_EPOCH:
        for param in model.mel_stream.parameters():
            param.requires_grad = True
        print(f"[unfreeze] Mel backbone unfrozen at epoch {epoch}")

    tr_loss, tr_acc       = train_epoch(model, tr_ldr, optimizer, crit_main,
                                         crit_machine, crit_fault, scalers,
                                         MIXUP_ALPHA, HIER_ALPHA)
    vl_loss, vl_acc, _, _ = eval_epoch(model, vl_ldr, crit_main, scalers)

    plateau_scheduler.step(vl_acc)
    for k, v in zip(["loss", "acc", "val_loss", "val_acc"], [tr_loss, tr_acc, vl_loss, vl_acc]):
        history[k].append(v)

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc; best_val_loss = vl_loss; es_counter = 0
        save_checkpoint(model, epoch, vl_loss, vl_acc, scalers, ckpt_path)
        tag = "  <- saved"
    else:
        es_counter += 1
        tag = f"  (patience {es_counter}/{ES_PATIENCE})"

    print(f"Epoch {epoch:3d}/{TRAIN_EPOCHS}  lr={current_lr:.2e}  "
          f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
          f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{tag}")

    if es_counter >= ES_PATIENCE:
        print(f"\nEarly stop: val acc unchanged for {ES_PATIENCE} epochs.")
        break

print(f"\nBest val acc: {best_val_acc:.4f}  (val loss: {best_val_loss:.4f})")

# Reload best checkpoint for evaluation
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_, test_acc, test_preds, test_labels = eval_epoch(model, te_ldr, crit_main, scalers)
ms_per_sample = (time.time() - t0) / len(te_ds) * 1000

print(f"\nTest Results")
print(f"  Accuracy : {test_acc:.4f}")
print(f"  Macro F1 : {f1_score(test_labels, test_preds, average='macro'):.4f}")
print()
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
print(f"Per-sample inference: {ms_per_sample:.3f} ms")

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.ylabel("True Label"); plt.xlabel("Predicted Label")
plt.title("Confusion Matrix — Phase 2b V4")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "phase2b_v4_confusion.png"))
plt.show()

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["loss"], label="train"); ax1.plot(history["val_loss"], label="val")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
ax2.plot(history["acc"],  label="train"); ax2.plot(history["val_acc"],  label="val")
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
plt.suptitle("Phase 2b V4 — Mel + Stat (Attention Pooling)")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "phase2b_v4_curves.png"))
plt.show()

print(f"\nSaved checkpoint: {ckpt_path}")
