"""
Phase 2b V3 — Mel-spectrogram CNN + Statistical features (19-d).
Run from project root: python train_phase2bV3.py

Requires: machine_listener/outputs/saved_models/phase1_best.pth
          (produced by train_phase1V4.py or train_phase1V3.py)

Changes from V2:
  - 19 stat features (adds spectral_flux + mfcc_1..13 to the previous 5)
  - AttentionPool2d in CNN block4 instead of AdaptiveAvgPool2d
  - Per-machine stat normalisation (3 separate scalers instead of one global)
  - Stat branch dropout reduced to 0.3
  - LR for new layers 3e-4, backbone unfrozen at epoch 7
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
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from machine_listener.src.preprocess import AudioPreprocessor, PreprocessConfig, AugmentationConfig
from machine_listener.src.dataset import scan_wav_files, SPLIT_DIR
from machine_listener.src.features.mel_spectrogram import compute_mel_spectrogram
from machine_listener.src.features.statistical import compute_stat_features_v4, ALL_FEATURE_NAMES_V4
from machine_listener.src.models.cnn_mel_stat_v4 import MelStatCNNV4
from machine_listener.src.split_utils import load_clean_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT_DIR    = "Students"
MODELS_DIR  = "machine_listener/outputs/saved_models"
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

LABEL_SMOOTH   = 0.05
ENS_BETA       = 0.9999
MIXUP_ALPHA    = 0.3
FOCAL_GAMMA    = 1.0
HIER_ALPHA     = 0.2
WEIGHT_DECAY   = 5e-4
WARMUP_EPOCHS  = 0
RLROP_PATIENCE = 6
RLROP_FACTOR   = 0.5
ES_PATIENCE    = 15
UNFREEZE_EPOCH = 7

LR_MEL_STREAM = 1e-6
LR_NEW_LAYERS = 3e-4

STAT_DIM     = len(ALL_FEATURE_NAMES_V4)   # 19
STAT_DROPOUT = 0.3

CLASS_NAMES   = ["Machine1_Normal", "Machine1_Abnormal",
                 "Machine2_Normal", "Machine2_Abnormal",
                 "Machine3_Normal", "Machine3_Abnormal"]
MACHINE_NAMES = ["Machine1", "Machine2", "Machine3"]

MACHINE_FROM_CLASS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
FAULT_FROM_CLASS   = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}

print(f"Device: {DEVICE}")
print(f"STAT_DIM={STAT_DIM}  features={ALL_FEATURE_NAMES_V4}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def precompute_features(paths, labels, mel_dir, stat_dir, preprocessor):
    mel_dir  = pathlib.Path(mel_dir)
    stat_dir = pathlib.Path(stat_dir)
    mel_dir.mkdir(parents=True, exist_ok=True)
    stat_dir.mkdir(parents=True, exist_ok=True)

    already = sum(
        1 for i in range(len(paths))
        if (mel_dir / f"{i:06d}.npy").exists() and (stat_dir / f"{i:06d}.npy").exists()
    )
    if already == len(paths):
        print(f"All {len(paths)} feature files already cached — skipping.")
        return

    print(f"Pre-computing mel+stat for {len(paths)} files ...")
    for i, wav in enumerate(paths):
        mel_out  = mel_dir  / f"{i:06d}.npy"
        stat_out = stat_dir / f"{i:06d}.npy"
        if mel_out.exists() and stat_out.exists():
            continue
        try:
            w = preprocessor.preprocess(str(wav), mode="inference")
            if not mel_out.exists():
                np.save(mel_out, compute_mel_spectrogram(w))
            if not stat_out.exists():
                np.save(stat_out, compute_stat_features_v4(w))
        except Exception as e:
            print(f"  [warn] {wav.name}: {e}")
            if not mel_out.exists():
                np.save(mel_out, np.zeros((1, 128, 84), dtype=np.float32))
            if not stat_out.exists():
                np.save(stat_out, np.zeros(STAT_DIM, dtype=np.float32))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(paths)}")
    print("Pre-computation done.")


def fit_per_machine_scalers(stat_dir, labels, train_indices):
    stat_dir     = pathlib.Path(stat_dir)
    machine_data = {m: [] for m in range(3)}
    for i in train_indices:
        m = MACHINE_FROM_CLASS[labels[i]]
        machine_data[m].append(np.load(stat_dir / f"{i:06d}.npy"))
    scalers = []
    for m in range(3):
        arr  = np.stack(machine_data[m])
        mean = arr.mean(0)
        std  = arr.std(0) + 1e-8
        scalers.append((mean, std))
        print(f"  Machine{m+1}: n={len(machine_data[m])}")
    return scalers


def normalise_stat(stat, machine_ids, scalers, device):
    out = stat.clone()
    for m, (mean, std) in enumerate(scalers):
        mask = (machine_ids == m)
        if mask.any():
            out[mask] = (
                stat[mask]
                - torch.tensor(mean, dtype=torch.float32, device=device)
            ) / torch.tensor(std, dtype=torch.float32, device=device)
    return out


def spec_augment(mel, freq_mask=27, time_mask=15, n_freq=2, n_time=2):
    mel = mel.clone()
    _, F, T = mel.shape
    for _ in range(n_freq):
        f  = random.randint(0, freq_mask)
        f0 = random.randint(0, max(F - f, 1))
        mel[:, f0:f0 + f, :] = 0.0
    for _ in range(n_time):
        t  = random.randint(0, time_mask)
        t0 = random.randint(0, max(T - t, 1))
        mel[:, :, t0:t0 + t] = 0.0
    return mel


class PrecomputedDatasetMS(torch.utils.data.Dataset):
    def __init__(self, mel_dir, stat_dir, labels, indices, augment=False):
        self.mel_dir  = pathlib.Path(mel_dir)
        self.stat_dir = pathlib.Path(stat_dir)
        self.labels   = labels
        self.indices  = indices
        self.augment  = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ri  = self.indices[idx]
        lbl = self.labels[ri]
        mel  = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        stat = torch.tensor(np.load(self.stat_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment:
            mel = spec_augment(mel)
        return (
            mel, stat,
            torch.tensor(lbl,                          dtype=torch.long),
            torch.tensor(MACHINE_FROM_CLASS[lbl],      dtype=torch.long),
            torch.tensor(FAULT_FROM_CLASS[lbl],        dtype=torch.long),
        )


def collate_ms(batch):
    mel, stat, lbl, mach, fault = zip(*batch)
    return torch.stack(mel), torch.stack(stat), torch.stack(lbl), torch.stack(mach), torch.stack(fault)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_phase1_weights(model, ckpt_path, device):
    if not pathlib.Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Phase 1 checkpoint not found: {ckpt_path}\n"
            "Run train_phase1V4.py (or train_phase1V3.py) first."
        )
    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
    remapped = {"mel_stream." + k: v for k, v in ckpt["model_state_dict"].items()}
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    loaded = len(remapped) - len(unexpected)
    print(f"[ckpt] Loaded {loaded}/{len(remapped)} Phase 1 keys into mel_stream.")
    new_keys = [k for k in missing if not k.startswith("mel_stream.")]
    if new_keys:
        print(f"[ckpt] New layers (random init): {new_keys[:6]}{'...' if len(new_keys) > 6 else ''}")


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        import torch.nn.functional as F
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float().unsqueeze(1),
            pos_weight=self.pos_weight, reduction="none"
        )
        return ((1.0 - torch.exp(-bce)) ** self.gamma * bce).mean()


def effective_num_weights(label_counts, beta=0.9999, num_classes=6):
    eff = (1.0 - np.power(beta, label_counts)) / (1.0 - beta)
    w   = 1.0 / eff
    w   = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, crit_main, crit_machine, crit_fault,
                scalers, device, mixup_alpha=0.3, hier_alpha=0.2):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for mel, stat, y_main, y_machine, y_fault in loader:
        mel      = mel.to(device)
        stat     = stat.to(device)
        y_main   = y_main.to(device)
        y_machine = y_machine.to(device)
        y_fault  = y_fault.to(device)

        stat = normalise_stat(stat, y_machine, scalers, device)

        if mixup_alpha > 0:
            lam  = float(np.random.beta(mixup_alpha, mixup_alpha))
            lam  = max(lam, 1.0 - lam)
            perm = torch.randperm(mel.size(0), device=device)
            mel_mix  = lam * mel  + (1.0 - lam) * mel[perm]
            stat_mix = lam * stat + (1.0 - lam) * stat[perm]
            y_a, y_b = y_main, y_main[perm]
        else:
            mel_mix, stat_mix, y_a, y_b, lam = mel, stat, y_main, y_main, 1.0

        optimizer.zero_grad()
        out_main, out_machine, out_fault = model(mel_mix, stat_mix)

        loss = (
            mixup_criterion(crit_main, out_main, y_a, y_b, lam)
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


def eval_epoch(model, loader, crit_main, scalers, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for mel, stat, y_main, y_machine, y_fault in loader:
            mel      = mel.to(device)
            stat     = stat.to(device)
            y_main   = y_main.to(device)
            y_machine = y_machine.to(device)

            stat     = normalise_stat(stat, y_machine, scalers, device)
            out_main, _, _ = model(mel, stat)
            preds    = out_main.argmax(1)

            total_loss += crit_main(out_main, y_main).item()
            correct    += (preds == y_main).sum().item()
            total      += y_main.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_main.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels


def save_checkpoint(model, epoch, val_loss, val_acc, scalers, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch":            epoch,
        "val_loss":         val_loss,
        "val_acc":          val_acc,
        "stat_features":    ALL_FEATURE_NAMES_V4,
        "stat_dim":         STAT_DIM,
        "scalers":          scalers,
    }, path)


def compute_metrics(preds, labels):
    preds, labels = np.array(preds), np.array(labels)
    return {
        "accuracy":     (preds == labels).mean(),
        "macro_f1":     f1_score(labels, preds, average="macro"),
        "per_class_f1": f1_score(labels, preds, average=None),
    }


def plot_confusion_matrix(preds, labels, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "phase2b_v3_confusion_matrix.png"), dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

all_paths, all_labels = scan_wav_files(ROOT_DIR)

splits = load_clean_split(SPLIT_DIR)
train_idx = splits["train"]
val_idx   = splits["val"]
test_idx  = splits["test"]
print(f"Split — Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

preprocessor = AudioPreprocessor()
precompute_features(all_paths, all_labels, FEATS_DIR_MEL, FEATS_DIR_STAT, preprocessor)

print("\n-- Fitting per-machine stat scalers --")
scalers = fit_per_machine_scalers(FEATS_DIR_STAT, all_labels, train_idx)

label_counts  = np.bincount([all_labels[i] for i in train_idx], minlength=6)
class_weights = effective_num_weights(label_counts, beta=ENS_BETA).to(DEVICE)

machine_counts = np.array([
    label_counts[0] + label_counts[1],
    label_counts[2] + label_counts[3],
    label_counts[4] + label_counts[5],
], dtype=np.float64)
machine_ens = (1.0 - np.power(ENS_BETA, machine_counts)) / (1.0 - ENS_BETA)
machine_w   = torch.tensor(1.0 / machine_ens / (1.0 / machine_ens).sum() * 3,
                            dtype=torch.float32).to(DEVICE)

fault_counts = np.array([
    sum(label_counts[i] for i in [0, 2, 4]),
    sum(label_counts[i] for i in [1, 3, 5]),
], dtype=np.float64)
fault_pos_w = torch.tensor([fault_counts[0] / (fault_counts[1] + 1e-6)],
                             dtype=torch.float32).to(DEVICE)

print("\n-- Class counts (training split) --")
for i, (name, cnt, w) in enumerate(zip(CLASS_NAMES, label_counts, class_weights.cpu())):
    print(f"  [{i}] {name:<25s}  n={cnt:>5d}  ENS_weight={w:.4f}")

kw = dict(collate_fn=collate_ms, num_workers=NUM_WORKERS,
          pin_memory=True, persistent_workers=False)
tr_ds  = PrecomputedDatasetMS(FEATS_DIR_MEL, FEATS_DIR_STAT, all_labels, train_idx, augment=True)
vl_ds  = PrecomputedDatasetMS(FEATS_DIR_MEL, FEATS_DIR_STAT, all_labels, val_idx,   augment=False)
te_ds  = PrecomputedDatasetMS(FEATS_DIR_MEL, FEATS_DIR_STAT, all_labels, test_idx,  augment=False)
tr_ldr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw)
vl_ldr = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, **kw)
te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, **kw)
print(f"\nTrain: {len(tr_ds)}  Val: {len(vl_ds)}  Test: {len(te_ds)}")

model = MelStatCNNV4(num_classes=6, stat_dim=STAT_DIM, stat_dropout=STAT_DROPOUT).to(DEVICE)
load_phase1_weights(model, PHASE1_CKPT, DEVICE)

for param in model.mel_stream.parameters():
    param.requires_grad = False
print(f"[freeze] Mel backbone frozen — unfreezes at epoch {UNFREEZE_EPOCH}")

optimizer = torch.optim.AdamW([
    {"params": model.mel_stream.parameters(),   "lr": LR_MEL_STREAM},
    {"params": model.stat_branch.parameters(),  "lr": LR_NEW_LAYERS},
    {"params": model.fc1.parameters(),          "lr": LR_NEW_LAYERS},
    {"params": model.head_main.parameters(),    "lr": LR_NEW_LAYERS},
    {"params": model.head_machine.parameters(), "lr": LR_NEW_LAYERS},
    {"params": model.head_fault.parameters(),   "lr": LR_NEW_LAYERS},
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
ckpt_path     = os.path.join(MODELS_DIR, "phase2b_v3_best.pth")

print(f"\n-- Training Phase 2b V3 --")
print(f"   19 stat features | AttentionPool | Mixup a={MIXUP_ALPHA} | HierAlpha={HIER_ALPHA}")
print(f"   LabelSmooth={LABEL_SMOOTH} | ENS_beta={ENS_BETA} | Unfreeze@ep{UNFREEZE_EPOCH}")
print()

for epoch in range(1, TRAIN_EPOCHS + 1):
    current_lr = optimizer.param_groups[1]["lr"]

    if epoch == UNFREEZE_EPOCH:
        for param in model.mel_stream.parameters():
            param.requires_grad = True
        print(f"[unfreeze] Mel backbone unfrozen at epoch {epoch} (mel_lr={LR_MEL_STREAM:.0e})")

    tr_loss, tr_acc = train_epoch(
        model, tr_ldr, optimizer,
        crit_main, crit_machine, crit_fault,
        scalers, DEVICE, mixup_alpha=MIXUP_ALPHA, hier_alpha=HIER_ALPHA
    )
    vl_loss, vl_acc, _, _ = eval_epoch(model, vl_ldr, crit_main, scalers, DEVICE)
    plateau_scheduler.step(vl_acc)

    for k, v in zip(["loss", "acc", "val_loss", "val_acc"],
                    [tr_loss, tr_acc, vl_loss, vl_acc]):
        history[k].append(v)

    if vl_acc > best_val_acc:
        best_val_acc  = vl_acc
        best_val_loss = vl_loss
        es_counter    = 0
        save_checkpoint(model, epoch, vl_loss, vl_acc, scalers, ckpt_path)
        tag = "  <- saved"
    else:
        es_counter += 1
        tag = f"  (patience {es_counter}/{ES_PATIENCE})"

    print(
        f"Epoch {epoch:3d}/{TRAIN_EPOCHS}  lr={current_lr:.2e}  "
        f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
        f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{tag}"
    )

    if es_counter >= ES_PATIENCE:
        print(f"\n[early stop] Val acc did not improve for {ES_PATIENCE} epochs.")
        break

print(f"\nBest val acc: {best_val_acc:.4f}  (val loss: {best_val_loss:.4f})")

# Final evaluation on test set
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_, test_acc, test_preds, test_labels = eval_epoch(model, te_ldr, crit_main, scalers, DEVICE)
ms_per_sample = (time.time() - t0) / len(te_ds) * 1000

metrics = compute_metrics(test_preds, test_labels)
print(f"\n-- Test Results --")
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"Macro F1 : {metrics['macro_f1']:.4f}")
print("\nPer-class F1:")
for name, f1 in zip(CLASS_NAMES, metrics["per_class_f1"]):
    print(f"  {name}: {f1:.4f}")
print("\n", classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
print(f"Per-sample inference: {ms_per_sample:.3f} ms")

plot_confusion_matrix(test_preds, test_labels, CLASS_NAMES, "Phase 2b V3 — Confusion Matrix")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["loss"],     label="train"); ax1.plot(history["val_loss"], label="val")
ax1.set_title("Loss"); ax1.legend()
ax2.plot(history["acc"],      label="train"); ax2.plot(history["val_acc"],  label="val")
ax2.set_title("Accuracy"); ax2.legend()
plt.suptitle("Phase 2b V3 — Mel + 19 Stat features (Attention Pooling)")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "phase2b_v3_curves.png"), dpi=150)
plt.show()

scaler_path = os.path.join(MODELS_DIR, "stat_scaler_v3.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump({"scalers": scalers, "features": ALL_FEATURE_NAMES_V4}, f)
print(f"Scaler saved -> {scaler_path}")
print(f"Checkpoint   -> {ckpt_path}")
