"""
Phase 1 V4 — Mel-spectrogram 2-D CNN with hierarchical output heads.
Run from project root: python train_phase1V4.py

Saves checkpoint to: machine_listener/outputs/saved_models/phase1_best.pth
This checkpoint is the required input for train_phase2bV3.py and train_phase2bV4.py.

Changes from V3:
  - ReduceLROnPlateau replaces cosine annealing (adapts to val_loss instead of fixed schedule)
  - Mixup alpha reduced from 0.3 to 0.1 (less aggressive mixing for hard Machine3 boundary)
  - ES patience reduced from 10 to 4 (stops earlier to avoid overfitting)
  - WandB removed (local training)
"""

import os
import json
import time
import random
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from machine_listener.src.preprocess import AudioPreprocessor, PreprocessConfig, AugmentationConfig
from machine_listener.src.dataset import scan_wav_files, SPLIT_DIR
from machine_listener.src.features.mel_spectrogram import compute_mel_spectrogram
from machine_listener.src.models.cnn_baseline import MelCNNHier
from machine_listener.src.split_utils import load_clean_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT_DIR   = "Students"
MODELS_DIR = "machine_listener/outputs/saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATS_DIR = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mel"))
os.makedirs(FEATS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE    = 32
EPOCHS        = 20
NUM_WORKERS   = 2

LR             = 1e-3
WEIGHT_DECAY   = 5e-4
WARMUP_EPOCHS  = 2
ES_PATIENCE    = 4
RLROP_PATIENCE = 2
RLROP_FACTOR   = 0.5

MIXUP_ALPHA  = 0.1
LABEL_SMOOTH = 0.1
ENS_BETA     = 0.9999
HIER_ALPHA   = 0.4
FOCAL_GAMMA  = 2.0

CLASS_NAMES   = ["Machine1_Normal", "Machine1_Abnormal",
                 "Machine2_Normal", "Machine2_Abnormal",
                 "Machine3_Normal", "Machine3_Abnormal"]
MACHINE_NAMES = ["Machine1", "Machine2", "Machine3"]

MACHINE_FROM_CLASS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
FAULT_FROM_CLASS   = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}

print(f"Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def precompute_mel(paths, feats_dir, preprocessor):
    feats_dir = pathlib.Path(feats_dir)
    feats_dir.mkdir(parents=True, exist_ok=True)

    already = sum(1 for i in range(len(paths)) if (feats_dir / f"{i:06d}.npy").exists())
    if already == len(paths):
        print(f"All {len(paths)} mel features already cached — skipping.")
        return

    print(f"Pre-computing mel features for {len(paths)} files ...")
    for i, wav in enumerate(paths):
        out_path = feats_dir / f"{i:06d}.npy"
        if out_path.exists():
            continue
        try:
            w   = preprocessor.preprocess(str(wav), mode="inference")
            mel = compute_mel_spectrogram(w)
            np.save(out_path, mel)
        except Exception as e:
            print(f"  [warn] {wav.name}: {e}")
            np.save(out_path, np.zeros((1, 128, 84), dtype=np.float32))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(paths)}")
    print("Pre-computation done.")


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


class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, feats_dir, labels, indices, augment=False):
        self.feats_dir = pathlib.Path(feats_dir)
        self.labels    = labels
        self.indices   = indices
        self.augment   = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ri  = self.indices[idx]
        lbl = self.labels[ri]
        mel = torch.tensor(np.load(self.feats_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment:
            mel = spec_augment(mel)
        return (
            mel,
            torch.tensor(lbl,                       dtype=torch.long),
            torch.tensor(MACHINE_FROM_CLASS[lbl],   dtype=torch.long),
            torch.tensor(FAULT_FROM_CLASS[lbl],     dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
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


def mixup_batch(x, y, alpha=0.1, device="cpu"):
    if alpha <= 0:
        return x, y, y, 1.0
    lam  = float(np.random.beta(alpha, alpha))
    lam  = max(lam, 1.0 - lam)
    perm = torch.randperm(x.size(0), device=device)
    return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def make_warmup_scheduler(optimizer, warmup_epochs):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda ep: float(ep + 1) / float(warmup_epochs) if ep < warmup_epochs else 1.0,
    )


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, crit_main, crit_machine, crit_fault,
                device, mixup_alpha=0.1, hier_alpha=0.4):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        x, y_main, y_machine, y_fault = [t.to(device) for t in batch]

        x_mix, y_a, y_b, lam = mixup_batch(x, y_main, alpha=mixup_alpha, device=device)

        optimizer.zero_grad()
        out_main, out_machine, out_fault = model(x_mix)

        L_main    = mixup_criterion(crit_main, out_main, y_a, y_b, lam)
        L_machine = crit_machine(out_machine, y_machine)
        L_fault   = crit_fault(out_fault, y_fault)
        loss      = L_main + hier_alpha * L_machine + (1.0 - hier_alpha) * L_fault

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (out_main.argmax(1) == y_a).sum().item()
        total      += y_main.size(0)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, crit_main, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            x, y_main, y_machine, y_fault = [t.to(device) for t in batch]
            out_main, _, _ = model(x)
            total_loss += crit_main(out_main, y_main).item()
            preds = out_main.argmax(1)
            correct    += (preds == y_main).sum().item()
            total      += y_main.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_main.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, path):
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":    epoch,
        "val_loss": val_loss,
        "val_acc":  val_acc,
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
    plt.savefig(os.path.join(MODELS_DIR, "phase1_v4_confusion_matrix.png"), dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

all_paths, all_labels = scan_wav_files(ROOT_DIR)

splits    = load_clean_split(SPLIT_DIR)
train_idx = splits["train"]
val_idx   = splits["val"]
test_idx  = splits["test"]
print(f"Split — Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

preprocessor = AudioPreprocessor()
precompute_mel(all_paths, FEATS_DIR, preprocessor)

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

kw = dict(num_workers=NUM_WORKERS, pin_memory=True)
train_ds  = PrecomputedDataset(FEATS_DIR, all_labels, train_idx, augment=True)
val_ds    = PrecomputedDataset(FEATS_DIR, all_labels, val_idx,   augment=False)
test_ds   = PrecomputedDataset(FEATS_DIR, all_labels, test_idx,  augment=False)
tr_ldr    = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw)
vl_ldr    = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **kw)
te_ldr    = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, **kw)
print(f"\nTrain: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

model = MelCNNHier(num_classes=6).to(DEVICE)

crit_main    = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
crit_machine = nn.CrossEntropyLoss(weight=machine_w)
crit_fault   = BinaryFocalLoss(gamma=FOCAL_GAMMA, pos_weight=fault_pos_w)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
warmup_scheduler  = make_warmup_scheduler(optimizer, warmup_epochs=WARMUP_EPOCHS)
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=RLROP_FACTOR, patience=RLROP_PATIENCE, min_lr=1e-5
)

best_val_loss = float("inf")
best_val_acc  = 0.0
es_counter    = 0
history       = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
ckpt_path     = os.path.join(MODELS_DIR, "phase1_best.pth")

print(f"\n-- Training Phase 1 V4 --")
print(f"   Mixup a={MIXUP_ALPHA}  |  Focal g={FOCAL_GAMMA}  |  HierAlpha={HIER_ALPHA}")
print(f"   LabelSmooth={LABEL_SMOOTH}  |  ENS_beta={ENS_BETA}  |  Warmup={WARMUP_EPOCHS}ep")
print(f"   RLROP patience={RLROP_PATIENCE}  |  ES patience={ES_PATIENCE}")
print()

for epoch in range(1, EPOCHS + 1):
    current_lr = optimizer.param_groups[0]["lr"]

    tr_loss, tr_acc = train_epoch(
        model, tr_ldr, optimizer,
        crit_main, crit_machine, crit_fault,
        DEVICE, mixup_alpha=MIXUP_ALPHA, hier_alpha=HIER_ALPHA
    )
    vl_loss, vl_acc, _, _ = eval_epoch(model, vl_ldr, crit_main, DEVICE)

    if epoch <= WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        plateau_scheduler.step(vl_loss)

    for k, v in zip(["loss", "acc", "val_loss", "val_acc"],
                    [tr_loss, tr_acc, vl_loss, vl_acc]):
        history[k].append(v)

    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        best_val_acc  = vl_acc
        es_counter    = 0
        save_checkpoint(model, optimizer, epoch, vl_loss, vl_acc, ckpt_path)
        tag = "  <- saved"
    else:
        es_counter += 1
        tag = f"  (patience {es_counter}/{ES_PATIENCE})"

    print(
        f"Epoch {epoch:3d}/{EPOCHS}  lr={current_lr:.2e}  "
        f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
        f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{tag}"
    )

    if es_counter >= ES_PATIENCE:
        print(f"\n[early stop] Val loss did not improve for {ES_PATIENCE} epochs.")
        break

print(f"\nBest val loss: {best_val_loss:.4f}  (val acc: {best_val_acc:.4f})")

# Final evaluation on test set
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_, test_acc, test_preds, test_labels = eval_epoch(model, te_ldr, crit_main, DEVICE)
ms_per_sample = (time.time() - t0) / len(test_ds) * 1000

metrics = compute_metrics(test_preds, test_labels)
print(f"\n-- Test Results --")
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"Macro F1 : {metrics['macro_f1']:.4f}")
print("\nPer-class F1:")
for name, f1 in zip(CLASS_NAMES, metrics["per_class_f1"]):
    print(f"  {name}: {f1:.4f}")
print("\n", classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
print(f"Per-sample inference: {ms_per_sample:.3f} ms")

plot_confusion_matrix(test_preds, test_labels, CLASS_NAMES, "Phase 1 V4 — Confusion Matrix")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["loss"],     label="train"); ax1.plot(history["val_loss"], label="val")
ax1.set_title("Loss"); ax1.legend()
ax2.plot(history["acc"],      label="train"); ax2.plot(history["val_acc"],  label="val")
ax2.set_title("Accuracy"); ax2.legend()
plt.suptitle("Phase 1 V4 — Hierarchical Loss + Focal Loss")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "phase1_v4_curves.png"), dpi=150)
plt.show()

print(f"Checkpoint -> {ckpt_path}")
print("This file is required as input for train_phase2bV3.py and train_phase2bV4.py")
