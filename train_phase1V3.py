import os
import time
import random
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
import tqdm
from machine_listener.src.preprocess import AudioPreprocessor, PreprocessConfig, AugmentationConfig
from machine_listener.src.dataset import scan_wav_files, SPLIT_DIR
from machine_listener.src.features.mel_spectrogram import compute_mel_spectrogram
from machine_listener.src.models.cnn_baseline import MelCNNHier
from machine_listener.src.split_utils import load_clean_split
import machine_listener.src.train_utils as utils

ROOT_DIR   = "Students"
MODELS_DIR = "machine_listener/outputs/saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATS_DIR_MEL = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mel"))
os.makedirs(FEATS_DIR_MEL, exist_ok=True)

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 32
EPOCHS         = 20
NUM_WORKERS    = 4
LR             = 1e-3
WEIGHT_DECAY   = 5e-4
MIXUP_ALPHA    = 0.1
LABEL_SMOOTH   = 0.1
ENS_BETA       = 0.9999
HIER_ALPHA     = 0.4
FOCAL_GAMMA    = 2.0
WARMUP_EPOCHS  = 2
RLROP_PATIENCE = 2
RLROP_FACTOR   = 0.5
ES_PATIENCE    = 4

# machine_id: 0/1→0  2/3→1  4/5→2  |  fault: Normal→0  Abnormal→1
MACHINE_FROM_CLASS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
FAULT_FROM_CLASS   = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}

CLASS_NAMES   = [
    "Machine1_Normal", "Machine1_Abnormal",
    "Machine2_Normal", "Machine2_Abnormal",
    "Machine3_Normal", "Machine3_Abnormal",
]
MACHINE_NAMES = ["Machine1", "Machine2", "Machine3"]

print(f"Device: {DEVICE}")



def spec_augment(mel, freq_mask=27, time_mask=15, n_freq=2, n_time=2):
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


def _precompute_mel_one(args):
    idx, wav_path, feats_dir, preprocessor = args
    out = pathlib.Path(feats_dir) / f"{idx:06d}.npy"
    if out.exists():
        return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        np.save(out, compute_mel_spectrogram(w))
    except Exception:
        np.save(out, np.zeros((1, 128, 84), dtype=np.float32))


def precompute_mel(paths, feats_dir, preprocessor, n_workers=4):

    feats_dir = pathlib.Path(feats_dir)
    feats_dir.mkdir(parents=True, exist_ok=True)
    already = sum(1 for i in range(len(paths)) if (feats_dir / f"{i:06d}.npy").exists())
    if already == len(paths):
        print(f"All {len(paths)} mel-specs already cached  (skipping)")
        return
    print(f"Pre-computing {len(paths)} mel-specs with {n_workers} threads ...")
    from concurrent.futures import ThreadPoolExecutor
    args = [(i, p, str(feats_dir), preprocessor) for i, p in enumerate(paths)]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(tqdm.tqdm(ex.map(lambda a: _precompute_mel_one(a), args),
                       total=len(args), desc="mel"))
    print("Pre-computation done.")


def effective_num_weights(label_counts, beta=0.9999, num_classes=6):
    eff_num = (1.0 - np.power(beta, label_counts)) / (1.0 - beta)
    weights = 1.0 / eff_num
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def mixup_batch(x, y, alpha=0.1, device="cpu"):
    if alpha <= 0: return x, y, y, 1.0
    lam  = float(np.random.beta(alpha, alpha))
    lam  = max(lam, 1.0 - lam)
    B    = x.size(0)
    perm = torch.randperm(B, device=device)
    return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def make_warmup_scheduler(optimizer, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class BinaryFocalLoss(nn.Module):
    """FL(p_t) = -(1 - p_t)^γ * log(p_t) — down-weights easy samples."""
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        targets_f = targets.float().unsqueeze(1)
        bce   = F.binary_cross_entropy_with_logits(
            logits, targets_f, pos_weight=self.pos_weight, reduction="none")
        p_t   = torch.exp(-bce)
        focal = (1.0 - p_t) ** self.gamma * bce
        return focal.mean()


class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, feats_dir, labels, indices, augment=False):
        self.feats_dir = pathlib.Path(feats_dir)
        self.labels    = labels
        self.indices   = indices
        self.augment   = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri  = self.indices[idx]
        lbl = self.labels[ri]
        mel = torch.tensor(np.load(self.feats_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment:
            mel = spec_augment(mel)
        return (mel,
                torch.tensor(lbl,                           dtype=torch.long),
                torch.tensor(MACHINE_FROM_CLASS[lbl],       dtype=torch.long),
                torch.tensor(FAULT_FROM_CLASS[lbl],         dtype=torch.long))


def train_epoch_v3(model, loader, optimizer,
                   crit_main, crit_machine, crit_fault,
                   device, mixup_alpha=0.1, hier_alpha=0.4):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        mel, y_main, y_machine, y_fault = [t.to(device) for t in batch]
        mel_mix, y_a, y_b, lam = mixup_batch(mel, y_main, alpha=mixup_alpha, device=device)

        optimizer.zero_grad()
        out_main, out_machine, out_fault = model(mel_mix)

        L_main    = mixup_criterion(crit_main, out_main, y_a, y_b, lam)
        L_machine = crit_machine(out_machine, y_machine)
        L_fault   = crit_fault(out_fault, y_fault)
        loss = L_main + hier_alpha * L_machine + (1.0 - hier_alpha) * L_fault

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (out_main.argmax(1) == y_a).sum().item()
        total      += y_main.size(0)
    return total_loss / len(loader), correct / total


def eval_epoch_v3(model, loader, crit_main, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            mel, y_main, y_machine, y_fault = [t.to(device) for t in batch]
            out_main, _, _ = model(mel)
            loss  = crit_main(out_main, y_main)
            preds = out_main.argmax(1)
            total_loss += loss.item()
            correct    += (preds == y_main).sum().item()
            total      += y_main.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_main.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels




_infer_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=16000, default_duration_sec=2.75, trim_silence=True, normalize_mode="peak",
    augmentation=AugmentationConfig(enabled=False),
))

ALL_PATHS, ALL_LABELS = scan_wav_files(ROOT_DIR)
print(f"Found {len(ALL_PATHS)} files")

_splits = load_clean_split(SPLIT_DIR)


precompute_mel(ALL_PATHS, FEATS_DIR_MEL, _infer_prep, n_workers=NUM_WORKERS)



label_counts  = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
class_weights = effective_num_weights(label_counts, beta=ENS_BETA).to(DEVICE)

machine_counts = np.array([
    label_counts[0] + label_counts[1],
    label_counts[2] + label_counts[3],
    label_counts[4] + label_counts[5],
], dtype=np.float64)
machine_ens = (1.0 - np.power(ENS_BETA, machine_counts)) / (1.0 - ENS_BETA)
machine_w   = torch.tensor(1.0/machine_ens / (1.0/machine_ens).sum() * 3,
                            dtype=torch.float32).to(DEVICE)

fault_counts = np.array([
    sum(label_counts[i] for i in [0, 2, 4]),
    sum(label_counts[i] for i in [1, 3, 5]),
])
fault_pos_w  = torch.tensor([fault_counts[0] / (fault_counts[1] + 1e-6)],
                              dtype=torch.float32).to(DEVICE)

print("\n── Class counts in training split ──")
for i, (name, cnt, w) in enumerate(zip(CLASS_NAMES, label_counts, class_weights.cpu())):
    print(f"  [{i}] {name:<25s}  n={cnt:>5d}  ENS_weight={w:.4f}")

train_ds = PrecomputedDataset(FEATS_DIR_MEL, ALL_LABELS, _splits["train"], augment=True)
val_ds   = PrecomputedDataset(FEATS_DIR_MEL, ALL_LABELS, _splits["val"],   augment=False)
test_ds  = PrecomputedDataset(FEATS_DIR_MEL, ALL_LABELS, _splits["test"],  augment=False)

print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)


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
history       = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
ckpt_path     = os.path.join(MODELS_DIR, "phase1_v3_best.pth")

print("\n── Training Phase 1 V3 ────────────────────────────────────")
print(f"   Mixup α={MIXUP_ALPHA} | Focal γ={FOCAL_GAMMA} | HierAlpha={HIER_ALPHA}")
print(f"   LabelSmooth ε={LABEL_SMOOTH} | ENS β={ENS_BETA} | Warmup={WARMUP_EPOCHS}ep")
print(f"   RLROP patience={RLROP_PATIENCE} factor={RLROP_FACTOR} | ES patience={ES_PATIENCE}")
print()

for epoch in range(1, EPOCHS + 1):
    current_lr = optimizer.param_groups[0]["lr"]

    tr_loss, tr_acc = train_epoch_v3(
        model, train_loader, optimizer,
        crit_main, crit_machine, crit_fault,
        DEVICE, mixup_alpha=MIXUP_ALPHA, hier_alpha=HIER_ALPHA
    )
    vl_loss, vl_acc, _, _ = eval_epoch_v3(model, val_loader, crit_main, DEVICE)

    if epoch <= WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        plateau_scheduler.step(vl_loss)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)

    improved = vl_loss < best_val_loss
    tag = ""
    if improved:
        best_val_loss = vl_loss
        best_val_acc  = vl_acc
        es_counter    = 0
        torch.save({
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "val_loss": vl_loss, "val_acc": vl_acc,
        }, ckpt_path)
        tag = "  ← saved"
    else:
        es_counter += 1
        tag = f"  (patience {es_counter}/{ES_PATIENCE})"

    print(f"Epoch {epoch:3d}/{EPOCHS}  lr={current_lr:.2e}  "
          f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
          f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{tag}")

    if es_counter >= ES_PATIENCE:
        print(f"\n[early stop] Stopping after {ES_PATIENCE} epochs without val_loss improvement.")
        break

print(f"\nBest val loss: {best_val_loss:.4f}  (val acc: {best_val_acc:.4f})")


ckpt = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_, test_acc, test_preds, test_labels = eval_epoch_v3(model, test_loader, crit_main, DEVICE)
t_test = time.time() - t0
n_test = len(test_ds)
ms_per_sample = (t_test / n_test) * 1000

metrics = utils.compute_metrics(test_preds, test_labels)

print(f"\n── Test Results ─────────────────────────────────────────────")
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"Macro F1 : {metrics['macro_f1']:.4f}")
print(f"Epoch    : {ckpt['epoch']}")
print("\nPer-class F1:")
for cls_name, f1 in zip(CLASS_NAMES, metrics["per_class_f1"]):
    print(f"  {cls_name}: {f1:.4f}")
print()
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

print(f"\n── Inference Timing ─────────────────────────────────────────")
print(f"Test set size         : {n_test} samples")
print(f"Total inference time  : {t_test:.2f} s")
print(f"Per-sample time       : {ms_per_sample:.3f} ms  →  {1000/ms_per_sample:.0f} samples/sec")
print(f"Estimated   100 files : {ms_per_sample *   100 / 1000:.2f} s")
print(f"Estimated 1 000 files : {ms_per_sample *  1000 / 1000:.2f} s")
print(f"Estimated 10 000 files: {ms_per_sample * 10000 / 1000:.2f} s")

utils.plot_confusion_matrix(test_preds, test_labels, CLASS_NAMES)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="train"); ax1.plot(history["val_loss"], label="val")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
ax2.plot(history["train_acc"],  label="train"); ax2.plot(history["val_acc"],  label="val")
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
plt.suptitle("Phase 1 V3 — Hierarchical + Focal + RLROP")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "phase1_v3_curves.png"))
plt.show()

print(f"\nCheckpoint: {ckpt_path}")
print("Download phase1_v3_best.pth and use it as input for train_phase2bV2.py")
