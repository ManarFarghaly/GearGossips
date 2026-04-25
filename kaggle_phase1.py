"""
PHASE 1 — Mel-Spectrogram → 2D CNN Baseline                 
Paste this entire file into a Kaggle notebook cell.                                                                   
Before running:                                             
    1. Upload your 'Students' folder as a Kaggle dataset      
    2. Set ROOT_DIR below to the correct /kaggle/input/ path  
    3. GPU must be ON (Settings → Accelerator → GPU T4 x2)                                                              
Output: phase1_best.pth saved to /kaggle/working/           
        Download it and upload as a dataset for Phase 2     

"""


#  0. Install / imports 
import os, json, math, re, pathlib, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

# Config
ROOT_DIR   = "/kaggle/input/datasets/mostafaehab41/machine-fault-dataset"
MODELS_DIR = "/kaggle/working"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SR           = 16000
DURATION_SEC = 2.75
BATCH_SIZE   = 32
EPOCHS       = 20      # val_acc was 97.26% after epoch 1 — 20 is plenty, 50 would be ~41 hours
LR           = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4       # Kaggle T4×2 has exactly 4 vCPUs — more than 4 causes context-switch overhead
FEATS_DIR    = pathlib.Path("/kaggle/working/feats_mel")  # pre-computed mel-specs live here

CLASS_NAMES = [
    "Machine1_Normal", "Machine1_Abnormal",
    "Machine2_Normal", "Machine2_Abnormal",
    "Machine3_Normal", "Machine3_Abnormal",
]

# ── 2. PREPROCESSING (inline — no external imports needed) ───────────────────
from dataclasses import dataclass, field
from typing import Optional, Literal
import soundfile as sf
from scipy.signal import resample_poly

EPSILON = 1e-8

@dataclass
class AugmentationConfig:
    enabled: bool = True
    noise_prob: float = 0.35
    noise_snr_db_min: float = 15.0
    noise_snr_db_max: float = 35.0
    time_shift_prob: float = 0.30
    time_shift_max_sec: float = 0.20
    pitch_shift_prob: float = 0.20
    pitch_shift_min_semitones: float = -1.0
    pitch_shift_max_semitones: float = 1.0
    random_crop_train: bool = True

@dataclass
class PreprocessConfig:
    target_sr: int = 16000
    default_duration_sec: float = 2.75
    trim_silence: bool = True
    silence_threshold_ratio: float = 0.02
    trim_frame_ms: int = 20
    trim_hop_ms: int = 10
    min_retained_sec: float = 0.25
    denoise: bool = False
    normalize_mode: str = "peak"
    peak_target: float = 0.95
    rms_target: float = 0.10
    clip_value: float = 1.0
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

class AudioPreprocessor:
    def __init__(self, config=None):
        self.config = config or PreprocessConfig()

    def preprocess(self, audio_path, duration_sec=None, target_sr=None,
                   mode="inference", seed=None):
        cfg = self.config
        eff_sr  = int(target_sr  or cfg.target_sr)
        eff_dur = float(duration_sec or cfg.default_duration_sec)
        tgt_len = int(round(eff_sr * eff_dur))

        try:
            data, orig_sr = sf.read(str(audio_path), always_2d=False, dtype="float32")
        except Exception:
            return np.zeros(tgt_len, dtype=np.float32)

        w = np.asarray(data, dtype=np.float32)
        if w.ndim > 1:
            w = w.mean(axis=1)
        if orig_sr != eff_sr:
            d = math.gcd(orig_sr, eff_sr)
            w = resample_poly(w, eff_sr//d, orig_sr//d).astype(np.float32)
        if cfg.trim_silence:
            w = self._trim(w, eff_sr)
        w = self._normalize(w)

        rng = np.random.default_rng(seed) if mode == "train" else None
        if mode == "train":
            w = self._augment(w, eff_sr, rng)
        w = self._fix_length(w, tgt_len, mode == "train", rng)
        w = np.nan_to_num(w, 0.0)
        np.clip(w, -cfg.clip_value, cfg.clip_value, out=w)
        return w.astype(np.float32)

    def _trim(self, w, sr):
        cfg = self.config
        if w.size == 0: return w
        peak = np.abs(w).max()
        if peak <= EPSILON: return w
        thr = peak * cfg.silence_threshold_ratio
        fl = max(1, int(sr * cfg.trim_frame_ms / 1000))
        hl = max(1, int(sr * cfg.trim_hop_ms  / 1000))
        aw = np.abs(w)
        active = [s for s in range(0, w.size - fl + 1, hl) if aw[s:s+fl].max() >= thr]
        if not active: return w
        trimmed = w[active[0]:min(w.size, active[-1]+fl)]
        if trimmed.size < int(cfg.min_retained_sec * sr): return w
        return trimmed.astype(np.float32)

    def _normalize(self, w):
        mode = self.config.normalize_mode
        if mode == "peak":
            p = np.abs(w).max()
            if p > EPSILON: w = w * (self.config.peak_target / p)
        elif mode == "rms":
            r = np.sqrt(np.mean(w**2))
            if r > EPSILON: w = w * (self.config.rms_target / r)
        return w.astype(np.float32)

    def _augment(self, w, sr, rng):
        aug = self.config.augmentation
        if not aug.enabled or w.size == 0: return w
        if rng.random() < aug.noise_prob:
            sig_rms = np.sqrt(np.mean(w**2))
            if sig_rms > EPSILON:
                snr = rng.uniform(aug.noise_snr_db_min, aug.noise_snr_db_max)
                noise = rng.normal(0, 1, w.shape).astype(np.float32)
                n_rms = np.sqrt(np.mean(noise**2))
                if n_rms > EPSILON:
                    w = w + noise * (sig_rms / (10**(snr/20)) / (n_rms + EPSILON))
        if rng.random() < aug.time_shift_prob:
            ms = int(round(aug.time_shift_max_sec * sr))
            if ms > 0:
                w = np.roll(w, int(rng.integers(-ms, ms+1)))
        return w.astype(np.float32)

    def _fix_length(self, w, tgt, training, rng):
        cur = w.size
        if cur == tgt: return w
        aug = self.config.augmentation
        if cur > tgt:
            start = (int(rng.integers(0, cur-tgt+1))
                     if training and aug.random_crop_train and rng is not None
                     else (cur - tgt) // 2)
            return w[start:start+tgt]
        pad = tgt - cur
        lp = (int(rng.integers(0, pad+1))
              if training and aug.random_crop_train and rng is not None else 0)
        return np.pad(w, (lp, pad-lp), mode="constant")

# ── 3. FEATURE EXTRACTION ─────────────────────────────────────────────────────
def _minmax(S):
    lo, hi = S.min(), S.max()
    return np.zeros_like(S) if hi - lo < 1e-8 else (S - lo) / (hi - lo)

def compute_mel_spectrogram(waveform, sr=16000):
    """waveform (44000,) → np.ndarray (1, 128, 84)"""
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=128, n_fft=1024,
        hop_length=512, fmin=50, fmax=8000, center=False,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.expand_dims(_minmax(mel_db).astype(np.float32), 0)   # (1,128,84)

# ── SpecAugment — replaces audio-domain augmentation ──────────────────────────
# Applied directly on the mel tensor (no CPU librosa cost).
# Randomly masks horizontal bands (frequency) and vertical bands (time).
# Paper: "SpecAugment: A Simple Data Augmentation Method for ASR" (Park et al. 2019)
import random
def spec_augment(mel, freq_mask=30, time_mask=15, n_freq=2, n_time=2):
    """
    mel : torch.Tensor (1, 128, 84)
    Masks up to `freq_mask` frequency rows and `time_mask` time columns.
    Returns the augmented tensor (in-place safe via .clone()).
    """
    mel = mel.clone()
    _, F, T = mel.shape
    for _ in range(n_freq):                       # mask frequency bands
        f  = random.randint(0, freq_mask)
        f0 = random.randint(0, max(F - f, 1))
        mel[:, f0:f0+f, :] = 0.0
    for _ in range(n_time):                       # mask time bands
        t  = random.randint(0, time_mask)
        t0 = random.randint(0, max(T - t, 1))
        mel[:, :, t0:t0+t] = 0.0
    return mel

# ── Pre-computation helpers ────────────────────────────────────────────────────
# WHY PRE-COMPUTE:
#   Without pre-computation each __getitem__ call runs:
#     load .wav → resample → trim → normalize → librosa mel-spec  (~0.07s on CPU)
#   With 39,365 training samples × 0.07s = 46 min PER EPOCH.
#
#   Pre-computing runs that 0.07s ONCE per file (~15 min total for 56 k files)
#   then every __getitem__ is just:  np.load(path)  (~0.001s)
#   → epoch time drops from ~50 min to ~2-4 min.

def _precompute_one(args):
    """Worker function: preprocess one wav file and save mel-spec as .npy.
    Must be a top-level function (not a method) for multiprocessing to pickle it."""
    idx, wav_path, feats_dir, preprocessor = args
    out_path = pathlib.Path(feats_dir) / f"{idx:06d}.npy"
    if out_path.exists():
        return   # already done — safe to re-run
    try:
        waveform = preprocessor.preprocess(str(wav_path), mode="inference")
        mel      = compute_mel_spectrogram(waveform)
        np.save(out_path, mel)
    except Exception as e:
        # Save a silent placeholder so training doesn't crash on a bad file
        np.save(out_path, np.zeros((1, 128, 84), dtype=np.float32))

def precompute_all_mel(paths, feats_dir, preprocessor, n_workers=4):
    """
    Pre-compute mel-spectrograms for all wav files.
    Skips files already saved (safe to interrupt and resume).
    Uses multiprocessing so 4 CPU cores run in parallel (~15 min for 56k files).
    """
    import multiprocessing, tqdm as tqdm_module
    feats_dir = pathlib.Path(feats_dir)
    feats_dir.mkdir(parents=True, exist_ok=True)

    already = sum(1 for i in range(len(paths)) if (feats_dir/f"{i:06d}.npy").exists())
    if already == len(paths):
        print(f"All {len(paths)} mel-specs already cached in {feats_dir}  (skipping)")
        return
    print(f"Pre-computing mel-spectrograms for {len(paths)} files using {n_workers} workers...")
    print("This runs ONCE per session (~15 min). Training epochs will then take ~3 min each.")

    args = [(i, p, str(feats_dir), preprocessor) for i, p in enumerate(paths)]
    with multiprocessing.Pool(n_workers) as pool:
        list(tqdm_module.tqdm(
            pool.imap(_precompute_one, args, chunksize=64),
            total=len(paths), desc="mel-specs"
        ))
    print(f"Pre-computation done → {feats_dir}")

# ── 4. DATASET ────────────────────────────────────────────────────────────────
# Actual folder structure on Kaggle:
#   machine-fault-dataset/
#     machine1/Normal/*.wav    → label 0
#     machine1/Abnormal/*.wav  → label 1
#     machine2/Normal/*.wav    → label 2
#     machine2/Abnormal/*.wav  → label 3
#     machine3/Normal/*.wav    → label 4
#     machine3/Abnormal/*.wav  → label 5
#
# Key differences from the assumed structure:
#   • folder names are lowercase with no space: "machine1" not "Machine 1"
#   • wav files are 2 levels deep (machineX/State/file.wav), not 4
#   • no "Students" subfolder, no "machine_data" subfolder

class MachineDataset(Dataset):
    LABEL_MAP = {
        ("machine1","Normal"):0, ("machine1","Abnormal"):1,
        ("machine2","Normal"):2, ("machine2","Abnormal"):3,
        ("machine3","Normal"):4, ("machine3","Abnormal"):5,
    }

    def __init__(self, root_dir, preprocessor, feature_fn, split, augment=False,
                 split_save_dir=MODELS_DIR):
        self.root_dir      = pathlib.Path(root_dir)
        self.preprocessor  = preprocessor
        self.feature_fn    = feature_fn
        self.split         = split
        self.augment       = augment
        # split_indices.json MUST go to /kaggle/working — /kaggle/input is read-only
        self.split_save_dir = pathlib.Path(split_save_dir)
        self.paths, self.labels = self._scan()
        self.indices             = self._split()

    def _scan(self):
        paths, labels = [], []
        for f in self.root_dir.rglob("*.wav"):
            state   = f.parent.name        # "Normal" or "Abnormal"
            machine = f.parent.parent.name # "machine1", "machine2", "machine3"
            lbl = self.LABEL_MAP.get((machine, state))
            if lbl is not None:
                paths.append(f); labels.append(lbl)
        if not paths:
            raise RuntimeError(
                f"No labelled .wav files found under {self.root_dir}\n"
                f"Expected structure: machineX/Normal/*.wav and machineX/Abnormal/*.wav\n"
                f"Found top-level folders: {[p.name for p in self.root_dir.iterdir() if p.is_dir()]}"
            )
        print(f"Found {len(paths)} files across {len(set(labels))} classes")
        return paths, labels

    def _split(self):
        split_file = self.split_save_dir / "split_indices.json"
        if split_file.exists():
            return json.load(open(split_file))[self.split]
        idx = list(range(len(self.paths)))
        tr, tmp, _, tl = train_test_split(idx, self.labels, test_size=0.30,
                                          stratify=self.labels, random_state=42)
        va, te = train_test_split(tmp, test_size=0.50, stratify=tl, random_state=42)
        json.dump({"train":tr,"val":va,"test":te}, open(split_file,"w"))
        print(f"Split created → {split_file}  (train={len(tr)}, val={len(va)}, test={len(te)})")
        return {"train":tr,"val":va,"test":te}[self.split]

    def __len__(self):  return len(self.indices)

    def __getitem__(self, idx):
        ri   = self.indices[idx]
        path = self.paths[ri]
        lbl  = self.labels[ri]
        mode = "train" if (self.split=="train" and self.augment) else "inference"
        waveform = self.preprocessor.preprocess(path, mode=mode)   # (44000,)
        feat     = self.feature_fn(waveform)                        # (1,128,84)
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(lbl, dtype=torch.long)

# ── 4b. PRECOMPUTED DATASET — fast replacement for MachineDataset ─────────────
# Used after precompute_all_mel() has saved .npy files to FEATS_DIR.
# __getitem__ is just: np.load(path)  →  ~0.001s  (vs ~0.07s before)
# SpecAugment replaces audio-domain augmentation (no librosa cost at training time).

class PrecomputedDataset(Dataset):
    def __init__(self, feats_dir, labels, indices, augment=False):
        self.feats_dir = pathlib.Path(feats_dir)
        self.labels    = labels     # full label list (len = total files)
        self.indices   = indices    # subset indices for this split
        self.augment   = augment    # if True, apply SpecAugment

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri  = self.indices[idx]
        mel = np.load(self.feats_dir / f"{ri:06d}.npy")          # (1,128,84) float32
        mel_t = torch.tensor(mel, dtype=torch.float32)
        if self.augment:
            mel_t = spec_augment(mel_t)                           # fast tensor masking
        return mel_t, torch.tensor(self.labels[ri], dtype=torch.long)

# ── 5. MODEL ──────────────────────────────────────────────────────────────────
class MelCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1,32,3,padding=1),  nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)))
        self.fc1     = nn.Linear(256*4*4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, num_classes)

    def extract_features(self, x):
        """Returns (B,256) — used by Phase 2 to load Phase 1 weights."""
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        x = torch.flatten(x,1)
        return self.dropout(F.relu(self.fc1(x)))   # (B,256)

    def forward(self, x):
        return self.fc2(self.extract_features(x))  # (B,6)

# ── 6. TRAINING UTILITIES ─────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / len(loader), correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            preds = out.argmax(1)
            correct    += (preds == y).sum().item()
            total      += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels

def save_checkpoint(model, optimizer, epoch, val_acc, path):
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":    epoch,
        "val_acc":  val_acc,
    }, path)

def compute_metrics(preds, labels):
    preds, labels = np.array(preds), np.array(labels)
    return {
        "accuracy":    (preds == labels).mean(),
        "macro_f1":    f1_score(labels, preds, average="macro"),
        "per_class_f1":f1_score(labels, preds, average=None),
    }

def plot_confusion_matrix(preds, labels, class_names):
    cm = confusion_matrix(labels, preds)           # (y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout(); plt.show()

# ── 7. MAIN — pre-compute features, then train ────────────────────────────────

# Step A: scan files and create the split (same logic as before, but we need
#         the paths + labels BEFORE building the dataset so we can precompute.
_scan_ds = MachineDataset(ROOT_DIR,
                          AudioPreprocessor(PreprocessConfig(augmentation=AugmentationConfig(enabled=False))),
                          compute_mel_spectrogram, "train")   # split="train" triggers split creation
ALL_PATHS  = _scan_ds.paths   # full list of all wav paths (56 k)
ALL_LABELS = _scan_ds.labels  # matching labels

# Step B: pre-compute all mel-spectrograms once (~15 min, then cached)
# Uses inference-mode preprocessor (no random augmentation at save time).
# SpecAugment is applied at training time instead (free, on the tensor).
_infer_preprocessor = AudioPreprocessor(PreprocessConfig(
    target_sr=SR, default_duration_sec=DURATION_SEC,
    trim_silence=True, normalize_mode="peak",
    augmentation=AugmentationConfig(enabled=False),   # NO augmentation when saving
))
precompute_all_mel(ALL_PATHS, FEATS_DIR, _infer_preprocessor, n_workers=NUM_WORKERS)

# Step C: load the split indices that were just saved (or already existed)
_split_file = pathlib.Path(MODELS_DIR) / "split_indices.json"
_splits     = json.load(open(_split_file))

train_ds = PrecomputedDataset(FEATS_DIR, ALL_LABELS, _splits["train"], augment=True)
val_ds   = PrecomputedDataset(FEATS_DIR, ALL_LABELS, _splits["val"],   augment=False)
test_ds  = PrecomputedDataset(FEATS_DIR, ALL_LABELS, _splits["test"],  augment=False)

print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

# num_workers=4  : matches Kaggle's 4 vCPUs exactly
# persistent_workers=True : workers stay alive between batches (avoids fork overhead)
# prefetch_factor=2 : each worker pre-loads 2 batches ahead so GPU never waits
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)

# Compute class weights to handle any class imbalance
label_counts  = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
class_weights = torch.tensor(1.0 / (label_counts + 1), dtype=torch.float32).to(DEVICE)

model     = MelCNN(num_classes=6).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc  = 0.0
train_history = {"loss":[], "acc":[], "val_loss":[], "val_acc":[]}

print("\n── Training Phase 1 ──────────────────────────────────────")
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc     = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader, criterion, DEVICE)
    scheduler.step()

    train_history["loss"].append(tr_loss)
    train_history["acc"].append(tr_acc)
    train_history["val_loss"].append(vl_loss)
    train_history["val_acc"].append(vl_acc)

    print(f"Epoch {epoch:3d}/{EPOCHS}  "
          f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
          f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}", end="")

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        ckpt_path = os.path.join(MODELS_DIR, "phase1_best.pth")
        save_checkpoint(model, optimizer, epoch, vl_acc, ckpt_path)
        print("  ← saved", end="")
    print()

print(f"\nBest val accuracy: {best_val_acc:.4f}")

# ── 8. FINAL EVALUATION on test set ──────────────────────────────────────────
ckpt      = torch.load(os.path.join(MODELS_DIR, "phase1_best.pth"), map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

_, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, DEVICE)
metrics = compute_metrics(test_preds, test_labels)

print(f"\n── Test Results ──────────────────────────────────────────")
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"Macro F1 : {metrics['macro_f1']:.4f}")
print("\nPer-class F1:")
for i, (name, f1) in enumerate(zip(CLASS_NAMES, metrics["per_class_f1"])):
    print(f"  {name}: {f1:.4f}")
print("\n", classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

plot_confusion_matrix(test_preds, test_labels, CLASS_NAMES)

# ── 9. TRAINING CURVES ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_history["loss"],     label="train"); ax1.plot(train_history["val_loss"], label="val")
ax1.set_title("Loss"); ax1.legend()
ax2.plot(train_history["acc"],      label="train"); ax2.plot(train_history["val_acc"],  label="val")
ax2.set_title("Accuracy"); ax2.legend()
plt.suptitle("Phase 1 — Mel-Spectrogram CNN")
plt.tight_layout(); plt.show()

print(f"\nModel saved to: {ckpt_path}")
print("Download phase1_best.pth and upload it as a Kaggle dataset for Phase 2.")