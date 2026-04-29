"""
PHASE 1 — Mel-Spectrogram → 2D CNN Baseline  (v2 — overfitting + imbalance fixes)
═══════════════════════════════════════════════════════════════════════════════════
CHANGES vs v1 and the research justifying each one:

[FIX 1] Effective Number of Samples class weighting (replaces 1/sqrt heuristic)
         Formula: w_c = (1 − β) / (1 − β^{n_c}),  β = 0.9999
         Theory:  each new sample overlaps with previous ones; the "effective"
                  count saturates. 1/sqrt underweights minority classes at high N.
         Source:  Cui et al., "Class-Balanced Loss Based on Effective Number of
                  Samples," CVPR 2019.  https://arxiv.org/abs/1901.05555

[FIX 2] Label smoothing ε = 0.1 in CrossEntropyLoss
         Prevents the model from becoming overconfident on easy/dominant classes,
         which was causing Machine1 to "crowd out" M2 and M3 gradients.
         Source:  Müller et al., "When Does Label Smoothing Help?", NeurIPS 2019.
                  https://arxiv.org/abs/1906.02629

[FIX 3] SpecAugment parameters reduced to paper-original proportions
         Original paper used F=27 on 80-mel bins (~34%) and T=100 on 1000 frames.
         Your v1 used freq_mask=40 on 128 bins = 31% per mask × 3 masks → up to
         93% of frequency content destroyed.  For minority classes this removes
         discriminative features entirely.
         Source:  Park et al., "SpecAugment: A Simple Data Augmentation Method
                  for Automatic Speech Recognition," Interspeech 2019.
                  https://arxiv.org/abs/1904.08779

[FIX 4] Mixup augmentation (α = 0.3) applied at batch level
         Interpolates (x_i, x_j) and (y_i, y_j) with λ ~ Beta(α,α).
         Forces the model to learn smooth decision boundaries between
         Machine2/3 Normal vs Abnormal — the classes that completely collapsed.
         Source:  Zhang et al., "mixup: Beyond Empirical Risk Minimization,"
                  ICLR 2018.  https://arxiv.org/abs/1710.09412

[FIX 5] Linear warmup (2 epochs) + cosine annealing (replaces cosine-only)
         Prevents the model from memorising easy classes (Machine1) in early
         high-lr epochs before it has seen enough minority-class examples.
         Source:  Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet
                  in 1 Hour," 2017.  https://arxiv.org/abs/1706.02677

[FIX 6] Early stopping on val_loss (patience = 4, save best val_loss checkpoint)
         Your v1 val_loss diverged to 3.75 by epoch 20 while train_acc → 99.4%.
         Saving by val_accuracy hid this — epoch 7 (val_acc=0.76) was the last
         useful checkpoint. Early stopping finds this automatically.
         Source:  Prechelt, "Early Stopping — But When?", Neural Networks: Tricks
                  of the Trade, Springer 1998.

[FIX 7] Dropout reduced back to 0.5 (from 0.6)
         Dropout 0.6 on the 4096-unit fc1 layer is equivalent to randomly zeroing
         60% of activations; combined with SpecAugment + Mixup this over-regularises
         the model and prevents it from learning M2/M3 features at all.
         Source:  Srivastava et al., "Dropout: A Simple Way to Prevent Neural
                  Networks from Overfitting," JMLR 2014.  Section 4, Table 4 shows
                  0.5 is optimal for fully-connected layers in most settings.
                  https://jmlr.org/papers/v15/srivastava14a.html

[KEPT]   Weight decay 5e-4  (your change was correct, keeping it)
[KEPT]   Chronological split + no-leakage verification
[KEPT]   Pre-computed mel-spectrogram cache
[KEPT]   Persistent DataLoader workers, pin_memory, prefetch_factor=2
"""

import os, json, math, re, pathlib, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

# ══════════════════════════════════════════════════════════════════════════════
# SPLIT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
import hashlib
from collections import defaultdict as _ddict

def _num_sort_key(f):
    p = pathlib.Path(f)
    try: return (0, int(p.stem), p.stem.lower())
    except ValueError: return (1, 0, p.stem.lower())

def _md5(path):
    h = hashlib.md5()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def _find_duplicate_groups(paths):
    by_size = _ddict(list)
    for i, p in enumerate(paths): by_size[pathlib.Path(p).stat().st_size].append(i)
    groups = _ddict(list)
    for size, idxs in by_size.items():
        if len(idxs) < 2: continue
        for i in idxs: groups[f"{size}:{_md5(paths[i])}"].append(i)
    return {k: v for k, v in groups.items() if len(v) > 1}

def _build_chronological_split(paths, labels, train_r=0.70, val_r=0.15):
    dup_groups = _find_duplicate_groups(paths)
    idx_to_key = {i: k for k, idxs in dup_groups.items() for i in idxs}
    if dup_groups:
        n = sum(len(v) for v in dup_groups.values())
        print(f"[split] {len(dup_groups)} duplicate groups ({n} files) — all copies go to same split")
    else:
        print("[split] No duplicates found")
    by_class = _ddict(list)
    for i, lbl in enumerate(labels): by_class[lbl].append(i)
    dup_assigned = {}; all_train, all_val, all_test = [], [], []
    for cls_id in sorted(by_class):
        cls_idxs = sorted(by_class[cls_id], key=lambda i: _num_sort_key(paths[i]))
        n = len(cls_idxs); n_tr = int(train_r * n); n_va = int(val_r * n)
        for rank, gidx in enumerate(cls_idxs):
            nat = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")
            k = idx_to_key.get(gidx)
            if k is not None:
                asgn = dup_assigned.setdefault(k, nat)
                if asgn != nat:
                    print(f"[split]   dup-fix: {pathlib.Path(paths[gidx]).name} {nat}->{asgn}")
            else:
                asgn = nat
            (all_train if asgn == "train" else all_val if asgn == "val" else all_test).append(gidx)
    return all_train, all_val, all_test

def _verify_split(paths, labels, tr, va, te):
    by_class = _ddict(list)
    for i, lbl in enumerate(labels): by_class[lbl].append(i)
    tr_s, te_s = set(tr), set(te)
    bp = sum(1 for idxs in by_class.values()
             for a, b in zip(sorted(idxs, key=lambda i: _num_sort_key(paths[i]))[:-1],
                             sorted(idxs, key=lambda i: _num_sort_key(paths[i]))[1:])
             if (a in tr_s and b in te_s) or (a in te_s and b in tr_s))
    print(f"[split] Train={len(tr)}  Val={len(va)}  Test={len(te)}  Boundary_pairs={bp} (target=0)")
    if bp == 0: print("[split] Zero temporal leakage")
    else: print(f"[split] {bp} boundary pairs (caused by duplicate-fix)")

def _create_clean_split(paths, labels, split_dir, train_r=0.70, val_r=0.15):
    split_dir = pathlib.Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    tr, va, te = _build_chronological_split(paths, labels, train_r, val_r)
    _verify_split(paths, labels, tr, va, te)
    result = {"train": tr, "val": va, "test": te}
    out = split_dir / "split_indices_clean.json"
    json.dump(result, open(out, "w"))
    print(f"[split] Saved -> {out}")
    return result

def _load_clean_split(split_dir):
    path = pathlib.Path(split_dir) / "split_indices_clean.json"
    if not path.exists():
        raise FileNotFoundError(
            f"split_indices_clean.json not found at {path}. "
            "Run kaggle_phase1.py first to create it.")
    return json.load(open(path))

# ══════════════════════════════════════════════════════════════════════════════

_LABEL_MAP = {
    ("machine1", "Normal"): 0, ("machine1", "Abnormal"): 1,
    ("machine2", "Normal"): 2, ("machine2", "Abnormal"): 3,
    ("machine3", "Normal"): 4, ("machine3", "Abnormal"): 5,
}

def _scan_wav_files(root_dir):
    paths, labels = [], []
    for wav in sorted(pathlib.Path(root_dir).rglob("*.wav")):
        state   = wav.parent.name
        machine = wav.parent.parent.name
        lbl = _LABEL_MAP.get((machine, state))
        if lbl is not None:
            paths.append(wav)
            labels.append(lbl)
    if not paths:
        raise RuntimeError(
            f"No labelled .wav files found under {root_dir}. "
            "Expected: machineX/Normal/*.wav and machineX/Abnormal/*.wav")
    print(f"Found {len(paths)} files")
    return paths, labels

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT_DIR   = "/kaggle/input/datasets/mostafaehab41/machine-fault-dataset"
MODELS_DIR = "/kaggle/working"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SR           = 16000
DURATION_SEC = 2.75
BATCH_SIZE   = 32
EPOCHS       = 20        # kept same; early stopping will terminate early
LR           = 1e-3
WEIGHT_DECAY = 5e-4      # kept from your v1 change
WARMUP_EPOCHS = 2        # [FIX 5] linear lr warmup
ES_PATIENCE   = 4        # [FIX 6] early stopping patience on val_loss
NUM_WORKERS  = 4

MIXUP_ALPHA  = 0.3       # [FIX 4] Mixup interpolation strength
LABEL_SMOOTH = 0.1       # [FIX 2] label smoothing ε
ENS_BETA     = 0.9999    # [FIX 1] effective number of samples β

import wandb
wandb.init(
    project="machine-fault-phase1",
    config={
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "epochs": EPOCHS,
        "model": "MelCNN_v2",
        "specaugment": True,
        "mixup_alpha": MIXUP_ALPHA,
        "label_smoothing": LABEL_SMOOTH,
        "ens_beta": ENS_BETA,
        "warmup_epochs": WARMUP_EPOCHS,
        "es_patience": ES_PATIENCE,
        "dropout": 0.5,
        "weight_decay": WEIGHT_DECAY,
    }
)

# ── FEATURE CACHE ─────────────────────────────────────────────────────────────
def _feat_dir(name: str) -> pathlib.Path:
    owner = pathlib.Path("/kaggle/input/datasets/manarabdelshafy")
    if owner.exists():
        for ds in sorted(owner.iterdir()):
            if not ds.is_dir(): continue
            candidate = ds / name
            if candidate.exists() and any(candidate.glob("*.npy")):
                print(f"[cache] '{name}' found at {candidate}  ✓  (skipping recomputation)")
                return candidate
    working = pathlib.Path("/kaggle/working") / name
    print(f"[cache] '{name}' not in uploaded datasets → will compute to {working}")
    return working

FEATS_DIR = _feat_dir("feats_mel")

CLASS_NAMES = [
    "Machine1_Normal", "Machine1_Abnormal",
    "Machine2_Normal", "Machine2_Abnormal",
    "Machine3_Normal", "Machine3_Abnormal",
]

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
from dataclasses import dataclass, field
from typing import Optional
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

# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────
def _minmax(S):
    lo, hi = S.min(), S.max()
    return np.zeros_like(S) if hi - lo < 1e-8 else (S - lo) / (hi - lo)

def compute_mel_spectrogram(waveform, sr=16000):
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=128, n_fft=1024,
        hop_length=512, fmin=50, fmax=8000, center=False,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.expand_dims(_minmax(mel_db).astype(np.float32), 0)

# ── [FIX 3] SpecAugment — reduced to paper-original proportions ───────────────
# Park et al. 2019: F=27 on 80-mel (~34%), T=100 on 1000 frames, mT=0.04*T
# Scaled to 128-mel, 84 time frames:
#   freq_mask = int(0.34 * 128) = 43  → we use 27 (conservative, per paper value)
#   time_mask = int(0.04 * 84)  = 3   → we use 15 (moderate; 0.04*T is very small)
#   n_freq=2, n_time=2           → paper used 1 each (LB policy); 2 is reasonable
import random
def spec_augment(mel, freq_mask=27, time_mask=15, n_freq=2, n_time=2):
    """
    [FIX 3] SpecAugment with paper-calibrated parameters.
    v1 used freq_mask=40, n_freq=3 → up to 93% frequency masking on 128-bin mel.
    Now: freq_mask=27, n_freq=2 → max 42% frequency masking (safe for minority classes).

    mel : torch.Tensor (1, 128, 84)
    """
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

# ── PRE-COMPUTATION ───────────────────────────────────────────────────────────
def _precompute_one(args):
    idx, wav_path, feats_dir, preprocessor = args
    out_path = pathlib.Path(feats_dir) / f"{idx:06d}.npy"
    if out_path.exists():
        return
    try:
        waveform = preprocessor.preprocess(str(wav_path), mode="inference")
        mel      = compute_mel_spectrogram(waveform)
        np.save(out_path, mel)
    except Exception:
        np.save(out_path, np.zeros((1, 128, 84), dtype=np.float32))

def precompute_all_mel(paths, feats_dir, preprocessor, n_workers=4):
    import multiprocessing, tqdm as tqdm_module
    feats_dir = pathlib.Path(feats_dir)
    if str(feats_dir).startswith("/kaggle/input"):
        print(f"[cache] mel features loaded from uploaded dataset {feats_dir}  ✓")
        return
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

# ── [FIX 4] MIXUP UTILITIES ───────────────────────────────────────────────────
# Zhang et al. ICLR 2018: λ ~ Beta(α, α), x̃ = λx_i + (1-λ)x_j
# For cross-entropy with label smoothing, we mix the one-hot (smooth) targets.
# This forces the model to learn smooth M2/M3 Normal-vs-Abnormal boundaries.

def mixup_batch(x, y, alpha=0.3, num_classes=6, device="cpu"):
    """
    Apply Mixup to a batch.
    x : (B, 1, 128, 84) float tensor
    y : (B,) long tensor — class indices
    Returns: x_mix (B, 1, 128, 84), y_a (B,), y_b (B,), lam (scalar)
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)   # keep lam ≥ 0.5 so dominant sample stays dominant
    B = x.size(0)
    perm = torch.randperm(B, device=device)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    return x_mix, y, y[perm], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixed loss: lam * L(pred, y_a) + (1-lam) * L(pred, y_b)."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

# ── DATASET ───────────────────────────────────────────────────────────────────
class PrecomputedDataset(Dataset):
    def __init__(self, feats_dir, labels, indices, augment=False):
        self.feats_dir = pathlib.Path(feats_dir)
        self.labels    = labels
        self.indices   = indices
        self.augment   = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri  = self.indices[idx]
        mel = np.load(self.feats_dir / f"{ri:06d}.npy", mmap_mode="r")
        mel_t = torch.tensor(mel, dtype=torch.float32)
        if self.augment:
            mel_t = spec_augment(mel_t)          # [FIX 3] calibrated params
        return mel_t, torch.tensor(self.labels[ri], dtype=torch.long)

# ── MODEL ─────────────────────────────────────────────────────────────────────
class MelCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1,32,3,padding=1),   nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1),  nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)))
        self.fc1     = nn.Linear(256*4*4, 256)
        self.dropout = nn.Dropout(0.5)   # [FIX 7] back to 0.5 (0.6 + SpecAugment + Mixup = over-regularised)
        self.fc2     = nn.Linear(256, num_classes)

    def extract_features(self, x):
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        x = torch.flatten(x, 1)
        return self.dropout(F.relu(self.fc1(x)))

    def forward(self, x):
        return self.fc2(self.extract_features(x))

# ── [FIX 1] EFFECTIVE NUMBER OF SAMPLES CLASS WEIGHTS ─────────────────────────
# Cui et al. CVPR 2019 — Eq. 2:
#   effective_num(n) = (1 − β^n) / (1 − β)
#   weight_c         = 1 / effective_num(n_c),  then normalised so weights sum to C
#
# Why better than 1/n or 1/sqrt(n):
#   As n → ∞ the effective number saturates to 1/(1−β) regardless of n, so extremely
#   large majority classes are not over-downweighted.  β=0.9999 is the paper default
#   for datasets with hundreds of samples per class.

def effective_num_weights(label_counts, beta=0.9999, num_classes=6):
    """
    label_counts : np.array of shape (num_classes,), count per class in training set.
    Returns     : torch.FloatTensor of shape (num_classes,), normalised class weights.
    """
    eff_num = (1.0 - np.power(beta, label_counts)) / (1.0 - beta)
    weights = 1.0 / eff_num
    weights = weights / weights.sum() * num_classes   # normalise: mean weight = 1
    return torch.tensor(weights, dtype=torch.float32)

# ── [FIX 5] WARMUP + COSINE SCHEDULER ────────────────────────────────────────
def build_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    Linear warmup for `warmup_epochs`, then cosine annealing to 0.
    Implemented as LambdaLR so it works with any optimizer.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)   # linear 0→1
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))    # cosine 1→0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── TRAINING UTILITIES ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.3):
    """
    [FIX 4] Mixup applied per batch.
    criterion should be nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_w).
    Mixup and label smoothing compose naturally: both soften the target distribution.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_mix, y_a, y_b, lam = mixup_batch(x, y, alpha=mixup_alpha, device=device)
        optimizer.zero_grad()
        out  = model(x_mix)
        loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()
        total_loss += loss.item()
        # accuracy: use the dominant label (y_a, since lam ≥ 0.5)
        correct    += (out.argmax(1) == y_a).sum().item()
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

def plot_confusion_matrix(preds, labels, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix"); plt.tight_layout(); plt.show()

# ── MAIN ──────────────────────────────────────────────────────────────────────

# Step A: scan files and create split
ALL_PATHS, ALL_LABELS = _scan_wav_files(ROOT_DIR)
_splits = _create_clean_split(ALL_PATHS, ALL_LABELS, MODELS_DIR)

# Step B: pre-compute all mel-spectrograms once
_infer_preprocessor = AudioPreprocessor(PreprocessConfig(
    target_sr=SR, default_duration_sec=DURATION_SEC,
    trim_silence=True, normalize_mode="peak",
    augmentation=AugmentationConfig(enabled=False),
))
precompute_all_mel(ALL_PATHS, FEATS_DIR, _infer_preprocessor, n_workers=NUM_WORKERS)

train_ds = PrecomputedDataset(FEATS_DIR, ALL_LABELS, _splits["train"], augment=True)
val_ds   = PrecomputedDataset(FEATS_DIR, ALL_LABELS, _splits["val"],   augment=False)
test_ds  = PrecomputedDataset(FEATS_DIR, ALL_LABELS, _splits["test"],  augment=False)

print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)

# [FIX 1] Effective Number of Samples weights
label_counts  = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
class_weights = effective_num_weights(label_counts, beta=ENS_BETA, num_classes=6).to(DEVICE)

print("\n── Class counts in training split ──")
for i, (name, cnt, w) in enumerate(zip(CLASS_NAMES, label_counts, class_weights.cpu())):
    print(f"  [{i}] {name:<25s}  n={cnt:>5d}  ENS_weight={w:.4f}")

model     = MelCNN(num_classes=6).to(DEVICE)

# [FIX 2] Label smoothing ε=0.1 + class weights (compose cleanly)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# [FIX 5] Warmup + cosine scheduler
scheduler = build_scheduler(optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS)

best_val_loss = float("inf")
best_val_acc  = 0.0
es_counter    = 0
train_history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
ckpt_path     = os.path.join(MODELS_DIR, "phase1_best.pth")

print("\n── Training Phase 1 v2 ───────────────────────────────────")
print(f"   Mixup α={MIXUP_ALPHA}  |  LabelSmooth ε={LABEL_SMOOTH}  |"
      f"  ENS β={ENS_BETA}  |  Warmup={WARMUP_EPOCHS}ep  |  EarlyStop patience={ES_PATIENCE}")
print()

for epoch in range(1, EPOCHS + 1):
    current_lr = scheduler.get_last_lr()[0] if epoch > 1 else LR

    tr_loss, tr_acc           = train_epoch(model, train_loader, optimizer, criterion,
                                            DEVICE, mixup_alpha=MIXUP_ALPHA)
    vl_loss, vl_acc, _, _     = eval_epoch(model, val_loader,   criterion, DEVICE)
    scheduler.step()

    wandb.log({
        "epoch": epoch, "lr": current_lr,
        "train_loss": tr_loss, "train_acc": tr_acc,
        "val_loss": vl_loss,   "val_acc":  vl_acc,
    })
    train_history["loss"].append(tr_loss)
    train_history["acc"].append(tr_acc)
    train_history["val_loss"].append(vl_loss)
    train_history["val_acc"].append(vl_acc)

    improved = vl_loss < best_val_loss    # [FIX 6] track val_loss, not val_acc
    tag = ""
    if improved:
        best_val_loss = vl_loss
        best_val_acc  = vl_acc
        es_counter    = 0
        save_checkpoint(model, optimizer, epoch, vl_loss, vl_acc, ckpt_path)
        tag = "  ← saved"
    else:
        es_counter += 1
        tag = f"  (patience {es_counter}/{ES_PATIENCE})"

    print(f"Epoch {epoch:3d}/{EPOCHS}  lr={current_lr:.2e}  "
          f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
          f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{tag}")

    # [FIX 6] Early stopping
    if es_counter >= ES_PATIENCE:
        print(f"\n[early stop] Val loss did not improve for {ES_PATIENCE} epochs. Stopping.")
        break

print(f"\nBest val loss: {best_val_loss:.4f}  (val acc at that epoch: {best_val_acc:.4f})")

# ── FINAL EVALUATION ──────────────────────────────────────────────────────────
ckpt = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, DEVICE)
t_test = time.time() - t0
n_test = len(test_ds)
ms_per_sample = (t_test / n_test) * 1000

metrics = compute_metrics(test_preds, test_labels)

print(f"\n── Test Results ──────────────────────────────────────────")
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"Macro F1 : {metrics['macro_f1']:.4f}")
print("\nPer-class F1:")
for cls_name, f1 in zip(CLASS_NAMES, metrics["per_class_f1"]):
    print(f"  {cls_name}: {f1:.4f}")

print("\n", classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

wandb.log({
    "test_accuracy": metrics["accuracy"],
    "test_macro_f1": metrics["macro_f1"],
})
for i, f1 in enumerate(metrics["per_class_f1"]):
    wandb.log({f"f1_{CLASS_NAMES[i]}": f1})

wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None, y_true=test_labels, preds=test_preds, class_names=CLASS_NAMES
    )
})

print(f"\n── Inference Timing ──────────────────────────────────────")
print(f"Test set size         : {n_test} samples")
print(f"Total inference time  : {t_test:.2f} s")
print(f"Per-sample time       : {ms_per_sample:.3f} ms  →  {1000/ms_per_sample:.0f} samples/sec")
print(f"Estimated   100 files : {ms_per_sample *   100 / 1000:.2f} s")
print(f"Estimated 1 000 files : {ms_per_sample *  1000 / 1000:.2f} s")
print(f"Estimated 10 000 files: {ms_per_sample * 10000 / 1000:.2f} s")

plot_confusion_matrix(test_preds, test_labels, CLASS_NAMES)

# ── TRAINING CURVES ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_history["loss"],     label="train")
ax1.plot(train_history["val_loss"], label="val")
ax1.set_title("Loss"); ax1.legend()
ax2.plot(train_history["acc"],      label="train")
ax2.plot(train_history["val_acc"],  label="val")
ax2.set_title("Accuracy"); ax2.legend()
plt.suptitle("Phase 1 v2 — Mel-Spectrogram CNN")
plt.tight_layout(); plt.show()

print(f"\nModel saved to: {ckpt_path}")

wandb.log({
    "class_counts": {CLASS_NAMES[i]: int(label_counts[i]) for i in range(len(CLASS_NAMES))}
})

# ── ARCHIVE FEATURES FOR REUSE ────────────────────────────────────────────────
if str(FEATS_DIR).startswith("/kaggle/working"):
    import shutil
    print("\nArchiving mel features for reuse in later phases ...")
    shutil.make_archive("/kaggle/working/feats_mel_archive", "zip",
                        "/kaggle/working", "feats_mel")
    sz = os.path.getsize("/kaggle/working/feats_mel_archive.zip") / 1e9
    print(f"feats_mel_archive.zip  ({sz:.2f} GB)  saved to /kaggle/working/")
    print("→ Create a Kaggle dataset named anything, upload the .zip inside it.")
    print("  Next run: attach that dataset — script will find feats_mel/ automatically.")
else:
    print("Features were loaded from an uploaded dataset — nothing to archive.")

print("\nDownload phase1_best.pth and upload it as a Kaggle dataset for Phase 2.")

import shutil
shutil.make_archive("/kaggle/working/output", "zip", "/kaggle/working")
print("Zipped everything to /kaggle/working/output.zip")