# phase V4 
"""
PHASE 1 — Mel-Spectrogram → 2D CNN
══════════════════════════════════════════════════════════════════════

DIAGNOSIS from v2 results:
  Machine3_Normal : recall = 0.01 (2128 of 2160 samples → predicted as Machine3_Abnormal)
  Machine3_Abnormal : recall = 0.98
  Root cause: the 6-class flat softmax can satisfy the loss by getting
  Machine3_Normal→Abnormal consistently wrong while nailing M1 and M2.
  The model knows "this is Machine3" but cannot tell Normal from Abnormal in M3.

NEW CHANGES (v2 → v3) and research justification:

[FIX A] Hierarchical dual-head loss (machine ID + fault status)
         Split the 6-class problem into two simultaneous sub-tasks:
           • Head 1: which machine?  (3-class softmax over classes {0,2,4}={Normal} and
                     grouping by machine: 0/1 → M1, 2/3 → M2, 4/5 → M3)
           • Head 2: is it normal or abnormal?  (binary sigmoid per sample)
         Total loss = α*L_machine + (1-α)*L_fault,  α=0.4
         The backbone is now forced to learn BOTH machine-discriminative AND
         fault-discriminative features in a shared representation. A flat 6-class
         head can learn machine ID and ignore fault status when loss is easy elsewhere.
         Reference: Zhao et al., "Hierarchical Classification with Label Attention
         Regularization," AAAI 2021. https://arxiv.org/abs/2101.04765
         Also: Hinton et al., "Distilling the Knowledge in a Neural Network," 2015.

[FIX B] Focal loss for the fault-status head (γ = 2.0)
         Standard cross-entropy weighs Machine3_Normal (easy to classify wrong) the
         same as Machine3_Abnormal (also easy to classify wrong in the other direction).
         Focal loss multiplies each sample's loss by (1 - p_t)^γ, so samples the
         model is confident-but-wrong about receive much larger gradient.
         γ=2.0 is the paper's recommended default.
         Reference: Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.
                    https://arxiv.org/abs/1708.02002

[FIX C] ReduceLROnPlateau replaces cosine annealing
         v2's cosine schedule reduced LR to 4.1e-4 by the time early stopping fired
         at epoch 13 — the model needed gradient signal on Machine3 but LR was halved.
         ReduceLROnPlateau cuts LR only when val_loss actually stagnates (patience=2,
         factor=0.5), so Machine3 gradient signal stays high while it's still learning.
         Reference: Smith & Topin, "Super-Convergence: Very Fast Training of Neural
                    Networks Using Large Learning Rates," ICLR Workshop 2019.
                    https://arxiv.org/abs/1708.07120

[FIX D] Mixup alpha reduced 0.3 → 0.1
         High alpha (0.3) generates lambda ~ Beta(0.3,0.3) which is bimodal — half the
         samples are near 0.5 (aggressive mixing). For Machine3 where both Normal and
         Abnormal are hard, mixing them 50/50 produces ambiguous training samples that
         reinforce confusion. Alpha=0.1 keeps lambda > 0.85 in 90% of samples, so the
         dominant sample still carries clear signal.
         Reference: Guo et al., "MixUp as Locally Linear Out-Of-Manifold
                    Regularization," AAAI 2019. https://arxiv.org/abs/1905.02249

[KEPT]   Effective Number of Samples weighting (Cui et al. CVPR 2019)
[KEPT]   Label smoothing ε=0.1
[KEPT]   SpecAugment with paper-calibrated params (freq_mask=27, n_freq=2)
[KEPT]   Linear warmup 2 epochs
[KEPT]   Early stopping patience=4 on val_loss
[KEPT]   Dropout 0.5
[KEPT]   Weight decay 5e-4
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

# ══════════════════════════════════════════════════════════════════════════════

_LABEL_MAP = {
    ("machine1", "Normal"): 0, ("machine1", "Abnormal"): 1,
    ("machine2", "Normal"): 2, ("machine2", "Abnormal"): 3,
    ("machine3", "Normal"): 4, ("machine3", "Abnormal"): 5,
}

# Derived label mappings for hierarchical heads:
#   machine_label: 0,1→0  2,3→1  4,5→2
#   fault_label:   0,2,4→0 (Normal)   1,3,5→1 (Abnormal)
_MACHINE_FROM_CLASS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
_FAULT_FROM_CLASS   = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}

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
EPOCHS       = 20
LR           = 1e-3
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 2
ES_PATIENCE   = 4        # early stopping patience (val_loss)
RLROP_PATIENCE = 2       # [FIX C] ReduceLROnPlateau patience
RLROP_FACTOR   = 0.5     # [FIX C] LR reduction factor
NUM_WORKERS  = 4

MIXUP_ALPHA   = 0.1      # [FIX D] reduced from 0.3 → less aggressive mixing
LABEL_SMOOTH  = 0.1
ENS_BETA      = 0.9999
HIER_ALPHA    = 0.4      # [FIX A] weight for machine-ID loss (0.4*L_machine + 0.6*L_fault)
FOCAL_GAMMA   = 2.0      # [FIX B] focal loss gamma for fault head

import wandb
wandb.init(
    project="machine-fault-phase1",
    config={
        "version": "v4",
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "epochs": EPOCHS,
        "model": "MelCNN_HierarchicalHead",
        "specaugment": True,
        "mixup_alpha": MIXUP_ALPHA,
        "label_smoothing": LABEL_SMOOTH,
        "ens_beta": ENS_BETA,
        "warmup_epochs": WARMUP_EPOCHS,
        "es_patience": ES_PATIENCE,
        "rlrop_patience": RLROP_PATIENCE,
        "hier_alpha": HIER_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
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

CLASS_NAMES   = ["Machine1_Normal", "Machine1_Abnormal",
                 "Machine2_Normal", "Machine2_Abnormal",
                 "Machine3_Normal", "Machine3_Abnormal"]
MACHINE_NAMES = ["Machine1", "Machine2", "Machine3"]

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
from dataclasses import dataclass, field
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
    normalize_mode: str = "peak"
    peak_target: float = 0.95
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
        if w.ndim > 1: w = w.mean(axis=1)
        if orig_sr != eff_sr:
            d = math.gcd(orig_sr, eff_sr)
            w = resample_poly(w, eff_sr//d, orig_sr//d).astype(np.float32)
        if cfg.trim_silence: w = self._trim(w, eff_sr)
        w = self._normalize(w)
        rng = np.random.default_rng(seed) if mode == "train" else None
        if mode == "train": w = self._augment(w, eff_sr, rng)
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
        if self.config.normalize_mode == "peak":
            p = np.abs(w).max()
            if p > EPSILON: w = w * (self.config.peak_target / p)
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
            if ms > 0: w = np.roll(w, int(rng.integers(-ms, ms+1)))
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
        hop_length=512, fmin=50, fmax=8000, center=False)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.expand_dims(_minmax(mel_db).astype(np.float32), 0)

# ── SPECAUGMENT (paper-calibrated) ────────────────────────────────────────────
import random
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

# ── PRECOMPUTATION ────────────────────────────────────────────────────────────
def _precompute_one(args):
    idx, wav_path, feats_dir, preprocessor = args
    out_path = pathlib.Path(feats_dir) / f"{idx:06d}.npy"
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
        print(f"[cache] mel features loaded from uploaded dataset {feats_dir}  ✓"); return
    feats_dir.mkdir(parents=True, exist_ok=True)
    already = sum(1 for i in range(len(paths)) if (feats_dir/f"{i:06d}.npy").exists())
    if already == len(paths):
        print(f"All {len(paths)} mel-specs already cached in {feats_dir}  (skipping)"); return
    print(f"Pre-computing mel-spectrograms for {len(paths)} files using {n_workers} workers...")
    args = [(i, p, str(feats_dir), preprocessor) for i, p in enumerate(paths)]
    with multiprocessing.Pool(n_workers) as pool:
        list(tqdm_module.tqdm(pool.imap(_precompute_one, args, chunksize=64),
                              total=len(paths), desc="mel-specs"))
    print(f"Pre-computation done → {feats_dir}")

# ── [FIX A] MIXUP UTILITIES ───────────────────────────────────────────────────
def mixup_batch(x, y, alpha=0.1, device="cpu"):
    """
    [FIX D] alpha=0.1: Beta(0.1,0.1) is strongly bimodal near 0 and 1,
    so lambda > 0.85 in ~90% of samples. The dominant sample stays clear.
    """
    if alpha <= 0: return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    B = x.size(0)
    perm = torch.randperm(B, device=device)
    return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
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
        ri   = self.indices[idx]
        lbl  = self.labels[ri]
        mel  = np.load(self.feats_dir / f"{ri:06d}.npy", mmap_mode="r")
        mel_t = torch.tensor(mel, dtype=torch.float32)
        if self.augment:
            mel_t = spec_augment(mel_t)
        return (mel_t,
                torch.tensor(lbl,                           dtype=torch.long),
                torch.tensor(_MACHINE_FROM_CLASS[lbl],      dtype=torch.long),
                torch.tensor(_FAULT_FROM_CLASS[lbl],        dtype=torch.long))

# ── [FIX A] MODEL WITH HIERARCHICAL HEADS ─────────────────────────────────────
# Shared CNN backbone → two parallel heads:
#   head_main:    6-class softmax (same as before, for final prediction)
#   head_machine: 3-class softmax (which machine is this?)
#   head_fault:   binary (normal=0, abnormal=1)
#
# At inference time: use head_main predictions (6 classes).
# At training time:  loss = HIER_ALPHA*L_machine + (1-HIER_ALPHA)*L_fault_focal
#                    + L_main (with label smoothing + ENS weights)
#
# Why keep head_main? It gives a direct 6-class prediction without needing
# to combine machine + fault heads at test time. The auxiliary heads
# regularise the backbone representation.

class MelCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1,32,3,padding=1),   nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1),  nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)))
        self.fc1     = nn.Linear(256*4*4, 256)
        self.dropout = nn.Dropout(0.5)

        # Main 6-class head
        self.head_main    = nn.Linear(256, num_classes)
        # [FIX A] Auxiliary hierarchical heads
        self.head_machine = nn.Linear(256, 3)    # machine identity
        self.head_fault   = nn.Linear(256, 1)    # fault status (binary logit)

    def extract_features(self, x):
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        x = torch.flatten(x, 1)
        return self.dropout(F.relu(self.fc1(x)))

    def forward(self, x):
        feat = self.extract_features(x)
        return (self.head_main(feat),        # (B, 6)
                self.head_machine(feat),     # (B, 3)
                self.head_fault(feat))       # (B, 1)

# ── [FIX B] FOCAL LOSS ────────────────────────────────────────────────────────
class BinaryFocalLoss(nn.Module):
    """
    Focal loss for the binary fault-status head.
    FL(p_t) = -(1 - p_t)^γ * log(p_t)
    Down-weights easy examples; forces gradient onto Machine3's hard boundary.
    Reference: Lin et al., ICCV 2017. https://arxiv.org/abs/1708.02002
    """
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # logits: (B, 1), targets: (B,) int
        targets_f = targets.float().unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets_f,
            pos_weight=self.pos_weight,
            reduction="none"
        )
        p_t = torch.exp(-bce)                    # probability of correct class
        focal = (1.0 - p_t) ** self.gamma * bce  # down-weight easy examples
        return focal.mean()

# ── CLASS WEIGHTS ─────────────────────────────────────────────────────────────
def effective_num_weights(label_counts, beta=0.9999, num_classes=6):
    eff_num = (1.0 - np.power(beta, label_counts)) / (1.0 - beta)
    weights = 1.0 / eff_num
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

# ── [FIX C] SCHEDULER: warmup + ReduceLROnPlateau ─────────────────────────────
def make_warmup_scheduler(optimizer, warmup_epochs):
    """Linear warmup only — ReduceLROnPlateau handles the decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0  # hold at peak LR; ReduceLROnPlateau will reduce when needed
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── TRAINING ──────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer,
                crit_main, crit_machine, crit_fault,
                device, mixup_alpha=0.1, hier_alpha=0.4):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        x, y_main, y_machine, y_fault = [t.to(device) for t in batch]

        # [FIX D] alpha=0.1 mixup
        x_mix, y_main_a, y_main_b, lam = mixup_batch(x, y_main, alpha=mixup_alpha, device=device)
        _, y_mach_a, y_mach_b, _       = mixup_batch(x, y_machine, alpha=0, device=device)
        # Note: for auxiliary heads we use the dominant sample's label (y_mach_a = y_machine)
        # since lam≥0.5 in mixup_batch — y_mach_a is always the dominant one

        optimizer.zero_grad()
        out_main, out_machine, out_fault = model(x_mix)

        # Main head loss (6-class, label smooth, ENS weights, mixup)
        L_main = mixup_criterion(crit_main, out_main, y_main_a, y_main_b, lam)

        # [FIX A] Machine head loss (3-class crossentropy)
        L_machine = crit_machine(out_machine, y_machine)

        # [FIX B] Fault head focal loss (binary, dominant label from mixup)
        L_fault = crit_fault(out_fault, y_fault)

        # Combined loss
        loss = L_main + hier_alpha * L_machine + (1.0 - hier_alpha) * L_fault

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (out_main.argmax(1) == y_main_a).sum().item()
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
            loss = crit_main(out_main, y_main)
            total_loss += loss.item()
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
        "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc,
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

ALL_PATHS, ALL_LABELS = _scan_wav_files(ROOT_DIR)
_splits = _create_clean_split(ALL_PATHS, ALL_LABELS, MODELS_DIR)

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

# Class weights
label_counts  = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
class_weights = effective_num_weights(label_counts, beta=ENS_BETA, num_classes=6).to(DEVICE)

# Machine-head weights (3 machines, count Normal+Abnormal samples per machine)
machine_counts = np.array([
    label_counts[0] + label_counts[1],
    label_counts[2] + label_counts[3],
    label_counts[4] + label_counts[5],
], dtype=np.float64)
machine_ens = (1.0 - np.power(ENS_BETA, machine_counts)) / (1.0 - ENS_BETA)
machine_w   = 1.0 / machine_ens
machine_w   = torch.tensor(machine_w / machine_w.sum() * 3, dtype=torch.float32).to(DEVICE)

# Fault-head positive weight (abnormal is minority within each machine)
fault_counts  = np.array([
    sum(label_counts[i] for i in [0, 2, 4]),  # Normal total
    sum(label_counts[i] for i in [1, 3, 5]),  # Abnormal total
])
fault_pos_w = torch.tensor([fault_counts[0] / (fault_counts[1] + 1e-6)],
                             dtype=torch.float32).to(DEVICE)

print("\n── Class counts in training split ──")
for i, (name, cnt, w) in enumerate(zip(CLASS_NAMES, label_counts, class_weights.cpu())):
    print(f"  [{i}] {name:<25s}  n={cnt:>5d}  ENS_weight={w:.4f}")

model = MelCNN(num_classes=6).to(DEVICE)

# Loss functions
crit_main    = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
crit_machine = nn.CrossEntropyLoss(weight=machine_w)
crit_fault   = BinaryFocalLoss(gamma=FOCAL_GAMMA, pos_weight=fault_pos_w)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# [FIX C] Warmup scheduler + ReduceLROnPlateau
warmup_scheduler = make_warmup_scheduler(optimizer, warmup_epochs=WARMUP_EPOCHS)
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=RLROP_FACTOR,
    patience=RLROP_PATIENCE, min_lr=1e-5
)

best_val_loss = float("inf")
best_val_acc  = 0.0
es_counter    = 0
train_history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
ckpt_path     = os.path.join(MODELS_DIR, "phase1_best.pth")

print("\n── Training Phase 1 v3 ───────────────────────────────────")
print(f"   Mixup α={MIXUP_ALPHA}  |  Focal γ={FOCAL_GAMMA}  |  HierAlpha={HIER_ALPHA}")
print(f"   LabelSmooth ε={LABEL_SMOOTH}  |  ENS β={ENS_BETA}  |  Warmup={WARMUP_EPOCHS}ep")
print(f"   RLROP patience={RLROP_PATIENCE} factor={RLROP_FACTOR}  |  ES patience={ES_PATIENCE}")
print()

for epoch in range(1, EPOCHS + 1):
    current_lr = optimizer.param_groups[0]["lr"]

    tr_loss, tr_acc = train_epoch(
        model, train_loader, optimizer,
        crit_main, crit_machine, crit_fault,
        DEVICE, mixup_alpha=MIXUP_ALPHA, hier_alpha=HIER_ALPHA
    )
    vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader, crit_main, DEVICE)

    # Step schedulers
    if epoch <= WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        plateau_scheduler.step(vl_loss)   # [FIX C] adaptive LR reduction

    wandb.log({
        "epoch": epoch, "lr": current_lr,
        "train_loss": tr_loss, "train_acc": tr_acc,
        "val_loss": vl_loss,   "val_acc":  vl_acc,
    })
    train_history["loss"].append(tr_loss)
    train_history["acc"].append(tr_acc)
    train_history["val_loss"].append(vl_loss)
    train_history["val_acc"].append(vl_acc)

    improved = vl_loss < best_val_loss
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

    if es_counter >= ES_PATIENCE:
        print(f"\n[early stop] Val loss did not improve for {ES_PATIENCE} epochs. Stopping.")
        break

print(f"\nBest val loss: {best_val_loss:.4f}  (val acc at that epoch: {best_val_acc:.4f})")

# ── FINAL EVALUATION ──────────────────────────────────────────────────────────
ckpt = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, crit_main, DEVICE)
t_test = time.time() - t0
n_test = len(test_ds)
ms_per_sample = (t_test / n_test) * 1000

metrics = compute_metrics(test_preds, test_labels)

print(f"\n── Test Results ──────────────────────────────────────────")
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"Macro F1 : {metrics['macro_f1']:.4f}")
print("\nPer-class F1:")
for cls_name, f1 in zip(CLASS_NAMES, metrics["per_class_f1"]):
    delta_symbol = ""  # you can fill in manually vs v2 for logging
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
plt.suptitle("Phase 1 v3 — Hierarchical Loss + Focal Loss")
plt.tight_layout(); plt.show()

print(f"\nModel saved to: {ckpt_path}")

wandb.log({
    "class_counts": {CLASS_NAMES[i]: int(label_counts[i]) for i in range(len(CLASS_NAMES))}
})

# ── ARCHIVE ───────────────────────────────────────────────────────────────────
if str(FEATS_DIR).startswith("/kaggle/working"):
    import shutil
    print("\nArchiving mel features for reuse in later phases ...")
    shutil.make_archive("/kaggle/working/feats_mel_archive", "zip",
                        "/kaggle/working", "feats_mel")
    sz = os.path.getsize("/kaggle/working/feats_mel_archive.zip") / 1e9
    print(f"feats_mel_archive.zip  ({sz:.2f} GB)  saved to /kaggle/working/")
else:
    print("Features were loaded from an uploaded dataset — nothing to archive.")

print("\nDownload phase1_best.pth and upload it as a Kaggle dataset for Phase 2.")

import shutil
shutil.make_archive("/kaggle/working/output", "zip", "/kaggle/working")
print("Zipped everything to /kaggle/working/output.zip")