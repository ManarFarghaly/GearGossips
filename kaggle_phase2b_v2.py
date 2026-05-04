"""
PHASE 2b (V2) — Mel-Spectrogram + Statistical Features with kurtosis  (Phase 2 but instead of global norma we did per machine one)
═══════════════════════════════════════════════════════════════════════════════════
Changes from the original phase2b:

[FIX 0] ReduceLROnPlateau verbose=True removed — keyword dropped in PyTorch ≥ 2.2.
         LR is logged every epoch via wandb instead.

[FIX 1] Loads from Phase 1 v4 (hierarchical head architecture).
         Phase 1 v4 has three heads: head_main, head_machine, head_fault.
         Phase 2b only needs the CNN backbone (mel_stream). We load the
         backbone weights by strict=False + key remapping, so both v1/v2/v3
         phase1 checkpoints work transparently.

[FIX 2] Per-machine statistical feature normalization (replaces global normalization).
         Machine1/2/3 run at different speeds, loads, and acoustic environments.
         Their Normal class has different mean spectral energy, zcr, etc.
         Fitting a single global scaler mixes these distributions — Machine3_Normal's
         stat vector can end up closer to Machine2_Abnormal than Machine3_Abnormal.
         Per-machine normalization removes the inter-machine shift:
           fit scaler separately on {train samples from machine k} for k ∈ {0,1,2}
           apply the matching scaler at inference time (machine known from label/path)
         Reference: Ben-David et al., "A theory of learning from different domains,"
                    Machine Learning 2010. https://link.springer.com/article/10.1007/s10994-009-5152-4

[FIX 3] Effective Number of Samples (ENS) class weights for main loss.
         Reference: Cui et al., CVPR 2019. https://arxiv.org/abs/1901.05555

[FIX 4] Label smoothing ε = 0.1 on main CrossEntropyLoss.
         Reference: Müller et al., NeurIPS 2019. https://arxiv.org/abs/1906.02629

[FIX 5] Focal loss (γ = 2.0) on the fault-status auxiliary head.
         The phase2b model gains an auxiliary binary fault head (normal/abnormal)
         trained with focal loss, matching the Phase 1 v3 training strategy.
         This directly addresses the Machine3 Normal/Abnormal collapse.
         Reference: Lin et al., ICCV 2017. https://arxiv.org/abs/1708.02002

[FIX 6] Hierarchical auxiliary loss (machine-ID head) — same as Phase 1 v3.
         Weights: 0.4 * L_machine + 0.6 * L_fault (focal) + L_main

[FIX 7] Mixup α = 0.1 (reduced from any prior values).
         Reference: Guo et al., AAAI 2019. https://arxiv.org/abs/1905.02249

[FIX 8] ReduceLROnPlateau replaces cosine annealing.
         Reference: Smith & Topin, ICLR Workshop 2019. https://arxiv.org/abs/1708.07120

[FIX 9] Early stopping on val_loss (patience = 5, generous since stat branch needs warmup).

[NEW]   Statistical features — scientific justification:
         • Kurtosis: primary fault indicator for rotating machinery. Impulsive faults
           (bearing spall, gear crack) produce spike-like vibration → kurtosis rises
           sharply above the Normal Gaussian baseline (~3.0). The mel-CNN sees frequency
           patterns; kurtosis captures time-domain impulsiveness that mel compression loses.
           Reference: Antoni & Randall, "The spectral kurtosis: application to the
           vibratory surveillance and diagnostics of rotating machines,"
           Mechanical Systems and Signal Processing, 2006.
         • RMS: energy — abnormal machines produce higher mechanical energy overall.
         • ZCR: frequency content proxy — structural looseness shifts crossing rate.
         • Spectral rolloff + bandwidth: coarse spectral shape from resonance shifts.

[KEPT]  SpecAugment with paper-calibrated params (freq_mask=27, n_freq=2, time_mask=15, n_time=2)
[KEPT]  Gradient clipping max_norm=1.0
[KEPT]  Weight decay 5e-4
[KEPT]  Warmup 2 epochs (linear, then ReduceLROnPlateau takes over)
[KEPT]  Differential LRs: mel_stream 1e-4, new branches 5e-4

WHY STATS OVER MFCC HERE:
  Phase 2 (Mel+MFCC) adds a second CNN — 2× inference latency.
  Phase 2b (Mel+Stat) adds a 3-layer FC branch — negligible latency (+0.01 ms).
  Kurtosis specifically targets the impulsive Machine3 Normal/Abnormal difference
  that mel alone cannot resolve, without the compute cost of a second CNN.
  MFCC is a frequency-domain redundancy with mel — stat features add
  complementary time-domain information. This is the key distinction.
"""

import os, json, math, pathlib, random, time, pickle
from scipy.stats import kurtosis as scipy_kurtosis
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import soundfile as sf
from scipy.signal import resample_poly
import shutil

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
        print(f"[split] {len(dup_groups)} duplicate groups ({n} files)")
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
    print(f"[split] Train={len(tr)}  Val={len(va)}  Test={len(te)}  Boundary_pairs={bp}")
    print("[split] Zero temporal leakage" if bp == 0 else f"[split] {bp} boundary pairs")

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
        raise FileNotFoundError(f"split_indices_clean.json not found at {path}.")
    return json.load(open(path))

# ══════════════════════════════════════════════════════════════════════════════

_LABEL_MAP = {
    ("machine1", "Normal"): 0, ("machine1", "Abnormal"): 1,
    ("machine2", "Normal"): 2, ("machine2", "Abnormal"): 3,
    ("machine3", "Normal"): 4, ("machine3", "Abnormal"): 5,
}

# Derived label mappings for hierarchical heads (same as Phase 1 v3)
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
        raise RuntimeError(f"No labelled .wav files found under {root_dir}.")
    print(f"Found {len(paths)} files")
    return paths, labels

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT_DIR    = "/kaggle/input/datasets/mostafaehab41/machine-fault-dataset"
PHASE1_CKPT = "/kaggle/input/datasets/hanyfargally/phase1-best-pth/phase1_best.pth" # use phase 1 -> v3 best one 
MODELS_DIR  = "/kaggle/working"

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SR           = 16000
DURATION_SEC = 2.75
BATCH_SIZE   = 32
TRAIN_EPOCHS = 25
NUM_WORKERS  = 2

# Loss / regularisation hyperparams (aligned with Phase 1 v3)
LABEL_SMOOTH  = 0.1
ENS_BETA      = 0.9999
MIXUP_ALPHA   = 0.1
FOCAL_GAMMA   = 2.0
HIER_ALPHA    = 0.4      # weight of machine-ID auxiliary loss
WEIGHT_DECAY  = 5e-4
WARMUP_EPOCHS = 2
RLROP_PATIENCE = 2
RLROP_FACTOR   = 0.5
ES_PATIENCE    = 5       # generous: stat branch needs a few epochs to warm up

# Mel backbone LR kept very low — backbone already well-trained from Phase 1 v3
LR_MEL_STREAM = 1e-4
LR_NEW_LAYERS = 5e-4

CLASS_NAMES   = ["Machine1_Normal", "Machine1_Abnormal",
                 "Machine2_Normal", "Machine2_Abnormal",
                 "Machine3_Normal", "Machine3_Abnormal"]
MACHINE_NAMES = ["Machine1", "Machine2", "Machine3"]

STAT_FEATURES = ["rms", "zcr", "rolloff", "bandwidth", "kurtosis"]
STAT_DIM      = len(STAT_FEATURES)   # 5

import wandb
wandb.init(
    project="machine-fault-phase2b",
    config={
        "version": "v2",
        "batch_size": BATCH_SIZE,
        "lr_mel_stream": LR_MEL_STREAM,
        "lr_new_layers": LR_NEW_LAYERS,
        "train_epochs": TRAIN_EPOCHS,
        "model": "MelStatCNN_Hierarchical",
        "stat_features": STAT_FEATURES,
        "label_smoothing": LABEL_SMOOTH,
        "ens_beta": ENS_BETA,
        "mixup_alpha": MIXUP_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "hier_alpha": HIER_ALPHA,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "rlrop_patience": RLROP_PATIENCE,
        "es_patience": ES_PATIENCE,
        "per_machine_normalization": True,
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
                print(f"[cache] '{name}' found at {candidate}  ✓")
                return candidate
    working = pathlib.Path("/kaggle/working") / name
    print(f"[cache] '{name}' not found → will compute to {working}")
    return working

FEATS_DIR_MEL  = _feat_dir("feats_mel")
FEATS_DIR_STAT = _feat_dir("feats_stat_v2")

def _copy_if_input(src, dst):
    """Read-only /kaggle/input dirs must be copied to /kaggle/working to be usable."""
    src = pathlib.Path(src)
    if str(src).startswith("/kaggle/input"):
        dst = pathlib.Path(dst)
        if not dst.exists():
            print(f"Copying {src} → {dst}")
            shutil.copytree(src, dst, dirs_exist_ok=True)
        return dst
    return src

FEATS_DIR_MEL  = _copy_if_input(FEATS_DIR_MEL,  "/kaggle/working/feats_mel")
FEATS_DIR_STAT = _copy_if_input(FEATS_DIR_STAT, "/kaggle/working/feats_stat_v2")

print(f"PHASE1_CKPT : {PHASE1_CKPT}")
print(f"Checkpoint exists: {pathlib.Path(PHASE1_CKPT).exists()}")

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
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
        eff_sr  = int(target_sr or cfg.target_sr)
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

def compute_stat_v2(waveform, sr=16000):
    """5 features: [rms, zcr, rolloff, bandwidth, kurtosis]"""
    return np.array([
        float(np.sqrt(np.mean(waveform ** 2))),
        float(librosa.feature.zero_crossing_rate(waveform).mean()),
        float(librosa.feature.spectral_rolloff(y=waveform, sr=sr).mean()),
        float(librosa.feature.spectral_bandwidth(y=waveform, sr=sr).mean()),
        float(scipy_kurtosis(waveform, fisher=True)),
    ], dtype=np.float32)

# ── SPECAUGMENT (paper-calibrated, same as Phase 1 v3) ───────────────────────
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

# ── PRE-COMPUTATION ───────────────────────────────────────────────────────────
def _precompute_one(args):
    idx, wav_path, mel_dir, stat_dir, preprocessor = args
    mel_out  = pathlib.Path(mel_dir)  / f"{idx:06d}.npy"
    stat_out = pathlib.Path(stat_dir) / f"{idx:06d}.npy"
    if mel_out.exists() and stat_out.exists(): return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        if not mel_out.exists():  np.save(mel_out,  compute_mel_spectrogram(w))
        if not stat_out.exists(): np.save(stat_out, compute_stat_v2(w))
    except Exception:
        if not mel_out.exists():  np.save(mel_out,  np.zeros((1, 128, 84), dtype=np.float32))
        if not stat_out.exists(): np.save(stat_out, np.zeros(5,             dtype=np.float32))

def precompute_all(paths, mel_dir, stat_dir, preprocessor, n_workers=4):
    import multiprocessing, tqdm as tqdm_module
    mel_dir  = pathlib.Path(mel_dir)
    stat_dir = pathlib.Path(stat_dir)
    mel_dir.mkdir(parents=True, exist_ok=True)
    stat_dir.mkdir(parents=True, exist_ok=True)
    already = sum(1 for i in range(len(paths))
                  if (mel_dir/f"{i:06d}.npy").exists() and (stat_dir/f"{i:06d}.npy").exists())
    if already == len(paths):
        print(f"All {len(paths)} mel+stat features already cached  (skipping)"); return
    print(f"Pre-computing mel + stat for {len(paths)} files using {n_workers} workers ...")
    args = [(i, p, str(mel_dir), str(stat_dir), preprocessor) for i, p in enumerate(paths)]
    with multiprocessing.Pool(n_workers) as pool:
        list(tqdm_module.tqdm(pool.imap(_precompute_one, args, chunksize=64),
                              total=len(paths), desc="mel+stat"))
    print("Pre-computation done.")

# ── [FIX 2] PER-MACHINE STAT NORMALISATION ────────────────────────────────────
# Machine 0,1 → label 0,1 → machine_id 0
# Machine 2,3 → label 2,3 → machine_id 1
# Machine 4,5 → label 4,5 → machine_id 2
#
# We fit a separate (mean, std) for each machine using only training samples
# from that machine. At inference time, each sample is normalised by the
# scaler of the machine it belongs to (known from the label).
#
# Rationale: Machine3's Normal class has a different baseline RMS, ZCR, and
# kurtosis than Machine1/2 Normal. A global scaler merges these distributions,
# making Machine3_Normal's stat vector overlap with Machine2_Abnormal. Per-machine
# normalization removes this inter-machine shift so the fault-status boundary
# is learned within each machine's own distribution.
# Reference: Ben-David et al., "A theory of learning from different domains,"
#            Machine Learning 2010.

def fit_per_machine_scalers(stat_dir, labels, indices):
    """
    Returns:
        scalers: list of (mean, std) for machine 0, 1, 2
                 each mean/std is shape (STAT_DIM,)
    """
    stat_dir = pathlib.Path(stat_dir)
    machine_stats = {m: [] for m in range(3)}
    for i in indices:
        m = _MACHINE_FROM_CLASS[labels[i]]
        machine_stats[m].append(np.load(stat_dir / f"{i:06d}.npy"))

    scalers = []
    for m in range(3):
        arr = np.stack(machine_stats[m])
        mean = arr.mean(0)
        std  = arr.std(0) + 1e-8
        scalers.append((mean, std))
        print(f"  Machine{m+1} scaler: n={len(machine_stats[m])}  "
              f"mean={mean.round(4)}  std={std.round(4)}")
    return scalers

# ── DATASET ───────────────────────────────────────────────────────────────────
class PrecomputedDatasetMS(Dataset):
    """
    Returns (mel, stat_raw, label, machine_id, fault_label) per item.
    Stat normalisation is applied per-batch in the training loop using
    the machine-specific scalers — this avoids storing 3 copies of the
    dataset or doing per-sample branching in __getitem__.
    """
    def __init__(self, mel_dir, stat_dir, labels, indices, augment=False):
        self.mel_dir  = pathlib.Path(mel_dir)
        self.stat_dir = pathlib.Path(stat_dir)
        self.labels   = labels
        self.indices  = indices
        self.augment  = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri  = self.indices[idx]
        lbl = self.labels[ri]
        mel = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        stat = torch.tensor(np.load(self.stat_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment:
            mel = spec_augment(mel)
        return (mel, stat,
                torch.tensor(lbl,                        dtype=torch.long),
                torch.tensor(_MACHINE_FROM_CLASS[lbl],   dtype=torch.long),
                torch.tensor(_FAULT_FROM_CLASS[lbl],     dtype=torch.long))

def collate_ms(batch):
    mel, stat, lbl, mach, fault = zip(*batch)
    return (torch.stack(mel), torch.stack(stat),
            torch.stack(lbl), torch.stack(mach), torch.stack(fault))

def normalise_stat_per_machine(stat, machine_ids, scalers, device):
    """
    stat       : (B, STAT_DIM) raw stat tensor
    machine_ids: (B,) long tensor, values in {0,1,2}
    scalers    : list of (mean_np, std_np) arrays, one per machine
    Returns: (B, STAT_DIM) normalised tensor
    """
    out = stat.clone()
    for m, (mean, std) in enumerate(scalers):
        mask = (machine_ids == m)
        if mask.any():
            m_t   = torch.tensor(mean, dtype=torch.float32, device=device)
            s_t   = torch.tensor(std,  dtype=torch.float32, device=device)
            out[mask] = (stat[mask] - m_t) / s_t
    return out

# ── MODEL ─────────────────────────────────────────────────────────────────────
# Backbone: same MelCNN as Phase 1 (architecture unchanged so weights transfer)
# New: stat_branch + auxiliary heads

class MelCNN(nn.Module):
    """Phase 1 v3 backbone — must match exactly for weight loading."""
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1,32,3,padding=1),   nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1),  nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)))
        self.fc1     = nn.Linear(256*4*4, 256)
        self.dropout = nn.Dropout(0.5)
        # These heads exist to match Phase 1 v3 checkpoint keys; unused in Phase 2b forward
        self.head_main    = nn.Linear(256, num_classes)
        self.head_machine = nn.Linear(256, 3)
        self.head_fault   = nn.Linear(256, 1)

    def extract_features(self, x):
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        x = torch.flatten(x, 1)
        return self.dropout(F.relu(self.fc1(x)))   # (B, 256)

    def forward(self, x):
        feat = self.extract_features(x)
        return self.head_main(feat), self.head_machine(feat), self.head_fault(feat)


class MelStatCNN(nn.Module):
    """
    Architecture:
      mel_stream  (256-d) ─┐
                            ├─ cat(288-d) → fc1(256) → Dropout(0.4) → fc2(6)  [head_main]
      stat_branch  (32-d) ─┘              → head_machine(3), head_fault(1)    [auxiliary]

    Differential LRs:
      mel_stream : 1e-4  (low — already trained in Phase 1)
      everything else: 5e-4
    """
    def __init__(self, num_classes=6, stat_dim=5):
        super().__init__()
        self.mel_stream = MelCNN(num_classes)

        # Stat branch: two FC layers
        self.stat_branch = nn.Sequential(
            nn.Linear(stat_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32),       nn.ReLU()
        )

        # Fusion layers
        self.fc1     = nn.Linear(256 + 32, 256)
        self.dropout = nn.Dropout(0.4)

        # Output heads (hierarchical, same as Phase 1 v3)
        self.head_main    = nn.Linear(256, num_classes)
        self.head_machine = nn.Linear(256, 3)
        self.head_fault   = nn.Linear(256, 1)

    def forward(self, mel, stat):
        f_mel  = self.mel_stream.extract_features(mel)    # (B, 256)
        f_stat = self.stat_branch(stat)                   # (B, 32)
        fused  = self.dropout(F.relu(self.fc1(
            torch.cat([f_mel, f_stat], dim=1))))          # (B, 256)
        return (self.head_main(fused),
                self.head_machine(fused),
                self.head_fault(fused))


def load_phase1_weights(model, ckpt_path, device):
    """
    Load mel_stream weights from Phase 1 v3 checkpoint.
    Phase 1 v3 top-level keys: head_main.*, head_machine.*, head_fault.*, block*.*, fc1.*, dropout.*
    Phase 2b mel_stream keys:  mel_stream.head_main.*, mel_stream.block*.*, etc.
    We prefix 'mel_stream.' to every Phase 1 key and load with strict=False
    so the new stat_branch, fc1, head_* layers are randomly initialised.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    p1_state = ckpt["model_state_dict"]

    # Remap: prepend "mel_stream." to all Phase 1 keys
    remapped = {"mel_stream." + k: v for k, v in p1_state.items()}

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    mel_keys   = [k for k in missing if not k.startswith("mel_stream.")]
    extra_keys = [k for k in unexpected]
    print(f"[ckpt] Loaded {len(remapped) - len(missing)} / {len(remapped)} keys from Phase 1 checkpoint.")
    if mel_keys:
        print(f"[ckpt] New (randomly init): {mel_keys[:8]}{'...' if len(mel_keys)>8 else ''}")
    if extra_keys:
        print(f"[ckpt] Unexpected keys (ignored): {extra_keys[:4]}")

# ── [FIX 5,6] LOSS FUNCTIONS ──────────────────────────────────────────────────
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        targets_f = targets.float().unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets_f, pos_weight=self.pos_weight, reduction="none")
        p_t   = torch.exp(-bce)
        focal = (1.0 - p_t) ** self.gamma * bce
        return focal.mean()

def effective_num_weights(label_counts, beta=0.9999, num_classes=6):
    eff_num = (1.0 - np.power(beta, label_counts)) / (1.0 - beta)
    weights = 1.0 / eff_num
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

# ── [FIX 7] MIXUP ─────────────────────────────────────────────────────────────
def mixup_batch(x, y, alpha=0.1, device="cpu"):
    if alpha <= 0: return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    B   = x.size(0)
    perm = torch.randperm(B, device=device)
    return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

# ── WARMUP SCHEDULER ──────────────────────────────────────────────────────────
def make_warmup_scheduler(optimizer, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0   # ReduceLROnPlateau handles decay after warmup
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer,
                crit_main, crit_machine, crit_fault,
                scalers, device,
                mixup_alpha=0.1, hier_alpha=0.4):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for mel, stat, y_main, y_machine, y_fault in loader:
        mel, stat = mel.to(device), stat.to(device)
        y_main, y_machine, y_fault = y_main.to(device), y_machine.to(device), y_fault.to(device)

        # [FIX 2] Per-machine normalisation
        stat = normalise_stat_per_machine(stat, y_machine, scalers, device)

        # [FIX 7] Mixup on mel only (stat is low-dim; mixing it adds minimal value)
        mel_mix, y_a, y_b, lam = mixup_batch(mel, y_main, alpha=mixup_alpha, device=device)

        optimizer.zero_grad()
        out_main, out_machine, out_fault = model(mel_mix, stat)

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


def eval_epoch(model, loader, crit_main, scalers, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, stat, y_main, y_machine, y_fault in loader:
            mel, stat = mel.to(device), stat.to(device)
            y_main, y_machine = y_main.to(device), y_machine.to(device)

            stat = normalise_stat_per_machine(stat, y_machine, scalers, device)
            out_main, _, _ = model(mel, stat)

            loss  = crit_main(out_main, y_main)
            preds = out_main.argmax(1)
            total_loss += loss.item()
            correct    += (preds == y_main).sum().item()
            total      += y_main.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_main.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels


def save_checkpoint(model, epoch, val_loss, val_acc, scalers, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch":    epoch,
        "val_loss": val_loss,
        "val_acc":  val_acc,
        "stat_features": STAT_FEATURES,
        "stat_dim":      STAT_DIM,
        "scalers":       scalers,   # list of (mean, std) per machine
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
    plt.title("Confusion Matrix — Phase 2b"); plt.tight_layout(); plt.show()

# ── MAIN ──────────────────────────────────────────────────────────────────────

# Step A: scan files
ALL_PATHS, ALL_LABELS = _scan_wav_files(ROOT_DIR)

# Step B: pre-compute mel + stat
_infer_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=SR, default_duration_sec=DURATION_SEC,
    trim_silence=True, normalize_mode="peak",
    augmentation=AugmentationConfig(enabled=False),
))
precompute_all(ALL_PATHS, FEATS_DIR_MEL, FEATS_DIR_STAT, _infer_prep, n_workers=NUM_WORKERS)

# Step C: split — load if Phase 1 already created it, else create fresh
split_json = pathlib.Path(MODELS_DIR) / "split_indices_clean.json"
if split_json.exists():
    _splits = json.load(open(split_json))
    print(f"[split] Loaded existing split → Train={len(_splits['train'])}  "
          f"Val={len(_splits['val'])}  Test={len(_splits['test'])}")
else:
    _splits = _create_clean_split(ALL_PATHS, ALL_LABELS, MODELS_DIR)

# Step D: [FIX 2] fit per-machine scalers on training set only
print("\nFitting per-machine stat scalers (training split only) ...")
scalers = fit_per_machine_scalers(FEATS_DIR_STAT, ALL_LABELS, _splits["train"])

# Step E: class weights
label_counts  = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
class_weights = effective_num_weights(label_counts, beta=ENS_BETA).to(DEVICE)

machine_counts = np.array([label_counts[0]+label_counts[1],
                            label_counts[2]+label_counts[3],
                            label_counts[4]+label_counts[5]], dtype=np.float64)
machine_ens = (1.0 - np.power(ENS_BETA, machine_counts)) / (1.0 - ENS_BETA)
machine_w   = torch.tensor(1.0/machine_ens / (1.0/machine_ens).sum() * 3,
                            dtype=torch.float32).to(DEVICE)

fault_counts = np.array([sum(label_counts[i] for i in [0,2,4]),
                          sum(label_counts[i] for i in [1,3,5])], dtype=np.float64)
fault_pos_w  = torch.tensor([fault_counts[0] / (fault_counts[1]+1e-6)],
                              dtype=torch.float32).to(DEVICE)

print("\n── Class counts (training split) ──")
for i, (name, cnt, w) in enumerate(zip(CLASS_NAMES, label_counts, class_weights.cpu())):
    print(f"  [{i}] {name:<25s}  n={cnt:>5d}  ENS_weight={w:.4f}")

# Step F: datasets and loaders
tr_ds = PrecomputedDatasetMS(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["train"], augment=True)
vl_ds = PrecomputedDatasetMS(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["val"],   augment=False)
te_ds = PrecomputedDatasetMS(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["test"],  augment=False)

print(f"Train: {len(tr_ds)}  Val: {len(vl_ds)}  Test: {len(te_ds)}")

tr_ldr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=collate_ms, num_workers=NUM_WORKERS,
                    pin_memory=True, persistent_workers=False, prefetch_factor=2)
vl_ldr = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False,
                    collate_fn=collate_ms, num_workers=NUM_WORKERS,
                    pin_memory=True, persistent_workers=False, prefetch_factor=2)
te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                    collate_fn=collate_ms, num_workers=NUM_WORKERS,
                    pin_memory=True, persistent_workers=False, prefetch_factor=2)

# Step G: model + load Phase 1 weights
model = MelStatCNN(num_classes=6, stat_dim=STAT_DIM).to(DEVICE)
load_phase1_weights(model, PHASE1_CKPT, DEVICE)

# Step H: optimiser — differential LRs
optimizer = torch.optim.AdamW([
    {"params": model.mel_stream.parameters(),  "lr": LR_MEL_STREAM},
    {"params": model.stat_branch.parameters(), "lr": LR_NEW_LAYERS},
    {"params": model.fc1.parameters(),         "lr": LR_NEW_LAYERS},
    {"params": model.head_main.parameters(),   "lr": LR_NEW_LAYERS},
    {"params": model.head_machine.parameters(),"lr": LR_NEW_LAYERS},
    {"params": model.head_fault.parameters(),  "lr": LR_NEW_LAYERS},
], weight_decay=WEIGHT_DECAY)

# Loss functions
crit_main    = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
crit_machine = nn.CrossEntropyLoss(weight=machine_w)
crit_fault   = BinaryFocalLoss(gamma=FOCAL_GAMMA, pos_weight=fault_pos_w)

# [FIX 8] Warmup + ReduceLROnPlateau (verbose removed — PyTorch ≥ 2.2 dropped it)
warmup_scheduler  = make_warmup_scheduler(optimizer, warmup_epochs=WARMUP_EPOCHS)
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=RLROP_FACTOR, patience=RLROP_PATIENCE, min_lr=1e-6
)   # [FIX 0] verbose=True removed

# ── TRAINING ──────────────────────────────────────────────────────────────────
best_val_loss = float("inf")
best_val_acc  = 0.0
es_counter    = 0
train_history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
ckpt_path     = os.path.join(MODELS_DIR, "phase2b_best.pth")

print(f"\n── Training Phase 2b v2 ──────────────────────────────────")
print(f"   Mel+Stat | Mixup α={MIXUP_ALPHA} | Focal γ={FOCAL_GAMMA} | HierAlpha={HIER_ALPHA}")
print(f"   LabelSmooth ε={LABEL_SMOOTH} | ENS β={ENS_BETA} | Warmup={WARMUP_EPOCHS}ep")
print(f"   PerMachineNorm=True | RLROP patience={RLROP_PATIENCE} | ES patience={ES_PATIENCE}")
print()

for epoch in range(1, TRAIN_EPOCHS + 1):
    current_lr = optimizer.param_groups[0]["lr"]

    tr_loss, tr_acc = train_epoch(
        model, tr_ldr, optimizer,
        crit_main, crit_machine, crit_fault,
        scalers, DEVICE,
        mixup_alpha=MIXUP_ALPHA, hier_alpha=HIER_ALPHA
    )
    vl_loss, vl_acc, _, _ = eval_epoch(model, vl_ldr, crit_main, scalers, DEVICE)

    # Scheduler step
    if epoch <= WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        plateau_scheduler.step(vl_loss)

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
        save_checkpoint(model, epoch, vl_loss, vl_acc, scalers, ckpt_path)
        tag = "  ← saved"
    else:
        es_counter += 1
        tag = f"  (patience {es_counter}/{ES_PATIENCE})"

    print(f"Epoch {epoch:3d}/{TRAIN_EPOCHS}  lr={current_lr:.2e}  "
          f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
          f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{tag}")

    if es_counter >= ES_PATIENCE:
        print(f"\n[early stop] Val loss did not improve for {ES_PATIENCE} epochs. Stopping.")
        break

print(f"\nBest val loss: {best_val_loss:.4f}  (val acc: {best_val_acc:.4f})")

# ── FINAL EVALUATION ──────────────────────────────────────────────────────────
ckpt  = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_, test_acc, test_preds, test_labels = eval_epoch(model, te_ldr, crit_main, scalers, DEVICE)
t_test = time.time() - t0
n_test = len(te_ds)
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
    "test_accuracy":  metrics["accuracy"],
    "test_macro_f1":  metrics["macro_f1"],
})
for i, f1 in enumerate(metrics["per_class_f1"]):
    wandb.log({f"f1_{CLASS_NAMES[i]}": f1})
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None, y_true=test_labels, preds=test_preds, class_names=CLASS_NAMES)
})

print(f"\n── Inference Timing ──────────────────────────────────────")
print(f"Test set size         : {n_test} samples")
print(f"Total inference time  : {t_test:.2f} s")
print(f"Per-sample time       : {ms_per_sample:.3f} ms  →  {1000/ms_per_sample:.0f} samples/sec")
print(f"Estimated   100 files : {ms_per_sample *   100 / 1000:.2f} s")
print(f"Estimated 1 000 files : {ms_per_sample *  1000 / 1000:.2f} s")
print(f"Estimated 10 000 files: {ms_per_sample * 10000 / 1000:.2f} s")

plot_confusion_matrix(test_preds, test_labels, CLASS_NAMES)

# Save scaler for inference
scaler_path = os.path.join(MODELS_DIR, "stat_scaler_2b.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump({"scalers": scalers, "features": STAT_FEATURES}, f)
print(f"Per-machine scaler saved → {scaler_path}")

# ── TRAINING CURVES ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_history["loss"],     label="train")
ax1.plot(train_history["val_loss"], label="val")
ax1.set_title("Loss"); ax1.legend()
ax2.plot(train_history["acc"],      label="train")
ax2.plot(train_history["val_acc"],  label="val")
ax2.set_title("Accuracy"); ax2.legend()
plt.suptitle("Phase 2b v2 — Mel + Statistical Features")
plt.tight_layout(); plt.show()

print(f"\nFiles saved: phase2b_best.pth, stat_scaler_2b.pkl → /kaggle/working/")

# ── ARCHIVE ───────────────────────────────────────────────────────────────────
for folder, archive in [("feats_mel", "feats_mel_archive"),
                         ("feats_stat_v2", "feats_stat_v2_archive")]:
    src = pathlib.Path("/kaggle/working") / folder
    if src.exists() and any(src.glob("*.npy")):
        shutil.make_archive(f"/kaggle/working/{archive}", "zip", "/kaggle/working", folder)
        sz = os.path.getsize(f"/kaggle/working/{archive}.zip") / 1e9
        print(f"{archive}.zip  ({sz:.3f} GB)  → /kaggle/working/")

shutil.make_archive("/kaggle/working/output", "zip", "/kaggle/working")
print("Zipped everything to /kaggle/working/output.zip")