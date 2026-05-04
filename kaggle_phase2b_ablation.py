"""
Phase 2b (ablation)— statistical feature ablation

Caches all 6 stat features once, then runs 5 configs that each use a different
subset. All other settings are fixed (Phase 1 V2 checkpoint, global norm,
hier heads, focal, ENS, Mixup, RLROP). Only the feature set changes.

  A  mel_only    no stat branch — Phase 1 V2 fine-tuned (baseline)
  B  basic_4     rms, zcr, rolloff, bandwidth
  C  +kurtosis   basic_4 + kurtosis
  D  +flux       basic_4 + spectral_flux
  E  all_6       rms, zcr, rolloff, bandwidth, kurtosis, spectral_flux

All configs load from the Phase 1 V2 checkpoint (flat head, M2_Abn F1=0.54 there).
Phase 1 V3 is avoided — its backbone collapsed M2_Abnormal to F1=0.02.
"""

import os, json, math, pathlib, random, time, pickle, shutil
from dataclasses import dataclass, field
from scipy.stats import kurtosis as scipy_kurtosis
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
from scipy.signal import resample_poly
import hashlib
from collections import defaultdict as _ddict

# ── Split utilities (verbatim from split_utils.py) ────────────────────────────

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

def _load_clean_split(split_dir):
    path = pathlib.Path(split_dir) / "split_indices_clean.json"
    if not path.exists():
        raise FileNotFoundError(f"split_indices_clean.json not found at {path}.")
    return json.load(open(path))

# ── Label maps ────────────────────────────────────────────────────────────────

_LABEL_MAP = {
    ("machine1","Normal"):0, ("machine1","Abnormal"):1,
    ("machine2","Normal"):2, ("machine2","Abnormal"):3,
    ("machine3","Normal"):4, ("machine3","Abnormal"):5,
}
_MACHINE_FROM_CLASS = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2}
_FAULT_FROM_CLASS   = {0:0, 1:1, 2:0, 3:1, 4:0, 5:1}

CLASS_NAMES = ["Machine1_Normal","Machine1_Abnormal",
               "Machine2_Normal","Machine2_Abnormal",
               "Machine3_Normal","Machine3_Abnormal"]

def _scan_wav_files(root_dir):
    paths, labels = [], []
    for wav in sorted(pathlib.Path(root_dir).rglob("*.wav")):
        state, machine = wav.parent.name, wav.parent.parent.name
        lbl = _LABEL_MAP.get((machine, state))
        if lbl is not None:
            paths.append(wav); labels.append(lbl)
    if not paths:
        raise RuntimeError(f"No labelled .wav files under {root_dir}.")
    print(f"Found {len(paths)} files")
    return paths, labels

# ── Config ────────────────────────────────────────────────────────────────────

ROOT_DIR    = "/kaggle/input/datasets/mostafaehab41/machine-fault-dataset"
PHASE1_CKPT = "/kaggle/input/datasets/manarabdelshafy/phase1-v2-best-pth/phase1_v2_best.pth"
MODELS_DIR  = "/kaggle/working"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SR           = 16000
DURATION_SEC = 2.75
BATCH_SIZE   = 32
MAX_EPOCHS   = 20
NUM_WORKERS  = 2
WEIGHT_DECAY = 5e-4
LABEL_SMOOTH = 0.1
ENS_BETA     = 0.9999
MIXUP_ALPHA  = 0.1
FOCAL_GAMMA  = 2.0
HIER_ALPHA   = 0.4
ES_PATIENCE  = 5
LR_MEL       = 3e-4
LR_NEW       = 5e-4

import wandb

# Full 6-feature set — cache once, slice per config
ALL_STAT = ["rms", "zcr", "rolloff", "bandwidth", "kurtosis", "spectral_flux"]

# ── Ablation configs ──────────────────────────────────────────────────────────

@dataclass
class AblCfg:
    name:     str
    features: list   # subset of ALL_STAT; empty = mel only

CONFIGS = [
    AblCfg("A_mel_only",  features=[]),
    AblCfg("B_basic4",    features=["rms","zcr","rolloff","bandwidth"]),
    AblCfg("C_kurtosis",  features=["rms","zcr","rolloff","bandwidth","kurtosis"]),
    AblCfg("D_flux",      features=["rms","zcr","rolloff","bandwidth","spectral_flux"]),
    AblCfg("E_all6",      features=["rms","zcr","rolloff","bandwidth","kurtosis","spectral_flux"]),
]

def feat_indices(features):
    return [ALL_STAT.index(f) for f in features]

# ── Feature cache helpers ─────────────────────────────────────────────────────

def _feat_dir(name):
    owner = pathlib.Path("/kaggle/input/datasets/manarabdelshafy")
    if owner.exists():
        for ds in sorted(owner.iterdir()):
            if not ds.is_dir(): continue
            cand = ds / name
            if cand.exists() and any(cand.glob("*.npy")):
                print(f"[cache] '{name}' found at {cand}")
                return cand
    working = pathlib.Path("/kaggle/working") / name
    print(f"[cache] '{name}' not found → will compute to {working}")
    return working

def _copy_if_input(src, dst):
    src = pathlib.Path(src)
    if str(src).startswith("/kaggle/input"):
        dst = pathlib.Path(dst)
        if not dst.exists():
            print(f"Copying {src} → {dst}")
            shutil.copytree(src, dst, dirs_exist_ok=True)
        return dst
    return src

FEATS_DIR_MEL  = _copy_if_input(_feat_dir("feats_mel"),     "/kaggle/working/feats_mel")
FEATS_DIR_STAT = _copy_if_input(_feat_dir("feats_stat_v3"), "/kaggle/working/feats_stat_v3")

# ── Preprocessing ─────────────────────────────────────────────────────────────

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
        return np.nan_to_num(np.clip(w, -cfg.clip_value, cfg.clip_value), 0.0).astype(np.float32)

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
        return trimmed.astype(np.float32) if trimmed.size >= int(cfg.min_retained_sec * sr) else w

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

# ── Feature extraction ────────────────────────────────────────────────────────

def _minmax(S):
    lo, hi = S.min(), S.max()
    return np.zeros_like(S) if hi - lo < 1e-8 else (S - lo) / (hi - lo)

def compute_mel(waveform, sr=16000):
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=128, n_fft=1024,
        hop_length=512, fmin=50, fmax=8000, center=False)
    return np.expand_dims(_minmax(librosa.power_to_db(mel, ref=np.max)).astype(np.float32), 0)

def compute_stat_all6(waveform, sr=16000):
    """All 6 features — cache once, slice whichever subset you need."""
    S    = np.abs(librosa.stft(waveform, n_fft=1024, hop_length=512))
    flux = float(np.mean(np.sum(np.diff(S, axis=1) ** 2, axis=0)))
    return np.array([
        float(np.sqrt(np.mean(waveform ** 2))),
        float(librosa.feature.zero_crossing_rate(waveform).mean()),
        float(librosa.feature.spectral_rolloff(y=waveform, sr=sr).mean()),
        float(librosa.feature.spectral_bandwidth(y=waveform, sr=sr).mean()),
        float(scipy_kurtosis(waveform, fisher=True)),
        flux,
    ], dtype=np.float32)

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

# ── Pre-computation ───────────────────────────────────────────────────────────

def _precompute_one(args):
    idx, wav_path, mel_dir, stat_dir, preprocessor = args
    mel_out  = pathlib.Path(mel_dir)  / f"{idx:06d}.npy"
    stat_out = pathlib.Path(stat_dir) / f"{idx:06d}.npy"
    if mel_out.exists() and stat_out.exists(): return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        if not mel_out.exists():  np.save(mel_out,  compute_mel(w))
        if not stat_out.exists(): np.save(stat_out, compute_stat_all6(w))
    except Exception:
        if not mel_out.exists():  np.save(mel_out,  np.zeros((1, 128, 84), dtype=np.float32))
        if not stat_out.exists(): np.save(stat_out, np.zeros(6, dtype=np.float32))

def precompute_all(paths, mel_dir, stat_dir, preprocessor, n_workers=4):
    import multiprocessing, tqdm as tqdm_module
    mel_dir  = pathlib.Path(mel_dir);  mel_dir.mkdir(parents=True, exist_ok=True)
    stat_dir = pathlib.Path(stat_dir); stat_dir.mkdir(parents=True, exist_ok=True)
    already = sum(1 for i in range(len(paths))
                  if (mel_dir/f"{i:06d}.npy").exists() and (stat_dir/f"{i:06d}.npy").exists())
    if already == len(paths):
        print(f"All {len(paths)} mel+stat features cached  (skipping)"); return
    print(f"Pre-computing {len(paths)} files ...")
    args = [(i, p, str(mel_dir), str(stat_dir), preprocessor) for i, p in enumerate(paths)]
    with multiprocessing.Pool(n_workers) as pool:
        list(tqdm_module.tqdm(pool.imap(_precompute_one, args, chunksize=64),
                              total=len(paths), desc="mel+stat"))
    print("Pre-computation done.")

# ── Scaler ────────────────────────────────────────────────────────────────────

def fit_scaler(stat_dir, indices, feat_idx):
    """Fit (mean, std) on the given feature columns of the training split."""
    stat_dir = pathlib.Path(stat_dir)
    arr = np.stack([np.load(stat_dir / f"{i:06d}.npy")[feat_idx] for i in indices])
    return arr.mean(0), arr.std(0) + 1e-8

# ── Models ────────────────────────────────────────────────────────────────────

class MelCNNHier(nn.Module):
    """Mel backbone + hierarchical heads.  Used for mel_only (config A)."""
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1,32,3,padding=1),   nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1),  nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)))
        self.fc1          = nn.Linear(256*4*4, 256)
        self.dropout      = nn.Dropout(0.5)
        self.head_main    = nn.Linear(256, num_classes)
        self.head_machine = nn.Linear(256, 3)
        self.head_fault   = nn.Linear(256, 1)

    def extract_features(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        return self.dropout(F.relu(self.fc1(torch.flatten(x, 1))))

    def forward(self, x):
        feat = self.extract_features(x)
        return self.head_main(feat), self.head_machine(feat), self.head_fault(feat)


class MelStatCNN(nn.Module):
    """MelCNNHier backbone + stat branch.  Used for configs B–E."""
    def __init__(self, num_classes=6, stat_dim=4):
        super().__init__()
        self.mel_stream  = MelCNNHier(num_classes)
        self.stat_branch = nn.Sequential(
            nn.Linear(stat_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.fc1          = nn.Linear(256 + 32, 256)
        self.dropout      = nn.Dropout(0.4)
        self.head_main    = nn.Linear(256, num_classes)
        self.head_machine = nn.Linear(256, 3)
        self.head_fault   = nn.Linear(256, 1)

    def forward(self, mel, stat):
        f_mel  = self.mel_stream.extract_features(mel)
        f_stat = self.stat_branch(stat)
        fused  = self.dropout(F.relu(self.fc1(torch.cat([f_mel, f_stat], dim=1))))
        return self.head_main(fused), self.head_machine(fused), self.head_fault(fused)


def load_v2_weights(model, ckpt_path, device):
    """Load Phase 1 V2 backbone into mel_stream (blocks+fc1+dropout).
    V2 has a flat fc2 head; the hier heads in mel_stream are randomly init'd.
    """
    ckpt     = torch.load(ckpt_path, map_location=device)
    p1_state = ckpt["model_state_dict"]

    if hasattr(model, "mel_stream"):
        remapped = {"mel_stream." + k: v for k, v in p1_state.items()}
        missing, _ = model.load_state_dict(remapped, strict=False)
        loaded = len(remapped) - len([k for k in missing if k.startswith("mel_stream.")])
        print(f"[ckpt] Loaded {loaded}/{len(remapped)} keys into mel_stream.")
    else:
        # mel_only: model IS the mel backbone
        missing, _ = model.load_state_dict(p1_state, strict=False)
        print(f"[ckpt] Loaded {len(p1_state)-len(missing)}/{len(p1_state)} keys.")

# ── Loss / weights ────────────────────────────────────────────────────────────

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma; self.pos_weight = pos_weight

    def forward(self, logits, targets):
        t   = targets.float().unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, t,
                                                  pos_weight=self.pos_weight, reduction="none")
        return ((1.0 - torch.exp(-bce)) ** self.gamma * bce).mean()

def effective_num_weights(label_counts, beta=0.9999, n=6):
    eff = (1.0 - np.power(beta, label_counts)) / (1.0 - beta)
    w   = 1.0 / eff
    return torch.tensor(w / w.sum() * n, dtype=torch.float32)

def build_losses(label_counts, device):
    cw = effective_num_weights(label_counts).to(device)
    mc = np.array([label_counts[0]+label_counts[1],
                   label_counts[2]+label_counts[3],
                   label_counts[4]+label_counts[5]], dtype=np.float64)
    me = (1.0 - np.power(ENS_BETA, mc)) / (1.0 - ENS_BETA)
    mw = torch.tensor(1.0/me / (1.0/me).sum() * 3, dtype=torch.float32).to(device)
    fc  = np.array([sum(label_counts[i] for i in [0,2,4]),
                    sum(label_counts[i] for i in [1,3,5])], dtype=np.float64)
    fpw = torch.tensor([fc[0]/(fc[1]+1e-6)], dtype=torch.float32).to(device)

    return (nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTH),
            nn.CrossEntropyLoss(weight=mw),
            BinaryFocalLoss(gamma=FOCAL_GAMMA, pos_weight=fpw))

# ── Mixup ─────────────────────────────────────────────────────────────────────

def mixup_batch(x, y, device):
    lam  = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
    lam  = max(lam, 1.0 - lam)
    perm = torch.randperm(x.size(0), device=device)
    return lam*x + (1-lam)*x[perm], y, y[perm], lam

def mixup_loss(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1-lam) * crit(pred, yb)

# ── Datasets ──────────────────────────────────────────────────────────────────

class MelOnlyDataset(Dataset):
    def __init__(self, mel_dir, labels, indices, augment=False):
        self.mel_dir = pathlib.Path(mel_dir)
        self.labels  = labels; self.indices = indices; self.augment = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri  = self.indices[idx]
        lbl = self.labels[ri]
        mel = torch.tensor(np.load(self.mel_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment: mel = spec_augment(mel)
        return (mel,
                torch.tensor(lbl,                      dtype=torch.long),
                torch.tensor(_MACHINE_FROM_CLASS[lbl], dtype=torch.long),
                torch.tensor(_FAULT_FROM_CLASS[lbl],   dtype=torch.long))


class MelStatDataset(Dataset):
    def __init__(self, mel_dir, stat_dir, labels, indices, feat_idx, augment=False):
        self.mel_dir   = pathlib.Path(mel_dir)
        self.stat_dir  = pathlib.Path(stat_dir)
        self.labels    = labels; self.indices = indices
        self.feat_idx  = feat_idx; self.augment = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri  = self.indices[idx]
        lbl = self.labels[ri]
        mel  = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        stat = torch.tensor(np.load(self.stat_dir / f"{ri:06d}.npy")[self.feat_idx], dtype=torch.float32)
        if self.augment: mel = spec_augment(mel)
        return (mel, stat,
                torch.tensor(lbl,                      dtype=torch.long),
                torch.tensor(_MACHINE_FROM_CLASS[lbl], dtype=torch.long),
                torch.tensor(_FAULT_FROM_CLASS[lbl],   dtype=torch.long))

# ── Train / eval ──────────────────────────────────────────────────────────────

def train_epoch_mel(model, loader, optimizer, crit_main, crit_machine, crit_fault, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for mel, y_main, y_machine, y_fault in loader:
        mel, y_main = mel.to(device), y_main.to(device)
        y_machine, y_fault = y_machine.to(device), y_fault.to(device)
        mel_mix, ya, yb, lam = mixup_batch(mel, y_main, device)
        optimizer.zero_grad()
        out_main, out_mach, out_fault = model(mel_mix)
        loss = (mixup_loss(crit_main, out_main, ya, yb, lam)
                + HIER_ALPHA * crit_machine(out_mach, y_machine)
                + (1-HIER_ALPHA) * crit_fault(out_fault, y_fault))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += (out_main.argmax(1) == ya).sum().item()
        total      += y_main.size(0)
    return total_loss / len(loader), correct / total


def train_epoch_stat(model, loader, optimizer, crit_main, crit_machine, crit_fault,
                     sm, ss, device):
    model.train()
    sm_t = torch.tensor(sm, dtype=torch.float32, device=device)
    ss_t = torch.tensor(ss, dtype=torch.float32, device=device)
    total_loss, correct, total = 0.0, 0, 0
    for mel, stat, y_main, y_machine, y_fault in loader:
        mel, stat = mel.to(device), stat.to(device)
        y_main, y_machine, y_fault = y_main.to(device), y_machine.to(device), y_fault.to(device)
        stat = (stat - sm_t) / ss_t
        mel_mix, ya, yb, lam = mixup_batch(mel, y_main, device)
        optimizer.zero_grad()
        out_main, out_mach, out_fault = model(mel_mix, stat)
        loss = (mixup_loss(crit_main, out_main, ya, yb, lam)
                + HIER_ALPHA * crit_machine(out_mach, y_machine)
                + (1-HIER_ALPHA) * crit_fault(out_fault, y_fault))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += (out_main.argmax(1) == ya).sum().item()
        total      += y_main.size(0)
    return total_loss / len(loader), correct / total


def eval_epoch_mel(model, loader, crit_main, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, labels = [], []
    with torch.no_grad():
        for mel, y_main, y_machine, y_fault in loader:
            mel, y_main = mel.to(device), y_main.to(device)
            out_main, _, _ = model(mel)
            loss = crit_main(out_main, y_main)
            p = out_main.argmax(1)
            total_loss += loss.item()
            correct    += (p == y_main).sum().item()
            total      += y_main.size(0)
            preds.extend(p.cpu().numpy())
            labels.extend(y_main.cpu().numpy())
    return total_loss / len(loader), correct / total, preds, labels


def eval_epoch_stat(model, loader, crit_main, sm, ss, device):
    model.eval()
    sm_t = torch.tensor(sm, dtype=torch.float32, device=device)
    ss_t = torch.tensor(ss, dtype=torch.float32, device=device)
    total_loss, correct, total = 0.0, 0, 0
    preds, labels = [], []
    with torch.no_grad():
        for mel, stat, y_main, y_machine, y_fault in loader:
            mel, stat, y_main = mel.to(device), stat.to(device), y_main.to(device)
            stat = (stat - sm_t) / ss_t
            out_main, _, _ = model(mel, stat)
            loss = crit_main(out_main, y_main)
            p = out_main.argmax(1)
            total_loss += loss.item()
            correct    += (p == y_main).sum().item()
            total      += y_main.size(0)
            preds.extend(p.cpu().numpy())
            labels.extend(y_main.cpu().numpy())
    return total_loss / len(loader), correct / total, preds, labels


def run_training(model, tr_ldr, vl_ldr, optimizer,
                 crit_main, crit_machine, crit_fault,
                 warmup_sched, plateau_sched, device,
                 ckpt_path, sm=None, ss=None, use_stat=False, run=None):
    best_loss = float("inf")
    es = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        lr = optimizer.param_groups[0]["lr"]
        if use_stat:
            tr_l, tr_a = train_epoch_stat(model, tr_ldr, optimizer,
                                          crit_main, crit_machine, crit_fault, sm, ss, device)
            vl_l, vl_a, _, _ = eval_epoch_stat(model, vl_ldr, crit_main, sm, ss, device)
        else:
            tr_l, tr_a = train_epoch_mel(model, tr_ldr, optimizer,
                                         crit_main, crit_machine, crit_fault, device)
            vl_l, vl_a, _, _ = eval_epoch_mel(model, vl_ldr, crit_main, device)

        if epoch <= 2:
            warmup_sched.step()
        else:
            plateau_sched.step(vl_l)

        if run:
            run.log({"epoch": epoch, "lr": lr,
                     "train_loss": tr_l, "train_acc": tr_a,
                     "val_loss": vl_l, "val_acc": vl_a})

        tag = ""
        if vl_l < best_loss:
            best_loss = vl_l; es = 0
            torch.save(model.state_dict(), ckpt_path); tag = " ✓"
        else:
            es += 1; tag = f" ({es}/{ES_PATIENCE})"

        print(f"  ep{epoch:3d}  lr={lr:.1e}  "
              f"tr={tr_l:.4f}/{tr_a:.3f}  vl={vl_l:.4f}/{vl_a:.3f}{tag}")
        if es >= ES_PATIENCE:
            print(f"  Early stop at epoch {epoch}."); break

# ── Main ──────────────────────────────────────────────────────────────────────

ALL_PATHS, ALL_LABELS = _scan_wav_files(ROOT_DIR)

_infer_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=SR, default_duration_sec=DURATION_SEC,
    trim_silence=True, normalize_mode="peak",
    augmentation=AugmentationConfig(enabled=False),
))
precompute_all(ALL_PATHS, FEATS_DIR_MEL, FEATS_DIR_STAT, _infer_prep, n_workers=NUM_WORKERS)

# Find split — try /kaggle/working first, then uploaded datasets
_split_file = pathlib.Path(MODELS_DIR) / "split_indices_clean.json"
if not _split_file.exists():
    owner = pathlib.Path("/kaggle/input/datasets/manarabdelshafy")
    if owner.exists():
        for ds in sorted(owner.iterdir()):
            cand = ds / "split_indices_clean.json"
            if cand.exists():
                shutil.copy(cand, _split_file)
                print(f"[split] Copied from {cand}")
                break
_splits = _load_clean_split(MODELS_DIR)

label_counts = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
crit_main, crit_machine, crit_fault = build_losses(label_counts, DEVICE)

print("\nClass counts in training split:")
for i, (n, c) in enumerate(zip(CLASS_NAMES, label_counts)):
    print(f"  [{i}] {n:<25s} n={c}")

results = []

for cfg in CONFIGS:
    use_stat = len(cfg.features) > 0
    fidx     = feat_indices(cfg.features) if use_stat else []
    stat_dim = len(fidx)

    print(f"\n{'='*60}")
    print(f"Config: {cfg.name}  features={cfg.features or 'none (mel only)'}")
    print(f"{'='*60}")

    run = wandb.init(
        project="machine-fault-phase2b-stat-ablation",
        name=cfg.name,
        config={"features": cfg.features, "stat_dim": stat_dim,
                "lr_mel": LR_MEL, "lr_new": LR_NEW,
                "mixup": MIXUP_ALPHA, "focal_gamma": FOCAL_GAMMA},
    )

    if use_stat:
        sm, ss = fit_scaler(FEATS_DIR_STAT, _splits["train"], fidx)
        model  = MelStatCNN(num_classes=6, stat_dim=stat_dim).to(DEVICE)
        load_v2_weights(model, PHASE1_CKPT, DEVICE)
        optimizer = torch.optim.AdamW([
            {"params": model.mel_stream.parameters(),  "lr": LR_MEL},
            {"params": model.stat_branch.parameters(), "lr": LR_NEW},
            {"params": model.fc1.parameters(),         "lr": LR_NEW},
            {"params": model.head_main.parameters(),   "lr": LR_NEW},
            {"params": model.head_machine.parameters(),"lr": LR_NEW},
            {"params": model.head_fault.parameters(),  "lr": LR_NEW},
        ], weight_decay=WEIGHT_DECAY)
        tr_ds = MelStatDataset(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS,
                               _splits["train"], fidx, augment=True)
        vl_ds = MelStatDataset(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS,
                               _splits["val"],   fidx, augment=False)
        te_ds = MelStatDataset(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS,
                               _splits["test"],  fidx, augment=False)
    else:
        sm = ss = None
        model = MelCNNHier(num_classes=6).to(DEVICE)
        load_v2_weights(model, PHASE1_CKPT, DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=LR_MEL, weight_decay=WEIGHT_DECAY)
        tr_ds = MelOnlyDataset(FEATS_DIR_MEL, ALL_LABELS, _splits["train"], augment=True)
        vl_ds = MelOnlyDataset(FEATS_DIR_MEL, ALL_LABELS, _splits["val"],   augment=False)
        te_ds = MelOnlyDataset(FEATS_DIR_MEL, ALL_LABELS, _splits["test"],  augment=False)

    tr_ldr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True)
    vl_ldr = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    warmup_sched  = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: (e+1)/2 if e < 2 else 1.0)
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    ckpt = os.path.join(MODELS_DIR, f"phase2b_abl_{cfg.name}.pth")
    run_training(model, tr_ldr, vl_ldr, optimizer,
                 crit_main, crit_machine, crit_fault,
                 warmup_sched, plateau_sched, DEVICE,
                 ckpt, sm=sm, ss=ss, use_stat=use_stat, run=run)

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    if use_stat:
        _, _, test_preds, test_labels = eval_epoch_stat(model, te_ldr, crit_main, sm, ss, DEVICE)
    else:
        _, _, test_preds, test_labels = eval_epoch_mel(model, te_ldr, crit_main, DEVICE)

    f1s   = f1_score(test_labels, test_preds, average=None, zero_division=0)
    macro = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    acc   = np.mean(np.array(test_preds) == np.array(test_labels))
    results.append({"name": cfg.name, "f1": f1s, "macro": macro, "acc": acc})

    run.log({"test_macro_f1": macro, "test_acc": acc})
    for i, f in enumerate(f1s):
        run.log({f"f1_{CLASS_NAMES[i]}": f})
    run.finish()

    print(f"\n  Test  macro_f1={macro:.4f}  acc={acc:.4f}")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

# ── Summary table ─────────────────────────────────────────────────────────────

print(f"\n{'Config':<16} {'M1N':>5} {'M1A':>5} {'M2N':>5} {'M2A':>5} {'M3N':>5} {'M3A':>5} {'Macro':>7} {'Acc':>6}")
print("-" * 68)
for r in results:
    f = r["f1"]
    print(f"{r['name']:<16} "
          f"{f[0]:>5.3f} {f[1]:>5.3f} {f[2]:>5.3f} {f[3]:>5.3f} "
          f"{f[4]:>5.3f} {f[5]:>5.3f} {r['macro']:>7.3f} {r['acc']:>6.3f}")
