# phase 2b V3
import os, json, math, pathlib, random, time, pickle, hashlib, shutil
from collections import defaultdict as _ddict

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

USE_WANDB = True
if USE_WANDB:
    import wandb
    wandb.login()   # set WANDB_API_KEY env var, or: wandb.login(key="YOUR_KEY")
else:
    class _WandbStub:
        def init(self, **kw): return self
        def log(self, d, **kw): pass
        def finish(self): pass
        class plot:
            @staticmethod
            def confusion_matrix(**kw): return None
    wandb = _WandbStub()
    
    
# =============================================================================
# >>>>>>>>>>>>  EDIT THESE 4 PATHS  <<<<<<<<<<<<<<
# =============================================================================
ROOT_DIR       = "/kaggle/input/datasets/mostafaehab41/machine-fault-dataset"
PHASE1_CKPT    = "/kaggle/input/datasets/salmaelhosseiny/phase1-outputs/phase1_best.pth"
SPLIT_FILE_SRC = "/kaggle/input/datasets/salmaelhosseiny/phase1-outputs/split_indices_clean.json"
MODELS_DIR     = "/kaggle/working"
# =============================================================================

_split_dst = pathlib.Path(MODELS_DIR) / "split_indices_clean.json"
if not _split_dst.exists() and pathlib.Path(SPLIT_FILE_SRC).exists():
    shutil.copy(SPLIT_FILE_SRC, _split_dst); print(f"[setup] Copied split file -> {_split_dst}")
elif _split_dst.exists(): print(f"[setup] Split file already at {_split_dst}")
else: print("[setup] Split file not found -- will create a new one")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# -- Data / Training ----------------------------------------------------------
SR = 16000; DURATION_SEC = 4.0; BATCH_SIZE = 32; TRAIN_EPOCHS = 40; NUM_WORKERS = 2

# -- Regularisation -----------------------------------------------------------
LABEL_SMOOTH = 0.05; ENS_BETA = 0.9999; MIXUP_ALPHA = 0.3
FOCAL_GAMMA  = 1.0        # fault head (BinaryFocalLoss) only
HIER_ALPHA   = 0.2        # weight of machine-ID auxiliary loss
WEIGHT_DECAY = 5e-4

# -- Learning Rates -----------------------------------------------------------
LR_MEL_STREAM = 1e-6      # very low -- gentle backbone fine-tuning after unfreeze
LR_NEW_LAYERS = 3e-4      # stat_branch, fusion fc1, and all heads

# -- Schedulers / Early Stopping ----------------------------------------------
WARMUP_EPOCHS = 0; RLROP_PATIENCE = 6; RLROP_FACTOR = 0.5
ES_PATIENCE   = 15; UNFREEZE_EPOCH = 7    # backbone unfrozen FROM this epoch

# -- Features -----------------------------------------------------------------
STAT_FEATURES = (["rms", "zcr", "rolloff", "bandwidth", "spectral_flux", "kurtosis"] +
                 [f"mfcc_{i}" for i in range(1, 14)])
STAT_DIM      = len(STAT_FEATURES)   # 19

CLASS_NAMES   = ["Machine1_Normal", "Machine1_Abnormal",
                 "Machine2_Normal", "Machine2_Abnormal",
                 "Machine3_Normal", "Machine3_Abnormal"]
MACHINE_NAMES = ["Machine1", "Machine2", "Machine3"]

# feats_mel can be reused from V2; feats_stat_v4 is new (adds 13 MFCCs)
FEATS_DIR_MEL  = pathlib.Path(MODELS_DIR) / "feats_mel"
FEATS_DIR_STAT = pathlib.Path(MODELS_DIR) / "feats_stat_v4"
print(f"STAT_DIM={STAT_DIM}  features={STAT_FEATURES}")

# -- Label maps ---------------------------------------------------------------
_LABEL_MAP = {
    ("machine1", "Normal"): 0, ("machine1", "Abnormal"): 1,
    ("machine2", "Normal"): 2, ("machine2", "Abnormal"): 3,
    ("machine3", "Normal"): 4, ("machine3", "Abnormal"): 5,
}
_MACHINE_FROM_CLASS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
_FAULT_FROM_CLASS   = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}

def _scan_wav_files(root_dir):
    paths, labels = [], []
    for wav in sorted(pathlib.Path(root_dir).rglob("*.wav")):
        lbl = _LABEL_MAP.get((wav.parent.parent.name, wav.parent.name))
        if lbl is not None: paths.append(wav); labels.append(lbl)
    if not paths: raise RuntimeError(f"No labelled .wav files under {root_dir}.")
    print(f"Found {len(paths)} files"); return paths, labels

# -- Split logic --------------------------------------------------------------
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
    by_class = _ddict(list)
    for i, lbl in enumerate(labels): by_class[lbl].append(i)
    dup_assigned = {}; all_train, all_val, all_test = [], [], []
    for cls_id in sorted(by_class):
        cls_idxs = sorted(by_class[cls_id], key=lambda i: _num_sort_key(paths[i]))
        n = len(cls_idxs); n_tr = int(train_r * n); n_va = int(val_r * n)
        for rank, gidx in enumerate(cls_idxs):
            nat = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")
            k = idx_to_key.get(gidx); asgn = dup_assigned.setdefault(k, nat) if k else nat
            (all_train if asgn == "train" else all_val if asgn == "val" else all_test).append(gidx)
    return all_train, all_val, all_test

def _load_or_create_split(paths, labels, split_dir):
    path = pathlib.Path(split_dir) / "split_indices_clean.json"
    if path.exists():
        splits = json.load(open(path))
        print(f"[split] Loaded -> Train={len(splits['train'])}  Val={len(splits['val'])}  Test={len(splits['test'])}")
        return splits
    print("[split] Creating new chronological split ...")
    tr, va, te = _build_chronological_split(paths, labels)
    result = {"train": tr, "val": va, "test": te}
    json.dump(result, open(path, "w")); print(f"[split] Saved -> {path}"); return result

# -- Audio Preprocessor -------------------------------------------------------
EPSILON = 1e-8

@dataclass
class AugmentationConfig:
    enabled: bool = True; noise_prob: float = 0.35
    noise_snr_db_min: float = 15.0; noise_snr_db_max: float = 35.0
    time_shift_prob: float = 0.30; time_shift_max_sec: float = 0.20
    random_crop_train: bool = True

@dataclass
class PreprocessConfig:
    target_sr: int = 16000; default_duration_sec: float = 2.75
    trim_silence: bool = True; silence_threshold_ratio: float = 0.02
    trim_frame_ms: int = 20; trim_hop_ms: int = 10; min_retained_sec: float = 0.25
    normalize_mode: str = "peak"; peak_target: float = 0.95; clip_value: float = 1.0
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

class AudioPreprocessor:
    def __init__(self, config=None): self.config = config or PreprocessConfig()

    def preprocess(self, audio_path, duration_sec=None, target_sr=None, mode="inference", seed=None):
        cfg = self.config
        eff_sr = int(target_sr or cfg.target_sr); eff_dur = float(duration_sec or cfg.default_duration_sec)
        tgt_len = int(round(eff_sr * eff_dur))
        try: data, orig_sr = sf.read(str(audio_path), always_2d=False, dtype="float32")
        except Exception: return np.zeros(tgt_len, dtype=np.float32)
        w = np.asarray(data, dtype=np.float32)
        if w.ndim > 1: w = w.mean(axis=1)
        if orig_sr != eff_sr:
            d = math.gcd(orig_sr, eff_sr)
            w = resample_poly(w, eff_sr // d, orig_sr // d).astype(np.float32)
        if cfg.trim_silence: w = self._trim(w, eff_sr)
        w = self._normalize(w)
        rng = np.random.default_rng(seed) if mode == "train" else None
        if mode == "train": w = self._augment(w, eff_sr, rng)
        w = self._fix_length(w, tgt_len, mode == "train", rng)
        np.clip(np.nan_to_num(w, 0.0), -cfg.clip_value, cfg.clip_value, out=w)
        return w.astype(np.float32)

    def _trim(self, w, sr):
        cfg = self.config
        if w.size == 0: return w
        peak = np.abs(w).max()
        if peak <= EPSILON: return w
        thr = peak * cfg.silence_threshold_ratio
        fl = max(1, int(sr * cfg.trim_frame_ms / 1000)); hl = max(1, int(sr * cfg.trim_hop_ms / 1000))
        active = [s for s in range(0, w.size - fl + 1, hl) if np.abs(w[s:s+fl]).max() >= thr]
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
            sig_rms = np.sqrt(np.mean(w ** 2))
            if sig_rms > EPSILON:
                snr = rng.uniform(aug.noise_snr_db_min, aug.noise_snr_db_max)
                noise = rng.normal(0, 1, w.shape).astype(np.float32)
                n_rms = np.sqrt(np.mean(noise ** 2))
                if n_rms > EPSILON: w = w + noise * (sig_rms / (10 ** (snr / 20)) / (n_rms + EPSILON))
        if rng.random() < aug.time_shift_prob:
            ms = int(round(aug.time_shift_max_sec * sr))
            if ms > 0: w = np.roll(w, int(rng.integers(-ms, ms + 1)))
        return w.astype(np.float32)

    def _fix_length(self, w, tgt, training, rng):
        cur = w.size
        if cur == tgt: return w
        aug = self.config.augmentation
        if cur > tgt:
            start = (int(rng.integers(0, cur - tgt + 1))
                     if training and aug.random_crop_train and rng is not None else (cur - tgt) // 2)
            return w[start:start + tgt]
        pad = tgt - cur
        lp = (int(rng.integers(0, pad + 1)) if training and aug.random_crop_train and rng is not None else 0)
        return np.pad(w, (lp, pad - lp), mode="constant")

# -- Feature Computation ------------------------------------------------------
def _minmax(S):
    lo, hi = S.min(), S.max()
    return np.zeros_like(S) if hi - lo < 1e-8 else (S - lo) / (hi - lo)

def compute_mel_spectrogram(waveform, sr=16000):
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128, n_fft=1024,
                                          hop_length=512, fmin=50, fmax=8000, center=False)
    return np.expand_dims(_minmax(librosa.power_to_db(mel, ref=np.max)).astype(np.float32), 0)

def compute_stat_features(waveform, sr=16000):
    """Returns 19 features: rms, zcr, rolloff, bandwidth, spectral_flux, kurtosis, mfcc_1..13"""
    from scipy.stats import kurtosis as _kurtosis
    S = np.abs(librosa.stft(waveform, n_fft=1024, hop_length=512))
    spectral_flux = float(np.mean(np.sum(np.diff(S, axis=1) ** 2, axis=0)))
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13).mean(axis=1)
    return np.concatenate([
        np.array([float(np.sqrt(np.mean(waveform ** 2))),
                  float(librosa.feature.zero_crossing_rate(waveform).mean()),
                  float(librosa.feature.spectral_rolloff(y=waveform, sr=sr).mean()),
                  float(librosa.feature.spectral_bandwidth(y=waveform, sr=sr).mean()),
                  spectral_flux, float(_kurtosis(waveform, fisher=True))], dtype=np.float32),
        mfccs.astype(np.float32)])

def spec_augment(mel, freq_mask=27, time_mask=15, n_freq=2, n_time=2):
    mel = mel.clone(); _, F, T = mel.shape
    for _ in range(n_freq):
        f = random.randint(0, freq_mask); f0 = random.randint(0, max(F - f, 1))
        mel[:, f0:f0+f, :] = 0.0
    for _ in range(n_time):
        t = random.randint(0, time_mask); t0 = random.randint(0, max(T - t, 1))
        mel[:, :, t0:t0+t] = 0.0
    return mel

# -- Pre-computation ----------------------------------------------------------
def _precompute_one(args):
    idx, wav_path, mel_dir, stat_dir, preprocessor = args
    mel_out = pathlib.Path(mel_dir)  / f"{idx:06d}.npy"
    sta_out = pathlib.Path(stat_dir) / f"{idx:06d}.npy"
    if mel_out.exists() and sta_out.exists(): return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        if not mel_out.exists(): np.save(mel_out, compute_mel_spectrogram(w))
        if not sta_out.exists(): np.save(sta_out, compute_stat_features(w))
    except Exception:
        if not mel_out.exists(): np.save(mel_out, np.zeros((1, 128, 84), dtype=np.float32))
        if not sta_out.exists(): np.save(sta_out, np.zeros(STAT_DIM, dtype=np.float32))

def precompute_all(paths, mel_dir, stat_dir, preprocessor, n_workers=4):
    import multiprocessing, tqdm as tqdm_module
    mel_dir = pathlib.Path(mel_dir); stat_dir = pathlib.Path(stat_dir)
    mel_dir.mkdir(parents=True, exist_ok=True); stat_dir.mkdir(parents=True, exist_ok=True)
    already = sum(1 for i in range(len(paths))
                  if (mel_dir/f"{i:06d}.npy").exists() and (stat_dir/f"{i:06d}.npy").exists())
    if already == len(paths): print(f"All {len(paths)} mel+stat features already cached  (skipping)"); return
    print(f"Pre-computing mel+stat for {len(paths)} files ({n_workers} workers) ...")
    args = [(i, p, str(mel_dir), str(stat_dir), preprocessor) for i, p in enumerate(paths)]
    with multiprocessing.Pool(n_workers) as pool:
        list(tqdm_module.tqdm(pool.imap(_precompute_one, args, chunksize=64), total=len(paths), desc="mel+stat"))
    print("Pre-computation done.")

# -- Per-machine scalers ------------------------------------------------------
def fit_per_machine_scalers(stat_dir, labels, indices):
    """Fit separate (mean, std) per machine using training data only."""
    stat_dir = pathlib.Path(stat_dir); machine_stats = {m: [] for m in range(3)}
    for i in indices:
        m = _MACHINE_FROM_CLASS[labels[i]]; machine_stats[m].append(np.load(stat_dir / f"{i:06d}.npy"))
    scalers = []
    for m in range(3):
        arr = np.stack(machine_stats[m]); mean = arr.mean(0); std = arr.std(0) + 1e-8
        scalers.append((mean, std)); print(f"  Machine{m+1}: n={len(machine_stats[m])}")
    return scalers

def normalise_stat_per_machine(stat, machine_ids, scalers, device):
    out = stat.clone()
    for m, (mean, std) in enumerate(scalers):
        mask = (machine_ids == m)
        if mask.any():
            out[mask] = (stat[mask] - torch.tensor(mean, dtype=torch.float32, device=device)) /                          torch.tensor(std,  dtype=torch.float32, device=device)
    return out

# -- Dataset ------------------------------------------------------------------
class PrecomputedDatasetMS(Dataset):
    def __init__(self, mel_dir, stat_dir, labels, indices, augment=False):
        self.mel_dir = pathlib.Path(mel_dir); self.stat_dir = pathlib.Path(stat_dir)
        self.labels = labels; self.indices = indices; self.augment = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri = self.indices[idx]; lbl = self.labels[ri]
        mel  = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        stat = torch.tensor(np.load(self.stat_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment: mel = spec_augment(mel)
        return (mel, stat, torch.tensor(lbl, dtype=torch.long),
                torch.tensor(_MACHINE_FROM_CLASS[lbl], dtype=torch.long),
                torch.tensor(_FAULT_FROM_CLASS[lbl],   dtype=torch.long))

def collate_ms(batch):
    mel, stat, lbl, mach, fault = zip(*batch)
    return torch.stack(mel), torch.stack(stat), torch.stack(lbl), torch.stack(mach), torch.stack(fault)

print("Data pipeline ready.")


# -- Attention Pooling --------------------------------------------------------
class AttentionPool2d(nn.Module):
    """Replaces AdaptiveAvgPool2d: learns which time-freq regions matter most."""
    def __init__(self, in_channels, out_size=(4, 4)):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1), nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1))
        self.pool = nn.AdaptiveAvgPool2d(out_size)

    def forward(self, x):
        w = torch.softmax(self.attn(x).flatten(2), dim=-1)
        return self.pool(x * w.view(x.shape[0], 1, x.shape[2], x.shape[3]))

# -- Model --------------------------------------------------------------------
class MelCNN(nn.Module):
    """Phase 1 V3 backbone -- block4 uses AttentionPool2d instead of avg-pool."""
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1,   32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                    AttentionPool2d(256, (4, 4)))
        self.fc1 = nn.Linear(256 * 4 * 4, 256); self.dropout = nn.Dropout(0.5)
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
    """mel_stream (256-d) + stat_branch (64-d) -> fusion (320-d) -> heads."""
    def __init__(self, num_classes=6, stat_dim=19):
        super().__init__()
        self.mel_stream  = MelCNN(num_classes)
        self.stat_branch = nn.Sequential(
            nn.Linear(stat_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU())
        self.fc1 = nn.Linear(256 + 64, 256); self.dropout = nn.Dropout(0.5)
        self.head_main    = nn.Linear(256, num_classes)
        self.head_machine = nn.Linear(256, 3)
        self.head_fault   = nn.Linear(256, 1)

    def forward(self, mel, stat):
        f_mel  = self.mel_stream.extract_features(mel)
        f_stat = self.stat_branch(stat)
        fused  = self.dropout(F.relu(self.fc1(torch.cat([f_mel, f_stat], dim=1))))
        return self.head_main(fused), self.head_machine(fused), self.head_fault(fused)

def load_phase1_weights(model, ckpt_path, device):
    """Load Phase 1 V3 keys into mel_stream (strict=False; new attn layers stay random)."""
    ckpt     = torch.load(ckpt_path, map_location=device)
    remapped = {"mel_stream." + k: v for k, v in ckpt["model_state_dict"].items()}
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    loaded = len(remapped) - len(unexpected)
    print(f"[ckpt] Loaded {loaded}/{len(remapped)} Phase 1 keys into mel_stream.")
    new_keys = [k for k in missing if not k.startswith("mel_stream.")]
    if new_keys: print(f"[ckpt] New layers (random init): {new_keys[:8]}{'...' if len(new_keys)>8 else ''}")

# -- Loss Functions -----------------------------------------------------------
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__(); self.gamma = gamma; self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce   = F.binary_cross_entropy_with_logits(
            logits, targets.float().unsqueeze(1), pos_weight=self.pos_weight, reduction="none")
        return ((1.0 - torch.exp(-bce)) ** self.gamma * bce).mean()

def effective_num_weights(label_counts, beta=0.9999, num_classes=6):
    eff = (1.0 - np.power(beta, label_counts)) / (1.0 - beta)
    w   = 1.0 / eff; w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

def make_warmup_scheduler(optimizer, warmup_epochs):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda ep: float(ep + 1) / float(warmup_epochs) if ep < warmup_epochs else 1.0)

# -- Training / Eval ----------------------------------------------------------
def train_epoch(model, loader, optimizer, crit_main, crit_machine, crit_fault,
                scalers, device, mixup_alpha=0.3, hier_alpha=0.2):
    model.train(); total_loss, correct, total = 0.0, 0, 0
    for mel, stat, y_main, y_machine, y_fault in loader:
        mel, stat = mel.to(device), stat.to(device)
        y_main = y_main.to(device); y_machine = y_machine.to(device); y_fault = y_fault.to(device)
        stat = normalise_stat_per_machine(stat, y_machine, scalers, device)
        if mixup_alpha > 0:
            lam  = float(np.random.beta(mixup_alpha, mixup_alpha)); lam = max(lam, 1.0 - lam)
            perm = torch.randperm(mel.size(0), device=device)
            mel_mix  = lam * mel  + (1.0 - lam) * mel[perm]
            stat_mix = lam * stat + (1.0 - lam) * stat[perm]
            y_a, y_b = y_main, y_main[perm]
        else:
            mel_mix, stat_mix, y_a, y_b, lam = mel, stat, y_main, y_main, 1.0
        optimizer.zero_grad()
        out_main, out_machine, out_fault = model(mel_mix, stat_mix)
        loss = (mixup_criterion(crit_main, out_main, y_a, y_b, lam)
                + hier_alpha * crit_machine(out_machine, y_machine)
                + (1.0 - hier_alpha) * crit_fault(out_fault, y_fault))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item(); correct += (out_main.argmax(1) == y_a).sum().item(); total += y_main.size(0)
    return total_loss / len(loader), correct / total

def eval_epoch(model, loader, crit_main, scalers, device):
    model.eval(); total_loss, correct, total = 0.0, 0, 0; all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, stat, y_main, y_machine, y_fault in loader:
            mel, stat = mel.to(device), stat.to(device)
            y_main = y_main.to(device); y_machine = y_machine.to(device)
            stat = normalise_stat_per_machine(stat, y_machine, scalers, device)
            out_main, _, _ = model(mel, stat); preds = out_main.argmax(1)
            total_loss += crit_main(out_main, y_main).item()
            correct += (preds == y_main).sum().item(); total += y_main.size(0)
            all_preds.extend(preds.cpu().numpy()); all_labels.extend(y_main.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels

def save_checkpoint(model, epoch, val_loss, val_acc, scalers, path):
    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                "val_loss": val_loss, "val_acc": val_acc,
                "stat_features": STAT_FEATURES, "stat_dim": STAT_DIM, "scalers": scalers}, path)

def compute_metrics(preds, labels):
    preds, labels = np.array(preds), np.array(labels)
    return {"accuracy": (preds == labels).mean(),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "per_class_f1": f1_score(labels, preds, average=None)}

def plot_confusion_matrix(preds, labels, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix -- Phase 2b V3"); plt.tight_layout(); plt.show()

print("Model, loss, and training functions ready.")

ALL_PATHS, ALL_LABELS = _scan_wav_files(ROOT_DIR)
_splits = _load_or_create_split(ALL_PATHS, ALL_LABELS, MODELS_DIR)
preprocessor = AudioPreprocessor()
precompute_all(ALL_PATHS, FEATS_DIR_MEL, FEATS_DIR_STAT, preprocessor, n_workers=NUM_WORKERS)
print("\n-- Fitting per-machine stat scalers --")
scalers = fit_per_machine_scalers(FEATS_DIR_STAT, ALL_LABELS, _splits["train"])

label_counts  = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
class_weights = effective_num_weights(label_counts, beta=ENS_BETA).to(DEVICE)

machine_counts = np.array([label_counts[0]+label_counts[1], label_counts[2]+label_counts[3],
                            label_counts[4]+label_counts[5]], dtype=np.float64)
machine_ens = (1.0 - np.power(ENS_BETA, machine_counts)) / (1.0 - ENS_BETA)
machine_w   = torch.tensor(1.0/machine_ens / (1.0/machine_ens).sum() * 3, dtype=torch.float32).to(DEVICE)
fault_counts = np.array([sum(label_counts[i] for i in [0,2,4]),
                          sum(label_counts[i] for i in [1,3,5])], dtype=np.float64)
fault_pos_w  = torch.tensor([fault_counts[0] / (fault_counts[1]+1e-6)], dtype=torch.float32).to(DEVICE)

print("\n-- Class counts (training split) --")
for i, (name, cnt, w) in enumerate(zip(CLASS_NAMES, label_counts, class_weights.cpu())):
    print(f"  [{i}] {name:<25s}  n={cnt:>5d}  ENS_weight={w:.4f}")

tr_ds = PrecomputedDatasetMS(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["train"], augment=True)
vl_ds = PrecomputedDatasetMS(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["val"],   augment=False)
te_ds = PrecomputedDatasetMS(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["test"],  augment=False)
print(f"\nTrain: {len(tr_ds)}  Val: {len(vl_ds)}  Test: {len(te_ds)}")

kw = dict(collate_fn=collate_ms, num_workers=NUM_WORKERS, pin_memory=True,
          persistent_workers=False, prefetch_factor=2)
tr_ldr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw)
vl_ldr = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, **kw)
te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, **kw)

model = MelStatCNN(num_classes=6, stat_dim=STAT_DIM).to(DEVICE)
load_phase1_weights(model, PHASE1_CKPT, DEVICE)

for param in model.mel_stream.parameters(): param.requires_grad = False
print(f"[freeze] Mel backbone frozen -- unfreezes at epoch {UNFREEZE_EPOCH}")

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

warmup_scheduler  = make_warmup_scheduler(optimizer, warmup_epochs=WARMUP_EPOCHS)
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=RLROP_FACTOR, patience=RLROP_PATIENCE, min_lr=1e-7)

if USE_WANDB:
    wandb.init(project="machine-fault-phase2b-v3",
               config={"version": "v3_attn_mfcc", "batch_size": BATCH_SIZE,
                       "lr_mel_stream": LR_MEL_STREAM, "lr_new_layers": LR_NEW_LAYERS,
                       "train_epochs": TRAIN_EPOCHS, "stat_dim": STAT_DIM,
                       "label_smoothing": LABEL_SMOOTH, "ens_beta": ENS_BETA,
                       "mixup_alpha": MIXUP_ALPHA, "focal_gamma": FOCAL_GAMMA,
                       "hier_alpha": HIER_ALPHA, "weight_decay": WEIGHT_DECAY,
                       "unfreeze_epoch": UNFREEZE_EPOCH, "rlrop_patience": RLROP_PATIENCE,
                       "es_patience": ES_PATIENCE})

best_val_acc = 0.0; best_val_loss = float("inf"); es_counter = 0
train_history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
ckpt_path = os.path.join(MODELS_DIR, "phase2b_v3_best.pth")

print(f"\n-- Training Phase 2b V3 --")
print(f"   Mel+Stat+MFCC | AttentionPool | Mixup a={MIXUP_ALPHA} | HierAlpha={HIER_ALPHA}")
print(f"   LabelSmooth={LABEL_SMOOTH} | ENS_beta={ENS_BETA} | Unfreeze@ep{UNFREEZE_EPOCH}")
print()

for epoch in range(1, TRAIN_EPOCHS + 1):
    current_lr = optimizer.param_groups[1]["lr"]   # new-layers LR

    if epoch == UNFREEZE_EPOCH:
        for param in model.mel_stream.parameters(): param.requires_grad = True
        print(f"[unfreeze] Mel backbone unfrozen at epoch {epoch} (mel_lr={LR_MEL_STREAM:.0e})")

    tr_loss, tr_acc = train_epoch(model, tr_ldr, optimizer, crit_main, crit_machine,
                                   crit_fault, scalers, DEVICE,
                                   mixup_alpha=MIXUP_ALPHA, hier_alpha=HIER_ALPHA)
    vl_loss, vl_acc, _, _ = eval_epoch(model, vl_ldr, crit_main, scalers, DEVICE)

    if epoch <= WARMUP_EPOCHS: warmup_scheduler.step()
    else: plateau_scheduler.step(vl_acc)

    if USE_WANDB:
        wandb.log({"epoch": epoch, "lr": current_lr, "train_loss": tr_loss,
                   "train_acc": tr_acc, "val_loss": vl_loss, "val_acc": vl_acc})

    for k, v in zip(["loss","acc","val_loss","val_acc"], [tr_loss, tr_acc, vl_loss, vl_acc]):
        train_history[k].append(v)

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc; best_val_loss = vl_loss; es_counter = 0
        save_checkpoint(model, epoch, vl_loss, vl_acc, scalers, ckpt_path); tag = "  <- saved"
    else:
        es_counter += 1; tag = f"  (patience {es_counter}/{ES_PATIENCE})"

    print(f"Epoch {epoch:3d}/{TRAIN_EPOCHS}  lr={current_lr:.2e}  "
          f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
          f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{tag}")

    if es_counter >= ES_PATIENCE:
        print(f"\n[early stop] Val acc did not improve for {ES_PATIENCE} epochs."); break

print(f"\nBest val acc: {best_val_acc:.4f}  (val loss: {best_val_loss:.4f})")
if USE_WANDB: wandb.finish()

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
for cls_name, f1_val in zip(CLASS_NAMES, metrics["per_class_f1"]):
    print(f"  {cls_name}: {f1_val:.4f}")
print("\n", classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
print(f"Per-sample inference: {ms_per_sample:.3f} ms")

plot_confusion_matrix(test_preds, test_labels, CLASS_NAMES)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_history["loss"], label="train"); ax1.plot(train_history["val_loss"], label="val")
ax1.set_title("Loss"); ax1.legend()
ax2.plot(train_history["acc"],  label="train"); ax2.plot(train_history["val_acc"],  label="val")
ax2.set_title("Accuracy"); ax2.legend()
plt.suptitle("Phase 2b V3 -- Mel + Stat + MFCC (Attention Pooling)")
plt.tight_layout(); plt.show()

scaler_path = os.path.join(MODELS_DIR, "stat_scaler_v3.pkl")
with open(scaler_path, "wb") as f: pickle.dump({"scalers": scalers, "features": STAT_FEATURES}, f)
print(f"Scaler saved -> {scaler_path}")

for folder, archive in [("feats_mel", "feats_mel_archive"), ("feats_stat_v4", "feats_stat_v4_archive")]:
    src = pathlib.Path(MODELS_DIR) / folder
    if src.exists() and any(src.glob("*.npy")):
        shutil.make_archive(f"{MODELS_DIR}/{archive}", "zip", MODELS_DIR, folder)
        sz = os.path.getsize(f"{MODELS_DIR}/{archive}.zip") / 1e9
        print(f"{archive}.zip  ({sz:.3f} GB)")

shutil.make_archive(f"{MODELS_DIR}/output_v3", "zip", MODELS_DIR)
print(f"\nAll outputs zipped -> {MODELS_DIR}/output_v3.zip")
print("Key files: phase2b_v3_best.pth, stat_scaler_v3.pkl")