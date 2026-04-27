"""
Phase 4b — Mel + MFCC + [rms, zcr, rolloff, bandwidth, kurtosis]

Builds on Phase 3 with two changes:
  1. Centroid dropped — Phase 3 ablation showed it was the only feature
     that made accuracy go DOWN when added alone.
  2. Kurtosis added — classic fault indicator. Faulty machines produce
     impulsive signal bursts that push kurtosis way above a healthy baseline.

No ablation this time. Phase 3 already told us which features help.
We go straight to fine-tuning on the fixed 5-feature set.

Weight transfer from phase2b_best.pth:
  mel_stream  — transferred (already fine-tuned jointly with stat features)
  stat_branch — transferred (same 5 features [rms,zcr,rolloff,bw,kurtosis], same order)
  fc2         — transferred (same shape 256→6, warm class head)
  mfcc_stream — fresh init (new stream, not in Phase 2b)
  fc1         — fresh init (288→256 in Phase 2b vs 416→256 here — shape mismatch)

Feature cache:
  feats_mel/    — reused from Phase 2b upload
  feats_mfcc/   — NEW or reused from any prior phase that computed it
  feats_stat_v2/ — reused from Phase 2b upload (same 5-element format)
                   [rms, zcr, rolloff, bandwidth, kurtosis]

Before running:
  1. Add the machine-fault dataset (same as before)
  2. Upload phase2b_best.pth as a dataset (e.g. "phase2b-best-pth")
  3. Upload feats_mel + feats_stat_v2 archives from Phase 2b
     (feats_mfcc will be computed fresh if not already cached — ~15 min)

Output: phase4b_best.pth + stat_scaler_4b.pkl → /kaggle/working/
"""

import os, json, math, pathlib, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import pickle
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import soundfile as sf
from scipy.signal import resample_poly
from scipy.stats import kurtosis as scipy_kurtosis

# ---------- paths ----------------------------------------------------------------

_DATASET_BASE = "/kaggle/input/datasets/mostafaehab41/machine-fault-dataset"

def _find_machine_root(base):
    p = pathlib.Path(base)
    if any((p / f"Machine {i}").exists() for i in range(1, 4)):
        return str(p)
    for sub in sorted(p.rglob("Machine 1")):
        return str(sub.parent)
    print(f"WARNING: couldn't find Machine 1/2/3 under {base}, using base path")
    return str(p)

ROOT_DIR    = _find_machine_root(_DATASET_BASE)
PHASE2B_CKPT = "/kaggle/input/datasets/manarabdelshafy/phase2b-best-pth/phase2b_best.pth"
MODELS_DIR   = "/kaggle/working"

print(f"ROOT_DIR     : {ROOT_DIR}")
print(f"PHASE2B_CKPT : {PHASE2B_CKPT}  (exists: {pathlib.Path(PHASE2B_CKPT).exists()})")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR         = 16000
DURATION   = 2.75
BATCH_SIZE = 32
WORKERS    = 2   # 2 is enough for .npy loads; 4 workers on Kaggle's 4-CPU box
                 # compete with the main process and cause persistent_workers deadlocks

print("Device:", DEVICE)

# ---------- feature cache --------------------------------------------------------
# Scans any dataset uploaded under manarabdelshafy for the named subfolder.
# If not found, returns a path in /kaggle/working to compute into.

def _feat_dir(name):
    owner = pathlib.Path("/kaggle/input/datasets/manarabdelshafy")
    if owner.exists():
        for ds in sorted(owner.iterdir()):
            if not ds.is_dir():
                continue
            candidate = ds / name
            if candidate.exists() and any(candidate.glob("*.npy")):
                print(f"[cache] {name} found at {candidate}")
                return candidate
    dest = pathlib.Path("/kaggle/working") / name
    print(f"[cache] {name} not found — will compute to {dest}")
    return dest

FEATS_MEL    = _feat_dir("feats_mel")
FEATS_MFCC   = _feat_dir("feats_mfcc")
FEATS_STAT_V2 = _feat_dir("feats_stat_v2")   # 5-element: rms, zcr, rolloff, bandwidth, kurtosis

TRAIN_EPOCHS = 25   # single run, no ablation — differential LRs handle the new branch
CLASS_NAMES  = ["Machine1_Normal", "Machine1_Abnormal",
                "Machine2_Normal", "Machine2_Abnormal",
                "Machine3_Normal", "Machine3_Abnormal"]

STAT_FEATURES = ["rms", "zcr", "rolloff", "bandwidth", "kurtosis"]
STAT_COL_V2   = {"rms": 0, "zcr": 1, "rolloff": 2, "bandwidth": 3, "kurtosis": 4}

# ---------- preprocessing --------------------------------------------------------
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

    def preprocess(self, audio_path, duration_sec=None, target_sr=None, mode="inference", seed=None):
        cfg = self.config
        eff_sr  = int(target_sr or cfg.target_sr)
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
            w = resample_poly(w, eff_sr // d, orig_sr // d).astype(np.float32)
        if cfg.trim_silence and w.size > 0:
            peak = np.abs(w).max()
            if peak > EPSILON:
                thr = peak * cfg.silence_threshold_ratio
                fl  = max(1, int(eff_sr * cfg.trim_frame_ms / 1000))
                hl  = max(1, int(eff_sr * cfg.trim_hop_ms / 1000))
                aw  = np.abs(w)
                active = [s for s in range(0, w.size - fl + 1, hl) if aw[s:s+fl].max() >= thr]
                if active:
                    trimmed = w[active[0]: min(w.size, active[-1] + fl)]
                    if trimmed.size >= int(cfg.min_retained_sec * eff_sr):
                        w = trimmed.astype(np.float32)
        p = np.abs(w).max()
        if p > EPSILON:
            w = w * (cfg.peak_target / p)
        rng = np.random.default_rng(seed) if mode == "train" else None
        if mode == "train" and cfg.augmentation.enabled and w.size > 0:
            aug = cfg.augmentation
            if rng.random() < aug.noise_prob:
                sig_rms = np.sqrt(np.mean(w ** 2))
                if sig_rms > EPSILON:
                    snr   = rng.uniform(aug.noise_snr_db_min, aug.noise_snr_db_max)
                    noise = rng.normal(0, 1, w.shape).astype(np.float32)
                    n_rms = np.sqrt(np.mean(noise ** 2))
                    if n_rms > EPSILON:
                        w = w + noise * (sig_rms / (10 ** (snr / 20)) / (n_rms + EPSILON))
            if rng.random() < aug.time_shift_prob:
                ms = int(round(aug.time_shift_max_sec * eff_sr))
                if ms > 0:
                    w = np.roll(w, int(rng.integers(-ms, ms + 1)))
        cur = w.size
        if cur > tgt_len:
            start = (int(rng.integers(0, cur - tgt_len + 1))
                     if mode == "train" and rng is not None else (cur - tgt_len) // 2)
            w = w[start: start + tgt_len]
        elif cur < tgt_len:
            pad = tgt_len - cur
            lp  = int(rng.integers(0, pad + 1)) if mode == "train" and rng is not None else 0
            w   = np.pad(w, (lp, pad - lp), mode="constant")
        w = np.nan_to_num(w, 0.0)
        np.clip(w, -cfg.clip_value, cfg.clip_value, out=w)
        return w.astype(np.float32)

# ---------- feature functions ----------------------------------------------------

def _minmax(S):
    lo, hi = S.min(), S.max()
    return np.zeros_like(S) if hi - lo < 1e-8 else (S - lo) / (hi - lo)

def compute_mel(w, sr=16000):
    mel = librosa.feature.melspectrogram(
        y=w, sr=sr, n_mels=128, n_fft=1024, hop_length=512, fmin=50, fmax=8000, center=False)
    return np.expand_dims(_minmax(librosa.power_to_db(mel, ref=np.max)).astype(np.float32), 0)

def compute_mfcc(w, sr=16000):
    mfcc = librosa.feature.mfcc(y=w, sr=sr, n_mfcc=40, n_fft=1024, hop_length=512)
    d    = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)
    feats = np.stack([mfcc, d, d2], axis=0)
    for i in range(3):
        feats[i] = _minmax(feats[i])
    return feats.astype(np.float32)

def compute_stat_v2(w, sr=16000):
    """5 features: [rms, zcr, rolloff, bandwidth, kurtosis]. No centroid — redundant with mel CNN."""
    return np.array([
        float(np.sqrt(np.mean(w ** 2))),
        float(librosa.feature.zero_crossing_rate(w).mean()),
        float(librosa.feature.spectral_rolloff(y=w, sr=sr).mean()),
        float(librosa.feature.spectral_bandwidth(y=w, sr=sr).mean()),
        float(scipy_kurtosis(w, fisher=True)),   # excess kurtosis; healthy ≈ 0, faulty spikes high
    ], dtype=np.float32)

def spec_augment(mel, freq_mask=30, time_mask=15, n_freq=2, n_time=2):
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

# ---------- precompute -----------------------------------------------------------
# feats_mel and feats_mfcc are reused from Phase 3 (already uploaded).
# feats_stat_v2 is new — adds kurtosis. Computing it takes ~15 min on the full dataset.

def _worker_v2(args):
    idx, wav_path, mel_dir, mfcc_dir, stat_dir, prep = args
    mel_out  = pathlib.Path(mel_dir)  / f"{idx:06d}.npy"
    mfcc_out = pathlib.Path(mfcc_dir) / f"{idx:06d}.npy"
    stat_out = pathlib.Path(stat_dir) / f"{idx:06d}.npy"
    if mel_out.exists() and mfcc_out.exists() and stat_out.exists():
        return
    try:
        w = prep.preprocess(str(wav_path), mode="inference")
        if not mel_out.exists():  np.save(mel_out,  compute_mel(w))
        if not mfcc_out.exists(): np.save(mfcc_out, compute_mfcc(w))
        if not stat_out.exists(): np.save(stat_out, compute_stat_v2(w))
    except Exception:
        if not mel_out.exists():  np.save(mel_out,  np.zeros((1, 128, 84), dtype=np.float32))
        if not mfcc_out.exists(): np.save(mfcc_out, np.zeros((3, 40, 84),  dtype=np.float32))
        if not stat_out.exists(): np.save(stat_out, np.zeros(5,             dtype=np.float32))

def precompute(paths, mel_dir, mfcc_dir, stat_dir, prep, n_workers=4):
    import multiprocessing, tqdm as tqdm_mod
    mel_dir  = pathlib.Path(mel_dir)
    mfcc_dir = pathlib.Path(mfcc_dir)
    stat_dir = pathlib.Path(stat_dir)

    # If all three come from an uploaded dataset, nothing to do
    if all(str(d).startswith("/kaggle/input") for d in [mel_dir, mfcc_dir, stat_dir]):
        print("All feature dirs are from uploaded datasets — skipping precompute")
        return

    for d in [mel_dir, mfcc_dir, stat_dir]:
        if not str(d).startswith("/kaggle/input"):
            d.mkdir(parents=True, exist_ok=True)

    done = sum(1 for i in range(len(paths))
               if (mel_dir/f"{i:06d}.npy").exists()
               and (mfcc_dir/f"{i:06d}.npy").exists()
               and (stat_dir/f"{i:06d}.npy").exists())
    if done == len(paths):
        print(f"All {len(paths)} files already cached — nothing to compute")
        return

    # feats_stat_v2 is the only new cache; mel+mfcc skip automatically per file
    print(f"Computing features for {len(paths)} files ({n_workers} workers)...")
    print("mel+mfcc will skip if already cached; only stat_v2 needs new computation (~15 min)")
    args = [(i, p, str(mel_dir), str(mfcc_dir), str(stat_dir), prep)
            for i, p in enumerate(paths)]
    with multiprocessing.Pool(n_workers) as pool:
        list(tqdm_mod.tqdm(pool.imap(_worker_v2, args, chunksize=64),
                           total=len(paths), desc="features"))
    print("Done.")

# ---------- dataset --------------------------------------------------------------

class MachineScanner(Dataset):
    """Just scans the folder and builds the split — not used for actual training."""
    LABEL_MAP = {("machine1", "Normal"): 0, ("machine1", "Abnormal"): 1,
                 ("machine2", "Normal"): 2, ("machine2", "Abnormal"): 3,
                 ("machine3", "Normal"): 4, ("machine3", "Abnormal"): 5}

    def __init__(self, root_dir, split):
        self.root_dir = pathlib.Path(root_dir)
        self.paths, self.labels = self._scan()
        self.indices = self._split(split)

    def _scan(self):
        paths, labels = [], []
        for f in self.root_dir.rglob("*.wav"):
            lbl = self.LABEL_MAP.get((f.parent.parent.name, f.parent.name))
            if lbl is not None:
                paths.append(f); labels.append(lbl)
        if not paths:
            raise RuntimeError(f"No labelled wav files found under {self.root_dir}")
        print(f"Found {len(paths)} files")
        return paths, labels

    def _split(self, split):
        sf_path = pathlib.Path(MODELS_DIR) / "split_indices.json"
        if sf_path.exists():
            return json.load(open(sf_path))[split]
        idx = list(range(len(self.paths)))
        tr, tmp, _, tl = train_test_split(idx, self.labels, test_size=0.30,
                                          stratify=self.labels, random_state=42)
        va, te = train_test_split(tmp, test_size=0.50, stratify=tl, random_state=42)
        json.dump({"train": tr, "val": va, "test": te}, open(sf_path, "w"))
        print(f"Split saved → {sf_path}")
        return {"train": tr, "val": va, "test": te}[split]

    def __len__(self):  return len(self.indices)
    def __getitem__(self, i): pass   # training uses CachedDataset below


class CachedDataset(Dataset):
    """Loads mel + mfcc + stat from .npy files. No wav reads during training."""
    def __init__(self, mel_dir, mfcc_dir, stat_dir, labels, indices, stat_cols, augment=False):
        self.mel_dir   = pathlib.Path(mel_dir)
        self.mfcc_dir  = pathlib.Path(mfcc_dir)
        self.stat_dir  = pathlib.Path(stat_dir)
        self.labels    = labels
        self.indices   = indices
        self.stat_cols = stat_cols
        self.augment   = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        ri   = self.indices[i]
        mel  = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        mfcc = torch.tensor(np.load(self.mfcc_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment:
            mel = spec_augment(mel)
        # Slice the requested columns from the 6-element stat_v2 vector
        stat = np.load(self.stat_dir / f"{ri:06d}.npy")[self.stat_cols].astype(np.float32)
        return (mel, mfcc, torch.tensor(stat)), torch.tensor(self.labels[ri], dtype=torch.long)

def collate3(batch):
    feats, labels = zip(*batch)
    return (torch.stack([f[0] for f in feats]),
            torch.stack([f[1] for f in feats]),
            torch.stack([f[2] for f in feats])), torch.stack(labels)

# ---------- model ----------------------------------------------------------------

class MelCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.fc1     = nn.Linear(256 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, 6)

    def extract_features(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        return self.dropout(F.relu(self.fc1(torch.flatten(x, 1))))

    def forward(self, x):
        return self.fc2(self.extract_features(x))

class MFCCStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)))
        self.fc = nn.Linear(64 * 4 * 4, 128)

    def forward(self, x):
        return F.relu(self.fc(torch.flatten(self.features(x), 1)))

class MelMFCCStatCNN(nn.Module):
    def __init__(self, num_classes=6, stat_dim=5):
        super().__init__()
        self.mel_stream  = MelCNN(num_classes)
        self.mfcc_stream = MFCCStream()
        # stat_branch: maps stat_dim scalars → 32-d embedding
        # 256 (mel) + 128 (mfcc) + 32 (stat) = 416 going into fc1
        self.stat_branch = nn.Sequential(nn.Linear(stat_dim, 64), nn.ReLU(),
                                         nn.Linear(64, 32), nn.ReLU())
        self.fc1     = nn.Linear(256 + 128 + 32, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, mel, mfcc, stat):
        f = torch.cat([self.mel_stream.extract_features(mel),
                       self.mfcc_stream(mfcc),
                       self.stat_branch(stat)], dim=1)
        return self.fc2(self.dropout(F.relu(self.fc1(f))))

# ---------- training helpers -----------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, s_mean, s_std):
    model.train()
    sm = torch.tensor(s_mean, dtype=torch.float32, device=device)
    ss = torch.tensor(s_std,  dtype=torch.float32, device=device)
    total_loss, correct, total = 0.0, 0, 0
    for (mel, mfcc, stat), y in loader:
        mel, mfcc, stat, y = mel.to(device), mfcc.to(device), stat.to(device), y.to(device)
        stat = (stat - sm) / ss
        optimizer.zero_grad()
        out  = model(mel, mfcc, stat)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / len(loader), correct / total

def eval_one_epoch(model, loader, criterion, device, s_mean, s_std):
    model.eval()
    sm = torch.tensor(s_mean, dtype=torch.float32, device=device)
    ss = torch.tensor(s_std,  dtype=torch.float32, device=device)
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for (mel, mfcc, stat), y in loader:
            mel, mfcc, stat, y = mel.to(device), mfcc.to(device), stat.to(device), y.to(device)
            stat  = (stat - sm) / ss
            out   = model(mel, mfcc, stat)
            loss  = criterion(out, y)
            preds = out.argmax(1)
            total_loss += loss.item()
            correct    += (preds == y).sum().item()
            total      += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels

def load_phase2b_weights(model, ckpt_path, device):
    """Transfer mel_stream, stat_branch, and fc2 from the Phase 2b checkpoint.

    Phase 2b (Mel + Stat, 99.93%) trained mel_stream jointly with the exact
    same 5-feature stat set we use here — so both mel_stream and stat_branch
    are already well-calibrated for this feature set.

    What transfers:
      mel_stream  ✓  — fine-tuned with stat features, keeps that context
      stat_branch ✓  — same 5 features [rms,zcr,rolloff,bw,kurtosis], same order
      fc2         ✓  — same shape [256→6], warm class head

    What does NOT transfer:
      mfcc_stream   — doesn't exist in Phase 2b (fresh init, high LR)
      fc1           — Phase 2b: Linear(288→256); Phase 4b: Linear(416→256)
                      shape mismatch because mfcc adds 128-d to the concat
    """
    sd = torch.load(ckpt_path, map_location=device)["model_state_dict"]
    model.mel_stream.load_state_dict(
        {k[len("mel_stream."):]: v for k, v in sd.items() if k.startswith("mel_stream.")})
    model.stat_branch.load_state_dict(
        {k[len("stat_branch."):]: v for k, v in sd.items() if k.startswith("stat_branch.")})
    model.fc2.load_state_dict(
        {k[len("fc2."):]: v for k, v in sd.items() if k.startswith("fc2.")})
    print("Phase 2b weights loaded: mel_stream ✓  stat_branch ✓  fc2 ✓")
    print("  mfcc_stream : fresh init (new stream — not in Phase 2b)")
    print("  fc1         : fresh init (288→256 in Phase 2b vs 416→256 here)")

def fit_scaler(stat_dir, indices, stat_cols):
    """Fast scaler fit — reads .npy files directly, no wav reads."""
    stat_dir = pathlib.Path(stat_dir)
    arr = np.stack([np.load(stat_dir / f"{i:06d}.npy")[stat_cols] for i in indices])
    return arr.mean(0), arr.std(0) + 1e-8

# ---------- main -----------------------------------------------------------------

# Scan files and build/load the train/val/test split
scanner   = MachineScanner(ROOT_DIR, "train")
ALL_PATHS  = scanner.paths
ALL_LABELS = scanner.labels

# Precompute features. feats_mel + feats_mfcc are reused from Phase 3 if uploaded.
# feats_stat_v2 gets computed now (~15 min for 56k files).
_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=SR, default_duration_sec=DURATION, trim_silence=True, normalize_mode="peak",
    augmentation=AugmentationConfig(enabled=False)))
precompute(ALL_PATHS, FEATS_MEL, FEATS_MFCC, FEATS_STAT_V2, _prep, n_workers=WORKERS)

_splits   = json.load(open(pathlib.Path(MODELS_DIR) / "split_indices.json"))
stat_cols = [STAT_COL_V2[f] for f in STAT_FEATURES]   # [0,1,2,3,4] — all 5 elements of stat_v2

# Build datasets
train_ds = CachedDataset(FEATS_MEL, FEATS_MFCC, FEATS_STAT_V2,
                         ALL_LABELS, _splits["train"], stat_cols, augment=True)
val_ds   = CachedDataset(FEATS_MEL, FEATS_MFCC, FEATS_STAT_V2,
                         ALL_LABELS, _splits["val"],   stat_cols, augment=False)
test_ds  = CachedDataset(FEATS_MEL, FEATS_MFCC, FEATS_STAT_V2,
                         ALL_LABELS, _splits["test"],  stat_cols, augment=False)

# Scaler — loads from cached .npy, runs in seconds
print("Fitting stat scaler...")
s_mean, s_std = fit_scaler(FEATS_STAT_V2, _splits["train"], stat_cols)

tr_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate3,
                    num_workers=WORKERS, pin_memory=True, persistent_workers=False, prefetch_factor=2)
vl_ldr = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate3,
                    num_workers=WORKERS, pin_memory=True, persistent_workers=False, prefetch_factor=2)
te_ldr = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate3,
                    num_workers=WORKERS, pin_memory=True, persistent_workers=False, prefetch_factor=2)

model = MelMFCCStatCNN(num_classes=6, stat_dim=len(STAT_FEATURES)).to(DEVICE)
load_phase2b_weights(model, PHASE2B_CKPT, DEVICE)

# Differential learning rates:
#   mel_stream  — transferred, already good → protect with very low LR
#   stat_branch — transferred, already good → protect with very low LR
#   mfcc_stream — fresh init → needs full LR to learn from scratch
#   fc1         — fresh init (shape changed) → needs full LR
#   fc2         — transferred but fc1 re-init shifts its input → medium LR
optimizer = torch.optim.AdamW([
    {"params": model.mel_stream.parameters(),   "lr": 5e-5},
    {"params": model.stat_branch.parameters(),  "lr": 5e-5},
    {"params": model.mfcc_stream.parameters(),  "lr": 5e-4},
    {"params": model.fc1.parameters(),          "lr": 5e-4},
    {"params": model.fc2.parameters(),          "lr": 2e-4},
], weight_decay=1e-4)

label_counts = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
criterion    = nn.CrossEntropyLoss(
    weight=torch.tensor(1.0 / (label_counts + 1), dtype=torch.float32).to(DEVICE))
scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)

print(f"\nTraining {TRAIN_EPOCHS} epochs — features: {STAT_FEATURES}")
print(f"stat_dim={len(STAT_FEATURES)}  |  base: Phase 2b  |  differential LRs active\n")

best_val = 0.0
for epoch in range(1, TRAIN_EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, tr_ldr, optimizer, criterion, DEVICE, s_mean, s_std)
    vl_loss, vl_acc, _, _ = eval_one_epoch(model, vl_ldr, criterion, DEVICE, s_mean, s_std)
    scheduler.step()
    saved = ""
    if vl_acc > best_val:
        best_val  = vl_acc
        ckpt_path = os.path.join(MODELS_DIR, "phase4b_best.pth")
        torch.save({"model_state_dict": model.state_dict(),
                    "epoch":            epoch,
                    "val_acc":          vl_acc,
                    "stat_features":    STAT_FEATURES,
                    "stat_dim":         len(STAT_FEATURES)}, ckpt_path)
        saved = "  ← saved"
    print(f"Epoch {epoch:3d}/{TRAIN_EPOCHS}  train={tr_acc:.4f}  val={vl_acc:.4f}{saved}")

print(f"\nBest val accuracy: {best_val:.4f}")

# Save the scaler alongside the model weights
pickle.dump({"mean": s_mean, "std": s_std, "features": STAT_FEATURES},
            open(os.path.join(MODELS_DIR, "stat_scaler_4b.pkl"), "wb"))
print("stat_scaler_4b.pkl saved")

# ┌────────────────────────────────────────────────────────────┐
# │  Files written to /kaggle/working/ :                       │
# │    phase4b_best.pth    — model weights                     │
# │    stat_scaler_4b.pkl  — scaler for inference              │
# │  Download from: notebook page → Output tab on kaggle.com   │
# └────────────────────────────────────────────────────────────┘

# Load best checkpoint for testing
ckpt = torch.load(os.path.join(MODELS_DIR, "phase4b_best.pth"), map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_, test_acc, preds, labels = eval_one_epoch(model, te_ldr, criterion, DEVICE, s_mean, s_std)
t_test = time.time() - t0
n_test = len(test_ds)
ms     = (t_test / n_test) * 1000

print(f"\nTest accuracy : {test_acc:.4f}")
print(f"Macro F1      : {f1_score(labels, preds, average='macro'):.4f}")
print(f"Features used : {STAT_FEATURES}")
print("\n", classification_report(labels, preds, target_names=CLASS_NAMES))

print(f"\nInference: {ms:.3f} ms/sample  ({1000/ms:.0f} samples/sec)")
print(f"  100 files  → {ms * 100 / 1000:.2f}s")
print(f" 1000 files  → {ms * 1000 / 1000:.2f}s")
print(f"10000 files  → {ms * 10000 / 1000:.2f}s")

# Phase comparison
print(f"\nPhase comparison:")
print(f"  Phase 1  Mel only                         ~99.80%   ~0.25 ms")
print(f"  Phase 2  Mel + MFCC                       ~99.91%   ~0.50 ms")
print(f"  Phase 3  Mel + MFCC + 5 stat (w/ cent.)  ~99.93%   ~0.60 ms")
print(f"  Phase 4b Mel + MFCC + 5 stat (w/ kurt.)  {test_acc:.4%}  {ms:.3f} ms")

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.ylabel("True"); plt.xlabel("Predicted")
plt.title("Phase 4b — Mel + MFCC + stat (rms, zcr, rolloff, bandwidth, kurtosis)")
plt.tight_layout(); plt.show()

# Archive any freshly computed feature folders for upload
import shutil
for folder in ["feats_mel", "feats_mfcc", "feats_stat_v2"]:
    src = pathlib.Path("/kaggle/working") / folder
    if src.exists() and any(src.glob("*.npy")):
        archive = f"/kaggle/working/{folder}_archive"
        shutil.make_archive(archive, "zip", "/kaggle/working", folder)
        sz = os.path.getsize(f"{archive}.zip") / 1e9
        print(f"{folder}_archive.zip  ({sz:.2f} GB)")
