"""
Phase 1 — training improvements ablation

Trains 5 configs on the same split and data so results are comparable.
Each config adds one cluster of changes on top of the previous.
Prints a per-class F1 table at the end.

  A  baseline     flat head, CE (uniform weights), cosine LR
  B  +ens+ls      ENS class weights + label smoothing ε=0.1 + warmup
  C  +mixup       B + Mixup α=0.1
  D  +hier+focal  C + hierarchical heads + focal loss on fault head
  E  +rlrop       D + ReduceLROnPlateau replacing cosine
"""

import os, json, math, pathlib, random, time
from dataclasses import dataclass
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
from dataclasses import dataclass, field
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
            (all_train if asgn=="train" else all_val if asgn=="val" else all_test).append(gidx)
    return all_train, all_val, all_test

def _create_clean_split(paths, labels, split_dir, train_r=0.70, val_r=0.15):
    split_dir = pathlib.Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    tr, va, te = _build_chronological_split(paths, labels, train_r, val_r)
    print(f"[split] Train={len(tr)}  Val={len(va)}  Test={len(te)}")
    result = {"train": tr, "val": va, "test": te}
    json.dump(result, open(split_dir / "split_indices_clean.json", "w"))
    print(f"[split] Saved → {split_dir / 'split_indices_clean.json'}")
    return result

# ── Label maps ────────────────────────────────────────────────────────────────

_LABEL_MAP = {
    ("machine1","Normal"):0, ("machine1","Abnormal"):1,
    ("machine2","Normal"):2, ("machine2","Abnormal"):3,
    ("machine3","Normal"):4, ("machine3","Abnormal"):5,
}
_MACHINE_FROM_CLASS = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2}
_FAULT_FROM_CLASS   = {0:0, 1:1, 2:0, 3:1, 4:0, 5:1}

CLASS_NAMES   = ["Machine1_Normal","Machine1_Abnormal",
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
        raise RuntimeError(f"No labelled .wav files found under {root_dir}.")
    print(f"Found {len(paths)} files")
    return paths, labels

# ── Config ────────────────────────────────────────────────────────────────────

ROOT_DIR   = "/kaggle/input/datasets/mostafaehab41/machine-fault-dataset"
MODELS_DIR = "/kaggle/working"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SR           = 16000
DURATION_SEC = 2.75
BATCH_SIZE   = 32
MAX_EPOCHS   = 20
LR           = 1e-3
WEIGHT_DECAY = 5e-4
ES_PATIENCE  = 5
NUM_WORKERS  = 4

import wandb

# ── Feature cache ─────────────────────────────────────────────────────────────

def _feat_dir(name):
    owner = pathlib.Path("/kaggle/input/datasets/manarabdelshafy")
    if owner.exists():
        for ds in sorted(owner.iterdir()):
            if not ds.is_dir(): continue
            candidate = ds / name
            if candidate.exists() and any(candidate.glob("*.npy")):
                print(f"[cache] '{name}' found at {candidate}")
                return candidate
    working = pathlib.Path("/kaggle/working") / name
    print(f"[cache] '{name}' not found → will compute to {working}")
    return working

def _copy_if_input(src, dst):
    import shutil
    src = pathlib.Path(src)
    if str(src).startswith("/kaggle/input"):
        dst = pathlib.Path(dst)
        if not dst.exists():
            print(f"Copying {src} → {dst}")
            shutil.copytree(src, dst, dirs_exist_ok=True)
        return dst
    return src

FEATS_DIR = _feat_dir("feats_mel")
FEATS_DIR = _copy_if_input(FEATS_DIR, "/kaggle/working/feats_mel")

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

def _minmax(S):
    lo, hi = S.min(), S.max()
    return np.zeros_like(S) if hi - lo < 1e-8 else (S - lo) / (hi - lo)

def compute_mel_spectrogram(waveform, sr=16000):
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=128, n_fft=1024,
        hop_length=512, fmin=50, fmax=8000, center=False)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.expand_dims(_minmax(mel_db).astype(np.float32), 0)

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

def _precompute_one(args):
    idx, wav_path, feats_dir, preprocessor = args
    out = pathlib.Path(feats_dir) / f"{idx:06d}.npy"
    if out.exists(): return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        np.save(out, compute_mel_spectrogram(w))
    except Exception:
        np.save(out, np.zeros((1, 128, 84), dtype=np.float32))

def precompute_all_mel(paths, feats_dir, preprocessor, n_workers=4):
    import multiprocessing, tqdm as tqdm_module
    feats_dir = pathlib.Path(feats_dir)
    feats_dir.mkdir(parents=True, exist_ok=True)
    already = sum(1 for i in range(len(paths)) if (feats_dir/f"{i:06d}.npy").exists())
    if already == len(paths):
        print(f"All {len(paths)} mel-specs cached  (skipping)"); return
    print(f"Pre-computing mel-specs for {len(paths)} files ...")
    args = [(i, p, str(feats_dir), preprocessor) for i, p in enumerate(paths)]
    with multiprocessing.Pool(n_workers) as pool:
        list(tqdm_module.tqdm(pool.imap(_precompute_one, args, chunksize=64),
                              total=len(paths), desc="mel"))
    print("Pre-computation done.")

# ── Ablation configs ──────────────────────────────────────────────────────────

@dataclass
class AblCfg:
    name:         str
    use_hier:     bool   # MelCNNHier with 3 heads; otherwise flat MelCNN
    use_ens:      bool   # ENS class weights; otherwise uniform
    label_smooth: float
    mixup_alpha:  float
    use_focal:    bool   # focal loss on fault head; otherwise plain BCE
    use_rlrop:    bool   # ReduceLROnPlateau; otherwise CosineAnnealingLR
    use_warmup:   bool

CONFIGS = [
    AblCfg("A_baseline",    use_hier=False, use_ens=False, label_smooth=0.0,
            mixup_alpha=0.0, use_focal=False, use_rlrop=False, use_warmup=False),
    AblCfg("B_ens_ls",      use_hier=False, use_ens=True,  label_smooth=0.1,
            mixup_alpha=0.0, use_focal=False, use_rlrop=False, use_warmup=True),
    AblCfg("C_mixup",       use_hier=False, use_ens=True,  label_smooth=0.1,
            mixup_alpha=0.1, use_focal=False, use_rlrop=False, use_warmup=True),
    AblCfg("D_hier_focal",  use_hier=True,  use_ens=True,  label_smooth=0.1,
            mixup_alpha=0.1, use_focal=True,  use_rlrop=False, use_warmup=True),
    AblCfg("E_rlrop",       use_hier=True,  use_ens=True,  label_smooth=0.1,
            mixup_alpha=0.1, use_focal=True,  use_rlrop=True,  use_warmup=True),
]

# ── Models ────────────────────────────────────────────────────────────────────

class FlatMelCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1,32,3,padding=1),   nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1),  nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)))
        self.fc1     = nn.Linear(256*4*4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        return self.fc2(self.dropout(F.relu(self.fc1(torch.flatten(x, 1)))))


class HierMelCNN(nn.Module):
    """Three output heads — head_main for final prediction, head_machine and
    head_fault are auxiliary losses that force the backbone to learn both
    machine identity and fault status at the same time."""
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

# ── Loss / weights ────────────────────────────────────────────────────────────

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        t = targets.float().unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, t,
                                                  pos_weight=self.pos_weight, reduction="none")
        focal = (1.0 - torch.exp(-bce)) ** self.gamma * bce
        return focal.mean()

def effective_num_weights(label_counts, beta=0.9999, num_classes=6):
    eff = (1.0 - np.power(beta, label_counts)) / (1.0 - beta)
    w = 1.0 / eff
    return torch.tensor(w / w.sum() * num_classes, dtype=torch.float32)

def build_losses(cfg, label_counts, device):
    if cfg.use_ens:
        cw = effective_num_weights(label_counts).to(device)
    else:
        cw = torch.ones(6, dtype=torch.float32).to(device)

    crit_main = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.label_smooth)
    crit_machine = crit_fault = None

    if cfg.use_hier:
        mc = np.array([label_counts[0]+label_counts[1],
                       label_counts[2]+label_counts[3],
                       label_counts[4]+label_counts[5]], dtype=np.float64)
        me = (1.0 - np.power(0.9999, mc)) / (1.0 - 0.9999)
        mw = torch.tensor(1.0/me / (1.0/me).sum() * 3, dtype=torch.float32).to(device)
        crit_machine = nn.CrossEntropyLoss(weight=mw)

        fc = np.array([sum(label_counts[i] for i in [0,2,4]),
                       sum(label_counts[i] for i in [1,3,5])], dtype=np.float64)
        fpw = torch.tensor([fc[0]/(fc[1]+1e-6)], dtype=torch.float32).to(device)
        crit_fault = (BinaryFocalLoss(gamma=2.0, pos_weight=fpw) if cfg.use_focal
                      else nn.BCEWithLogitsLoss(pos_weight=fpw))

    return crit_main, crit_machine, crit_fault

def build_schedulers(cfg, optimizer):
    warmup = None
    if cfg.use_warmup:
        warmup = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: (e+1)/2 if e < 2 else 1.0)
    if cfg.use_rlrop:
        main = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)
    else:
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    return warmup, main

# ── Mixup ─────────────────────────────────────────────────────────────────────

def mixup_batch(x, y, alpha, device):
    if alpha <= 0: return x, y, y, 1.0
    lam  = float(np.random.beta(alpha, alpha))
    lam  = max(lam, 1.0 - lam)
    perm = torch.randperm(x.size(0), device=device)
    return lam*x + (1-lam)*x[perm], y, y[perm], lam

def mixup_loss(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1-lam) * crit(pred, yb)

# ── Dataset ───────────────────────────────────────────────────────────────────

class MelDataset(Dataset):
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
        if self.augment: mel = spec_augment(mel)
        return (mel,
                torch.tensor(lbl,                      dtype=torch.long),
                torch.tensor(_MACHINE_FROM_CLASS[lbl], dtype=torch.long),
                torch.tensor(_FAULT_FROM_CLASS[lbl],   dtype=torch.long))

# ── Train / eval ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, cfg,
                crit_main, crit_machine, crit_fault, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for mel, y_main, y_machine, y_fault in loader:
        mel, y_main = mel.to(device), y_main.to(device)
        y_machine, y_fault = y_machine.to(device), y_fault.to(device)

        mel_mix, ya, yb, lam = mixup_batch(mel, y_main, cfg.mixup_alpha, device)
        optimizer.zero_grad()

        if cfg.use_hier:
            out_main, out_mach, out_fault = model(mel_mix)
            loss = (mixup_loss(crit_main, out_main, ya, yb, lam)
                    + 0.4 * crit_machine(out_mach, y_machine)
                    + 0.6 * crit_fault(out_fault, y_fault))
        else:
            out_main = model(mel_mix)
            loss = mixup_loss(crit_main, out_main, ya, yb, lam)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (out_main.argmax(1) == ya).sum().item()
        total      += y_main.size(0)
    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, crit_main, device, use_hier):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, labels = [], []
    with torch.no_grad():
        for mel, y_main, y_machine, y_fault in loader:
            mel, y_main = mel.to(device), y_main.to(device)
            out = model(mel)
            out_main = out[0] if use_hier else out
            loss = crit_main(out_main, y_main)
            p = out_main.argmax(1)
            total_loss += loss.item()
            correct    += (p == y_main).sum().item()
            total      += y_main.size(0)
            preds.extend(p.cpu().numpy())
            labels.extend(y_main.cpu().numpy())
    return total_loss / len(loader), correct / total, preds, labels


def run_training(model, tr_ldr, vl_ldr, optimizer, cfg,
                 crit_main, crit_machine, crit_fault,
                 warmup_sched, main_sched, device, ckpt_path, run):
    best_loss = float("inf")
    es = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        lr = optimizer.param_groups[0]["lr"]
        tr_l, tr_a = train_epoch(model, tr_ldr, optimizer, cfg,
                                  crit_main, crit_machine, crit_fault, device)
        vl_l, vl_a, _, _ = eval_epoch(model, vl_ldr, crit_main, device, cfg.use_hier)

        if warmup_sched and epoch <= 2:
            warmup_sched.step()
        elif cfg.use_rlrop:
            main_sched.step(vl_l)
        else:
            main_sched.step()

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
precompute_all_mel(ALL_PATHS, FEATS_DIR, _infer_prep, n_workers=NUM_WORKERS)

_splits = _create_clean_split(ALL_PATHS, ALL_LABELS, MODELS_DIR)
label_counts = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)

print("\nClass counts in training split:")
for i, (n, c) in enumerate(zip(CLASS_NAMES, label_counts)):
    print(f"  [{i}] {n:<25s} n={c}")

tr_ds = MelDataset(FEATS_DIR, ALL_LABELS, _splits["train"], augment=True)
vl_ds = MelDataset(FEATS_DIR, ALL_LABELS, _splits["val"],   augment=False)
te_ds = MelDataset(FEATS_DIR, ALL_LABELS, _splits["test"],  augment=False)
print(f"Train: {len(tr_ds)}  Val: {len(vl_ds)}  Test: {len(te_ds)}")

tr_ldr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=True)
vl_ldr = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)
te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)

results = []

for cfg in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Config: {cfg.name}")
    print(f"  hier={cfg.use_hier}  ens={cfg.use_ens}  ls={cfg.label_smooth}"
          f"  mixup={cfg.mixup_alpha}  focal={cfg.use_focal}"
          f"  rlrop={cfg.use_rlrop}  warmup={cfg.use_warmup}")
    print(f"{'='*60}")

    run = wandb.init(
        project="machine-fault-phase1-ablation",
        name=cfg.name,
        config={k: v for k, v in vars(cfg).items()},
    )

    model = (HierMelCNN if cfg.use_hier else FlatMelCNN)(num_classes=6).to(DEVICE)
    crit_main, crit_machine, crit_fault = build_losses(cfg, label_counts, DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup_sched, main_sched = build_schedulers(cfg, optimizer)

    ckpt = os.path.join(MODELS_DIR, f"phase1_abl_{cfg.name}.pth")
    run_training(model, tr_ldr, vl_ldr, optimizer, cfg,
                 crit_main, crit_machine, crit_fault,
                 warmup_sched, main_sched, DEVICE, ckpt, run)

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    _, _, test_preds, test_labels = eval_epoch(model, te_ldr, crit_main, DEVICE, cfg.use_hier)

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
