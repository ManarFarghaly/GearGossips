"""
Phase 2b — Mel-Spectrogram + Statistical Features (no MFCC)

A lighter alternative to Phase 2 — skips the MFCC CNN entirely and adds
a small FC branch for 5 global statistics instead.

  Phase 1  : Mel only          ~0.24 ms/sample   99.80%
  Phase 2  : Mel + MFCC        ~0.49 ms/sample   99.91%  (2× slower)
  Phase 2b : Mel + Stat        ~0.25 ms/sample   TBD     (this script)

Features used: rms, zcr, rolloff, bandwidth, kurtosis  (no centroid)
  Centroid dropped — redundant with the mel CNN's frequency representations.
  Kurtosis added — standard fault indicator; impulsive signals spike high.

No ablation here. Phase 3 already validated which features help.

Pipeline:
  1. Precompute mel (reuse Phase 1 cache) + stat_v2 features once
  2. Direct fine-tune: 25 epochs with differential LRs
  3. Save phase2b_best.pth + stat_scaler_2b.pkl

Before running:
  1. Add machine-fault dataset (same as Phase 1)
  2. Upload phase1_best.pth as a Kaggle dataset (e.g. "phase1ckpt")
  3. Set PHASE1_CKPT below if your dataset name differs
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

# ══════════════════════════════════════════════════════════════════════════════
# SPLIT UTILITIES — verbatim copy of split_utils.py logic, standalone for Kaggle
# ══════════════════════════════════════════════════════════════════════════════
import hashlib
from collections import defaultdict as _ddict

def _num_sort_key(f):
    """Numeric-first sort: 1.wav < 2.wav < 10.wav. Non-integer names sort alphabetically."""
    p = pathlib.Path(f)
    try: return (0, int(p.stem), p.stem.lower())
    except ValueError: return (1, 0, p.stem.lower())

def _md5(path):
    """MD5 hash — only computed when file sizes match; fast in practice."""
    h = hashlib.md5()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def _find_duplicate_groups(paths):
    """Group by (size, MD5). Returns {key: [idx, ...]} for groups >= 2."""
    by_size = _ddict(list)
    for i, p in enumerate(paths): by_size[pathlib.Path(p).stat().st_size].append(i)
    groups = _ddict(list)
    for size, idxs in by_size.items():
        if len(idxs) < 2: continue
        for i in idxs: groups[f"{size}:{_md5(paths[i])}"].append(i)
    return {k: v for k, v in groups.items() if len(v) > 1}

def _build_chronological_split(paths, labels, train_r=0.70, val_r=0.15):
    """Sort numerically per class, force duplicates into the same split."""
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
    """Build, verify, and save the split. Called once from Phase 1."""
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
    """Load split_indices_clean.json. Raises FileNotFoundError if not found."""
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
    """Return (paths, labels) for all labelled .wav files under root_dir."""
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

# ── PATHS ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = "/kaggle/input/datasets/mostafaehab41/machine-fault-dataset"
PHASE1_CKPT = "/kaggle/input/datasets/manarabdelshafy/phase1-best-pth/phase1_best.pth"   # ← CHANGE if your dataset name differs
MODELS_DIR  = "/kaggle/working"

# ── FEATURE CACHE — AUTO-DETECT ──────────────────────────────────────────────
# Upload your features dataset under your account (manarabdelshafy), any name.
# The script finds feats_mel/ and feats_stat/ automatically — no config needed.

def _feat_dir(name: str) -> pathlib.Path:
    """Auto-detect feature folder from any dataset by manarabdelshafy.
    Falls back to /kaggle/working/<name> if not found."""
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

FEATS_DIR_MEL  = _feat_dir("feats_mel")      # reuse Phase 1 cache if available
FEATS_DIR_STAT = _feat_dir("feats_stat_v2")  # 5-element: rms, zcr, rolloff, bandwidth, kurtosis

def copy_if_input(src, dst):
    if str(src).startswith("/kaggle/input"):
        print(f"Copying {src} → {dst}")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return pathlib.Path(dst)
    return src

FEATS_DIR_MEL  = copy_if_input(FEATS_DIR_MEL, "/kaggle/working/feats_mel")
FEATS_DIR_STAT = copy_if_input(FEATS_DIR_STAT, "/kaggle/working/feats_stat_v2")

print(f"ROOT_DIR    : {ROOT_DIR}")
print(f"PHASE1_CKPT : {PHASE1_CKPT}")
print(f"Checkpoint exists: {pathlib.Path(PHASE1_CKPT).exists()}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SR = 16000; DURATION_SEC = 2.75; BATCH_SIZE = 32
NUM_WORKERS  = 2   # 2 is enough for .npy loads; 4 workers on Kaggle's 4-CPU box
                   # compete with the main process and cause persistent_workers deadlocks
TRAIN_EPOCHS = 25   # single run, differential LRs warm up the stat branch naturally

CLASS_NAMES = ["Machine1_Normal","Machine1_Abnormal","Machine2_Normal",
               "Machine2_Abnormal","Machine3_Normal","Machine3_Abnormal"]

# Fixed feature set — no ablation needed, Phase 3 already validated these
STAT_FEATURES = ["rms", "zcr", "rolloff", "bandwidth", "kurtosis"]
STAT_COL      = {"rms": 0, "zcr": 1, "rolloff": 2, "bandwidth": 3, "kurtosis": 4}

# ── PREPROCESSING (inline — no external package imports needed) ────────────────
EPSILON = 1e-8

@dataclass
class AugmentationConfig:
    enabled:bool=True; noise_prob:float=0.35; noise_snr_db_min:float=15.0
    noise_snr_db_max:float=35.0; time_shift_prob:float=0.30; time_shift_max_sec:float=0.20
    pitch_shift_prob:float=0.20; pitch_shift_min_semitones:float=-1.0
    pitch_shift_max_semitones:float=1.0; random_crop_train:bool=True

@dataclass
class PreprocessConfig:
    target_sr:int=16000; default_duration_sec:float=2.75; trim_silence:bool=True
    silence_threshold_ratio:float=0.02; trim_frame_ms:int=20; trim_hop_ms:int=10
    min_retained_sec:float=0.25; denoise:bool=False; normalize_mode:str="peak"
    peak_target:float=0.95; rms_target:float=0.10; clip_value:float=1.0
    augmentation:AugmentationConfig=field(default_factory=AugmentationConfig)

class AudioPreprocessor:
    def __init__(self,config=None): self.config=config or PreprocessConfig()
    def preprocess(self,audio_path,duration_sec=None,target_sr=None,mode="inference",seed=None):
        cfg=self.config; eff_sr=int(target_sr or cfg.target_sr)
        eff_dur=float(duration_sec or cfg.default_duration_sec); tgt_len=int(round(eff_sr*eff_dur))
        try: data,orig_sr=sf.read(str(audio_path),always_2d=False,dtype="float32")
        except: return np.zeros(tgt_len,dtype=np.float32)
        w=np.asarray(data,dtype=np.float32)
        if w.ndim>1: w=w.mean(axis=1)
        if orig_sr!=eff_sr:
            d=math.gcd(orig_sr,eff_sr); w=resample_poly(w,eff_sr//d,orig_sr//d).astype(np.float32)
        if cfg.trim_silence: w=self._trim(w,eff_sr)
        w=self._normalize(w)
        rng=np.random.default_rng(seed) if mode=="train" else None
        if mode=="train": w=self._augment(w,eff_sr,rng)
        w=self._fix_length(w,tgt_len,mode=="train",rng)
        w=np.nan_to_num(w,0.0); np.clip(w,-cfg.clip_value,cfg.clip_value,out=w)
        return w.astype(np.float32)
    def _trim(self,w,sr):
        cfg=self.config
        if w.size==0: return w
        peak=np.abs(w).max()
        if peak<=EPSILON: return w
        thr=peak*cfg.silence_threshold_ratio; fl=max(1,int(sr*cfg.trim_frame_ms/1000)); hl=max(1,int(sr*cfg.trim_hop_ms/1000))
        aw=np.abs(w); active=[s for s in range(0,w.size-fl+1,hl) if aw[s:s+fl].max()>=thr]
        if not active: return w
        trimmed=w[active[0]:min(w.size,active[-1]+fl)]
        if trimmed.size<int(cfg.min_retained_sec*sr): return w
        return trimmed.astype(np.float32)
    def _normalize(self,w):
        mode=self.config.normalize_mode
        if mode=="peak":
            p=np.abs(w).max()
            if p>EPSILON: w=w*(self.config.peak_target/p)
        return w.astype(np.float32)
    def _augment(self,w,sr,rng):
        aug=self.config.augmentation
        if not aug.enabled or w.size==0: return w
        if rng.random()<aug.noise_prob:
            sig_rms=np.sqrt(np.mean(w**2))
            if sig_rms>EPSILON:
                snr=rng.uniform(aug.noise_snr_db_min,aug.noise_snr_db_max)
                noise=rng.normal(0,1,w.shape).astype(np.float32); n_rms=np.sqrt(np.mean(noise**2))
                if n_rms>EPSILON: w=w+noise*(sig_rms/(10**(snr/20))/(n_rms+EPSILON))
        if rng.random()<aug.time_shift_prob:
            ms=int(round(aug.time_shift_max_sec*sr))
            if ms>0: w=np.roll(w,int(rng.integers(-ms,ms+1)))
        return w.astype(np.float32)
    def _fix_length(self,w,tgt,training,rng):
        cur=w.size; aug=self.config.augmentation
        if cur==tgt: return w
        if cur>tgt:
            start=(int(rng.integers(0,cur-tgt+1)) if training and aug.random_crop_train and rng is not None else (cur-tgt)//2)
            return w[start:start+tgt]
        pad=tgt-cur; lp=(int(rng.integers(0,pad+1)) if training and aug.random_crop_train and rng is not None else 0)
        return np.pad(w,(lp,pad-lp),mode="constant")

# ── FEATURE FUNCTIONS ─────────────────────────────────────────────────────────
def _mm(S):
    lo,hi=S.min(),S.max()
    return np.zeros_like(S) if hi-lo<1e-8 else (S-lo)/(hi-lo)

def compute_mel_spectrogram(waveform, sr=16000):
    """waveform (44000,) → np.ndarray (1,128,84)"""
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=128, n_fft=1024,
        hop_length=512, fmin=50, fmax=8000, center=False)
    return np.expand_dims(_mm(librosa.power_to_db(mel, ref=np.max)).astype(np.float32), 0)

def compute_stat_v2(waveform, sr=16000):
    """5 features in canonical order: [rms, zcr, rolloff, bandwidth, kurtosis]. No centroid."""
    return np.array([
        float(np.sqrt(np.mean(waveform ** 2))),
        float(librosa.feature.zero_crossing_rate(waveform).mean()),
        float(librosa.feature.spectral_rolloff(y=waveform, sr=sr).mean()),
        float(librosa.feature.spectral_bandwidth(y=waveform, sr=sr).mean()),
        float(scipy_kurtosis(waveform, fisher=True)),
    ], dtype=np.float32)

def spec_augment(mel, freq_mask=30, time_mask=15, n_freq=2, n_time=2):
    mel=mel.clone(); _,F,T=mel.shape
    for _ in range(n_freq):
        f=random.randint(0,freq_mask); f0=random.randint(0,max(F-f,1)); mel[:,f0:f0+f,:]=0.0
    for _ in range(n_time):
        t=random.randint(0,time_mask); t0=random.randint(0,max(T-t,1)); mel[:,:,t0:t0+t]=0.0
    return mel

# ── PRE-COMPUTATION ────────────────────────────────────────────────────────────
# WHY PRE-COMPUTE STAT:
#   Statistical features are fast (~0.005 s per file) but still add up at 56 k files.
#   Saving them once as 5-element .npy files means training only calls np.load().
#   The stat array stores ALL 5 features; ablation slices the relevant columns.

def _precompute_one_mel_stat(args):
    idx, wav_path, mel_dir, stat_dir, preprocessor = args
    mel_out  = pathlib.Path(mel_dir)  / f"{idx:06d}.npy"
    stat_out = pathlib.Path(stat_dir) / f"{idx:06d}.npy"
    if mel_out.exists() and stat_out.exists():
        return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        if not mel_out.exists():  np.save(mel_out,  compute_mel_spectrogram(w))
        if not stat_out.exists(): np.save(stat_out, compute_stat_v2(w))
    except Exception:
        if not mel_out.exists():  np.save(mel_out,  np.zeros((1, 128, 84), dtype=np.float32))
        if not stat_out.exists(): np.save(stat_out, np.zeros(5,             dtype=np.float32))

def precompute_all_mel_stat(paths, mel_dir, stat_dir, preprocessor, n_workers=4):
    import multiprocessing, tqdm as tqdm_module
    mel_dir  = pathlib.Path(mel_dir)
    stat_dir = pathlib.Path(stat_dir)
    # Read-only uploaded datasets — skip entirely
    if str(mel_dir).startswith("/kaggle/input") and str(stat_dir).startswith("/kaggle/input"):
        print("[cache] mel+stat loaded from uploaded dataset  ✓"); return
    mel_dir.mkdir(parents=True, exist_ok=True)
    stat_dir.mkdir(parents=True, exist_ok=True)
    already = sum(1 for i in range(len(paths))
                  if (mel_dir/f"{i:06d}.npy").exists() and (stat_dir/f"{i:06d}.npy").exists())
    if already == len(paths):
        print(f"All {len(paths)} mel+stat features already cached  (skipping)"); return
    print(f"Pre-computing mel + stat for {len(paths)} files using {n_workers} workers ...")
    print("Runs ONCE per session (~5-10 min). Training epochs then take ~3 min each.")
    args = [(i, p, str(mel_dir), str(stat_dir), preprocessor) for i, p in enumerate(paths)]
    with multiprocessing.Pool(n_workers) as pool:
        list(tqdm_module.tqdm(pool.imap(_precompute_one_mel_stat, args, chunksize=64),
                              total=len(paths), desc="mel+stat"))
    print("Pre-computation done.")

# ── DATASET ────────────────────────────────────────────────────────────────────
class PrecomputedDatasetMS(Dataset):
    """Loads mel + stat from .npy. Stat is the full 5-element stat_v2 vector — no slicing needed."""
    def __init__(self, mel_dir, stat_dir, labels, indices, augment=False):
        self.mel_dir  = pathlib.Path(mel_dir)
        self.stat_dir = pathlib.Path(stat_dir)
        self.labels   = labels
        self.indices  = indices
        self.augment  = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri   = self.indices[idx]
        mel  = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        stat = torch.tensor(np.load(self.stat_dir / f"{ri:06d}.npy"), dtype=torch.float32)  # (5,)
        if self.augment:
            mel = spec_augment(mel)
        return (mel, stat), torch.tensor(self.labels[ri], dtype=torch.long)

def collate_ms(batch):
    feats,labels=zip(*batch)
    return (torch.stack([f[0] for f in feats]),
            torch.stack([f[1] for f in feats])), torch.stack(labels)

# ── MODEL ──────────────────────────────────────────────────────────────────────
class MelCNN(nn.Module):
    def __init__(self,num_classes=6):
        super().__init__()
        self.block1=nn.Sequential(nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),nn.MaxPool2d(2))
        self.block2=nn.Sequential(nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64), nn.ReLU(),nn.MaxPool2d(2))
        self.block3=nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2))
        self.block4=nn.Sequential(nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.fc1=nn.Linear(256*4*4,256); self.dropout=nn.Dropout(0.5); self.fc2=nn.Linear(256,num_classes)
    def extract_features(self,x):
        x=self.block1(x);x=self.block2(x);x=self.block3(x);x=self.block4(x)
        return self.dropout(F.relu(self.fc1(torch.flatten(x,1))))
    def forward(self,x): return self.fc2(self.extract_features(x))

class MelStatCNN(nn.Module):
    """
    mel_stream  → 256-d  (MelCNN.extract_features, loaded from Phase 1)
    stat_branch →  32-d  (Linear(stat_dim→64)→ReLU→Linear(64→32)→ReLU)
    concat      → 288-d  → fc1(256) → Dropout(0.4) → fc2(6)
    """
    def __init__(self,num_classes=6,stat_dim=5):
        super().__init__()
        self.mel_stream=MelCNN(num_classes)
        self.stat_branch=nn.Sequential(
            nn.Linear(stat_dim,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.fc1=nn.Linear(256+32,256); self.dropout=nn.Dropout(0.4); self.fc2=nn.Linear(256,num_classes)
    def forward(self,mel,stat):
        f_mel=self.mel_stream.extract_features(mel); f_stat=self.stat_branch(stat)
        return self.fc2(self.dropout(F.relu(self.fc1(torch.cat([f_mel,f_stat],dim=1)))))

# ── HELPERS ────────────────────────────────────────────────────────────────────
def train_epoch_ms(model,loader,optimizer,criterion,device,sm,ss):
    model.train(); sm=torch.tensor(sm,dtype=torch.float32).to(device); ss=torch.tensor(ss,dtype=torch.float32).to(device)
    tl,cor,tot=0.0,0,0
    for (mel,stat),y in loader:
        mel,stat,y=mel.to(device),stat.to(device),y.to(device); stat=(stat-sm)/ss
        optimizer.zero_grad(); out=model(mel,stat); loss=criterion(out,y)
        loss.backward(); optimizer.step()
        tl+=loss.item(); cor+=(out.argmax(1)==y).sum().item(); tot+=y.size(0)
    return tl/len(loader),cor/tot

def eval_epoch_ms(model,loader,criterion,device,sm,ss):
    model.eval(); sm=torch.tensor(sm,dtype=torch.float32).to(device); ss=torch.tensor(ss,dtype=torch.float32).to(device)
    tl,cor,tot=0.0,0,0; ap,al=[],[]
    with torch.no_grad():
        for (mel,stat),y in loader:
            mel,stat,y=mel.to(device),stat.to(device),y.to(device); stat=(stat-sm)/ss
            out=model(mel,stat); loss=criterion(out,y); preds=out.argmax(1)
            tl+=loss.item(); cor+=(preds==y).sum().item(); tot+=y.size(0)
            ap.extend(preds.cpu().numpy()); al.extend(y.cpu().numpy())
    return tl/len(loader),cor/tot,ap,al

def fit_scaler(stat_dir, indices):
    """Load stat .npy files directly — no wav reads, runs in seconds."""
    arr = np.stack([np.load(pathlib.Path(stat_dir) / f"{i:06d}.npy") for i in indices])
    return arr.mean(0), arr.std(0) + 1e-8

def save_ckpt(model,epoch,val_acc,path,stat_features,stat_dim):
    torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"val_acc":val_acc,
                "stat_features":stat_features,"stat_dim":stat_dim},path)

def plot_cm(preds,labels,class_names):
    cm=confusion_matrix(labels,preds)
    plt.figure(figsize=(8,6)); sns.heatmap(cm,annot=True,fmt="d",cmap="Purples",
        xticklabels=class_names,yticklabels=class_names)
    plt.ylabel("True Label"); plt.xlabel("Predicted Label"); plt.tight_layout(); plt.show()

# ── MAIN ───────────────────────────────────────────────────────────────────────

# Step A: scan all files
ALL_PATHS, ALL_LABELS = _scan_wav_files(ROOT_DIR)

# Step B: pre-compute mel + stat once
# mel: reuses cached files from Phase 1/2 if /kaggle/working/feats_mel already exists
# stat: new — 5-element vector per file, very fast to compute
_infer_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=SR, default_duration_sec=DURATION_SEC, trim_silence=True, normalize_mode="peak",
    augmentation=AugmentationConfig(enabled=False),
))
precompute_all_mel_stat(ALL_PATHS, FEATS_DIR_MEL, FEATS_DIR_STAT, _infer_prep, n_workers=NUM_WORKERS)

# Step C: load split (created in Phase 1)
_splits = _load_clean_split(MODELS_DIR)

label_counts  = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
class_weights = torch.tensor(1.0/(label_counts+1), dtype=torch.float32).to(DEVICE)

# Build datasets
tr_ds = PrecomputedDatasetMS(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["train"], augment=True)
vl_ds = PrecomputedDatasetMS(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["val"],   augment=False)
te_ds = PrecomputedDatasetMS(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS, _splits["test"],  augment=False)

print("Fitting scaler...")
sm, ss = fit_scaler(FEATS_DIR_STAT, _splits["train"])

tr_ldr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_ms,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False, prefetch_factor=2)
vl_ldr = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_ms,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False, prefetch_factor=2)
te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_ms,
                    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False, prefetch_factor=2)

model = MelStatCNN(num_classes=6, stat_dim=len(STAT_FEATURES)).to(DEVICE)
p1    = torch.load(PHASE1_CKPT, map_location=DEVICE)
model.mel_stream.load_state_dict(p1["model_state_dict"])

# mel_stream came from Phase 1 — low LR to keep what it already learned
optimizer = torch.optim.AdamW([
    {"params": model.mel_stream.parameters(),  "lr": 1e-4},
    {"params": model.stat_branch.parameters(), "lr": 5e-4},
    {"params": model.fc1.parameters(),         "lr": 5e-4},
    {"params": model.fc2.parameters(),         "lr": 5e-4},
], weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)

print(f"\nTraining {TRAIN_EPOCHS} epochs — features: {STAT_FEATURES}\n")

best_val  = 0.0
ckpt_path = os.path.join(MODELS_DIR, "phase2b_best.pth")

for epoch in range(1, TRAIN_EPOCHS + 1):
    tr_l, tr_a     = train_epoch_ms(model, tr_ldr, optimizer, criterion, DEVICE, sm, ss)
    vl_l, vl_a,_,_ = eval_epoch_ms( model, vl_ldr, criterion, DEVICE, sm, ss)
    scheduler.step()
    saved = ""
    if vl_a > best_val:
        best_val = vl_a
        save_ckpt(model, epoch, vl_a, ckpt_path, STAT_FEATURES, len(STAT_FEATURES))
        saved = "  ← saved"
    print(f"Epoch {epoch:3d}/{TRAIN_EPOCHS}  train={tr_a:.4f}  val={vl_a:.4f}{saved}")

print(f"\nBest val accuracy: {best_val:.4f}")

# ── TEST EVALUATION ────────────────────────────────────────────────────────────
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  CHECKPOINTS SAVED TO:  /kaggle/working/                                     │
# │    phase2b_best.pth   — model weights + stat_features + stat_dim             │
# │    stat_scaler_2b.pkl — {"mean", "std", "features"} for inference            │
# │  → Download both, upload as a single dataset for Phase 3 if desired.         │
# └──────────────────────────────────────────────────────────────────────────────┘
ckpt = torch.load(ckpt_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

# Save scaler so inference scripts don't need to refit it
scaler_path = os.path.join(MODELS_DIR, "stat_scaler_2b.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump({"mean": sm, "std": ss, "features": STAT_FEATURES}, f)
print(f"Scaler saved → {scaler_path}")

t0 = time.time()
_, ta, preds, labels = eval_epoch_ms(model, te_ldr, criterion, DEVICE, sm, ss)
t_test = time.time() - t0
n_test = len(te_ds)
ms_per_sample = (t_test / n_test) * 1000

print(f"\n── Test Results ──────────────────────────────────────────")
print(f"Test accuracy : {ta:.4f}")
print(f"Macro F1      : {f1_score(labels,preds,average='macro'):.4f}")
print(f"Stat features : {STAT_FEATURES}")
print("\n",classification_report(labels,preds,target_names=CLASS_NAMES))

print(f"\n── Inference Timing ──────────────────────────────────────")
print(f"Test set size         : {n_test} samples")
print(f"Total inference time  : {t_test:.2f} s")
print(f"Per-sample time       : {ms_per_sample:.3f} ms  →  {1000/ms_per_sample:.0f} samples/sec")
print(f"Estimated   100 files : {ms_per_sample *   100 / 1000:.2f} s")
print(f"Estimated 1 000 files : {ms_per_sample *  1000 / 1000:.2f} s")
print(f"Estimated 10 000 files: {ms_per_sample * 10000 / 1000:.2f} s")

print(f"\n── Phase Comparison ──────────────────────────────────────")
print(f"Phase 1  (Mel only)   : ~0.24 ms/sample  99.80%  baseline")
print(f"Phase 2  (Mel+MFCC)  : ~0.49 ms/sample  99.91%  +0.11% acc, 2× slower")
print(f"Phase 2b (Mel+Stat)  : {ms_per_sample:.3f} ms/sample  {ta:.2%}  this run")
print(f"Phase 3  (All three) : run kaggle_phase3.py for the full ensemble")

plot_cm(preds, labels, CLASS_NAMES)
print(f"\nFiles saved: phase2b_best.pth, stat_scaler_2b.pkl  → /kaggle/working/")

# ── ARCHIVE STAT FEATURES FOR REUSE ──────────────────────────────────────────
# feats_stat is tiny (~1 MB for 56k files) but saves recomputation time.
# Download feats_stat_archive.zip and add it to your features dataset alongside
# feats_mel/ and feats_mfcc/ from Phases 1 & 2.
import shutil
for folder, archive in [("feats_mel","feats_mel_archive"), ("feats_stat_v2","feats_stat_v2_archive")]:
    src = pathlib.Path("/kaggle/working") / folder
    if src.exists() and any(src.glob("*.npy")):   # only archive if freshly computed this session
        shutil.make_archive(f"/kaggle/working/{archive}", "zip", "/kaggle/working", folder)
        sz = os.path.getsize(f"/kaggle/working/{archive}.zip") / 1e9
        print(f"{archive}.zip  ({sz:.3f} GB)  → /kaggle/working/")
