"""
PHASE 3 — Mel + MFCC + Statistical Features (Ablation)     
Runs 5 ablation experiments (CNN frozen, ~10 epochs each)  
then does one final end-to-end fine-tune with the best      
statistical config.                                         
Before running:                                             
1. Add the machine-fault dataset (same as Phase 1 & 2)   
2. Download phase2_best.pth from Phase 2's output         
    → upload it as a Kaggle dataset (e.g. "phase2ckpt")   
    → add that dataset to this notebook                    
3. Set PHASE2_CKPT below if your dataset name differs     
Output: phase3_best.pth + stat_scaler.pkl → /kaggle/working 

"""

import os, json, math, pathlib, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import soundfile as sf
from scipy.signal import resample_poly

# ── CONFIG ────────────────────────────────────────────────────────────────────
# ── PATHS ─────────────────────────────────────────────────────────────────────
_DATASET_BASE = "/kaggle/input/datasets/mostafaehab41/machine-fault-dataset"

def _find_machine_root(base: str) -> str:
    """Return the directory that directly contains 'Machine 1', 'Machine 2', 'Machine 3'."""
    base_p = pathlib.Path(base)
    if any((base_p / f"Machine {i}").exists() for i in range(1, 4)):
        return str(base_p)
    for sub in sorted(base_p.rglob("Machine 1")):
        return str(sub.parent)
    print(f"WARNING: Could not auto-detect machine folders under {base}. Using base path.")
    return str(base_p)

ROOT_DIR    = _find_machine_root(_DATASET_BASE)
# Download phase2_best.pth from Phase 2's output, upload as a Kaggle dataset,
# add it to this notebook, then update the path below if needed.
PHASE2_CKPT = "/kaggle/input/phase2ckpt/phase2_best.pth"   # ← CHANGE if your dataset name differs
MODELS_DIR  = "/kaggle/working"

print(f"ROOT_DIR    : {ROOT_DIR}")
print(f"PHASE2_CKPT : {PHASE2_CKPT}")
print(f"Checkpoint exists: {pathlib.Path(PHASE2_CKPT).exists()}")

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SR=16000; DURATION_SEC=2.75; BATCH_SIZE=32
NUM_WORKERS      = 4
FEATS_DIR_MEL    = pathlib.Path("/kaggle/working/feats_mel")
FEATS_DIR_MFCC   = pathlib.Path("/kaggle/working/feats_mfcc")
ABLATION_EPOCHS  = 10    # fast — CNN is frozen, only FC branches train
FINETUNE_EPOCHS  = 20    # full end-to-end fine-tune after ablation
CLASS_NAMES=["Machine1_Normal","Machine1_Abnormal","Machine2_Normal",
             "Machine2_Abnormal","Machine3_Normal","Machine3_Abnormal"]

# Ablation configs — adds one feature at a time so you can see each contribution
ABLATION_CONFIGS = [
    {"name": "rms_only",                     "features": ["rms"]},
    {"name": "rms_zcr",                      "features": ["rms","zcr"]},
    {"name": "rms_zcr_centroid",             "features": ["rms","zcr","centroid"]},
    {"name": "rms_zcr_centroid_rolloff",     "features": ["rms","zcr","centroid","rolloff"]},
    {"name": "all_five",                     "features": ["rms","zcr","centroid","rolloff","bandwidth"]},
]

# ── PREPROCESSING (same block as phases 1 & 2) ───────────────────────────────
EPSILON=1e-8
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
        cfg=self.config; eff_sr=int(target_sr or cfg.target_sr); eff_dur=float(duration_sec or cfg.default_duration_sec); tgt_len=int(round(eff_sr*eff_dur))
        try: data,orig_sr=sf.read(str(audio_path),always_2d=False,dtype="float32")
        except: return np.zeros(tgt_len,dtype=np.float32)
        w=np.asarray(data,dtype=np.float32)
        if w.ndim>1: w=w.mean(axis=1)
        if orig_sr!=eff_sr:
            d=math.gcd(orig_sr,eff_sr); w=resample_poly(w,eff_sr//d,orig_sr//d).astype(np.float32)
        if cfg.trim_silence:
            if w.size>0:
                peak=np.abs(w).max()
                if peak>EPSILON:
                    thr=peak*cfg.silence_threshold_ratio; fl=max(1,int(eff_sr*cfg.trim_frame_ms/1000)); hl=max(1,int(eff_sr*cfg.trim_hop_ms/1000))
                    aw=np.abs(w); active=[s for s in range(0,w.size-fl+1,hl) if aw[s:s+fl].max()>=thr]
                    if active:
                        trimmed=w[active[0]:min(w.size,active[-1]+fl)]
                        if trimmed.size>=int(cfg.min_retained_sec*eff_sr): w=trimmed.astype(np.float32)
        p=np.abs(w).max()
        if p>EPSILON: w=w*(cfg.peak_target/p)
        rng=np.random.default_rng(seed) if mode=="train" else None
        if mode=="train" and cfg.augmentation.enabled and w.size>0:
            aug=cfg.augmentation
            if rng.random()<aug.noise_prob:
                sig_rms=np.sqrt(np.mean(w**2))
                if sig_rms>EPSILON:
                    snr=rng.uniform(aug.noise_snr_db_min,aug.noise_snr_db_max); noise=rng.normal(0,1,w.shape).astype(np.float32); n_rms=np.sqrt(np.mean(noise**2))
                    if n_rms>EPSILON: w=w+noise*(sig_rms/(10**(snr/20))/(n_rms+EPSILON))
            if rng.random()<aug.time_shift_prob:
                ms=int(round(aug.time_shift_max_sec*eff_sr))
                if ms>0: w=np.roll(w,int(rng.integers(-ms,ms+1)))
        cur=w.size
        if cur>tgt_len:
            start=(int(rng.integers(0,cur-tgt_len+1)) if mode=="train" and rng is not None else (cur-tgt_len)//2)
            w=w[start:start+tgt_len]
        elif cur<tgt_len:
            pad=tgt_len-cur; lp=(int(rng.integers(0,pad+1)) if mode=="train" and rng is not None else 0)
            w=np.pad(w,(lp,pad-lp),mode="constant")
        w=np.nan_to_num(w,0.0); np.clip(w,-cfg.clip_value,cfg.clip_value,out=w)
        return w.astype(np.float32)

# ── FEATURE FUNCTIONS ─────────────────────────────────────────────────────────
def _mm(S):
    lo,hi=S.min(),S.max()
    return np.zeros_like(S) if hi-lo<1e-8 else (S-lo)/(hi-lo)

def compute_mel_spectrogram(waveform,sr=16000):
    mel=librosa.feature.melspectrogram(y=waveform,sr=sr,n_mels=128,n_fft=1024,hop_length=512,fmin=50,fmax=8000,center=False)
    return np.expand_dims(_mm(librosa.power_to_db(mel,ref=np.max)).astype(np.float32),0)

def compute_mfcc(waveform,sr=16000):
    mfcc=librosa.feature.mfcc(y=waveform,sr=sr,n_mfcc=40,n_fft=1024,hop_length=512)
    d=librosa.feature.delta(mfcc); d2=librosa.feature.delta(mfcc,order=2)
    feats=np.stack([mfcc,d,d2],axis=0)
    for i in range(3): feats[i]=_mm(feats[i])
    return feats.astype(np.float32)

def compute_statistical_features(waveform, sr=16000, feature_names=None):
    """Returns raw (unscaled) statistical features."""
    all_names=["rms","zcr","centroid","rolloff","bandwidth"]
    if feature_names is None: feature_names=all_names
    result=[]
    for name in feature_names:
        if name=="rms":       result.append(float(np.sqrt(np.mean(waveform**2))))
        elif name=="zcr":     result.append(float(librosa.feature.zero_crossing_rate(waveform).mean()))
        elif name=="centroid":result.append(float(librosa.feature.spectral_centroid(y=waveform,sr=sr).mean()))
        elif name=="rolloff": result.append(float(librosa.feature.spectral_rolloff(y=waveform,sr=sr).mean()))
        elif name=="bandwidth":result.append(float(librosa.feature.spectral_bandwidth(y=waveform,sr=sr).mean()))
    return np.array(result,dtype=np.float32)

def spec_augment(mel, freq_mask=30, time_mask=15, n_freq=2, n_time=2):
    """Mask random frequency and time bands on a mel tensor. Zero CPU cost vs librosa augmentation."""
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

def _precompute_one_dual(args):
    """Top-level worker for multiprocessing: preprocess one wav and save mel + mfcc as .npy."""
    idx, wav_path, mel_dir, mfcc_dir, preprocessor = args
    mel_out  = pathlib.Path(mel_dir)  / f"{idx:06d}.npy"
    mfcc_out = pathlib.Path(mfcc_dir) / f"{idx:06d}.npy"
    if mel_out.exists() and mfcc_out.exists():
        return
    try:
        w = preprocessor.preprocess(str(wav_path), mode="inference")
        np.save(mel_out,  compute_mel_spectrogram(w))
        np.save(mfcc_out, compute_mfcc(w))
    except Exception:
        np.save(mel_out,  np.zeros((1, 128, 84), dtype=np.float32))
        np.save(mfcc_out, np.zeros((3, 40,  84), dtype=np.float32))

def precompute_all_dual(paths, mel_dir, mfcc_dir, preprocessor, n_workers=4):
    """Pre-compute mel + mfcc for every wav file. Skips already-saved files (resumable)."""
    import multiprocessing, tqdm as tqdm_module
    mel_dir  = pathlib.Path(mel_dir);  mel_dir.mkdir(parents=True, exist_ok=True)
    mfcc_dir = pathlib.Path(mfcc_dir); mfcc_dir.mkdir(parents=True, exist_ok=True)
    already = sum(1 for i in range(len(paths))
                  if (mel_dir/f"{i:06d}.npy").exists() and (mfcc_dir/f"{i:06d}.npy").exists())
    if already == len(paths):
        print(f"All {len(paths)} mel+mfcc features already cached  (skipping)"); return
    print(f"Pre-computing mel + mfcc for {len(paths)} files using {n_workers} workers ...")
    print("Runs ONCE per session (~20 min). Training epochs will then take ~3-5 min each.")
    args = [(i, p, str(mel_dir), str(mfcc_dir), preprocessor) for i, p in enumerate(paths)]
    with multiprocessing.Pool(n_workers) as pool:
        list(tqdm_module.tqdm(pool.imap(_precompute_one_dual, args, chunksize=64),
                              total=len(paths), desc="mel+mfcc"))
    print("Pre-computation done.")

# ── DATASET (returns mel, mfcc, stat as a 3-tuple) ───────────────────────────
# Actual folder structure:
#   machine-fault-dataset/machine1/Normal/*.wav   → label 0
#   machine-fault-dataset/machine1/Abnormal/*.wav → label 1  ... etc.
# f.parent.name = "Normal"/"Abnormal"   (state)
# f.parent.parent.name = "machine1/2/3" (machine)  ← only 2 levels, NOT 3

class MachineDataset3(Dataset):
    """Returns (mel, mfcc, stat_raw), label.  Stat is not yet scaled."""
    LABEL_MAP={("machine1","Normal"):0,("machine1","Abnormal"):1,
               ("machine2","Normal"):2,("machine2","Abnormal"):3,
               ("machine3","Normal"):4,("machine3","Abnormal"):5}
    def __init__(self,root_dir,preprocessor,stat_features,split,augment=False):
        self.root_dir=pathlib.Path(root_dir); self.preprocessor=preprocessor
        self.stat_features=stat_features; self.split=split; self.augment=augment
        self.paths,self.labels=self._scan(); self.indices=self._split()
    def _scan(self):
        paths,labels=[],[]
        for f in self.root_dir.rglob("*.wav"):
            state   = f.parent.name         # "Normal" or "Abnormal"
            machine = f.parent.parent.name  # "machine1", "machine2", "machine3"
            lbl=self.LABEL_MAP.get((machine,state))
            if lbl is not None: paths.append(f); labels.append(lbl)
        if not paths: raise RuntimeError(
            f"No labelled .wav files under {self.root_dir}\n"
            f"Top-level folders found: {[p.name for p in self.root_dir.iterdir() if p.is_dir()]}")
        print(f"Found {len(paths)} files"); return paths,labels
    def _split(self):
        # /kaggle/working is writable; /kaggle/input is READ-ONLY — never write there
        sf_path=pathlib.Path(MODELS_DIR)/"split_indices.json"
        if sf_path.exists(): return json.load(open(sf_path))[self.split]
        idx=list(range(len(self.paths)))
        tr,tmp,_,tl=train_test_split(idx,self.labels,test_size=0.30,stratify=self.labels,random_state=42)
        va,te=train_test_split(tmp,test_size=0.50,stratify=tl,random_state=42)
        json.dump({"train":tr,"val":va,"test":te},open(sf_path,"w"))
        print(f"Split saved → {sf_path}")
        return {"train":tr,"val":va,"test":te}[self.split]
    def __len__(self): return len(self.indices)
    def __getitem__(self,idx):
        ri=self.indices[idx]; path=self.paths[ri]; lbl=self.labels[ri]
        mode="train" if (self.split=="train" and self.augment) else "inference"
        w=self.preprocessor.preprocess(path,mode=mode)
        mel  = compute_mel_spectrogram(w)
        mfcc = compute_mfcc(w)
        stat = compute_statistical_features(w,feature_names=self.stat_features)
        return (torch.tensor(mel,dtype=torch.float32),
                torch.tensor(mfcc,dtype=torch.float32),
                torch.tensor(stat,dtype=torch.float32)), torch.tensor(lbl,dtype=torch.long)

def collate3(batch):
    feats,labels=zip(*batch)
    return (torch.stack([f[0] for f in feats]),
            torch.stack([f[1] for f in feats]),
            torch.stack([f[2] for f in feats])), torch.stack(labels)

class PrecomputedDataset3(Dataset):
    """Loads pre-computed mel + mfcc .npy files; computes stat features on-the-fly (fast numpy).
    SpecAugment applied to mel at train time."""
    def __init__(self, mel_dir, mfcc_dir, labels, indices, stat_features, augment=False):
        self.mel_dir       = pathlib.Path(mel_dir)
        self.mfcc_dir      = pathlib.Path(mfcc_dir)
        self.labels        = labels
        self.indices       = indices
        self.stat_features = stat_features
        self.augment       = augment
        # Pre-load all wav paths from the full dataset scan so we can compute stat features
        # We need the wav paths — they are stored as ALL_PATHS at module level after scanning
        self._all_paths    = None  # set externally: ds.set_paths(ALL_PATHS)
    def set_paths(self, all_paths):
        self._all_paths = all_paths
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        ri   = self.indices[idx]
        mel  = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        mfcc = torch.tensor(np.load(self.mfcc_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment:
            mel = spec_augment(mel)
        # Compute stat features inline — stat is fast pure-numpy (< 0.005s each).
        # We read the raw wav directly (no augmentation, no resampling needed for stat).
        if self._all_paths is not None:
            import soundfile as _sf
            try:
                w, _sr = _sf.read(str(self._all_paths[ri]), always_2d=False, dtype="float32")
                if w.ndim > 1: w = w.mean(axis=1)
            except Exception:
                w = np.zeros(int(SR * DURATION_SEC), dtype=np.float32)
        else:
            w = np.zeros(int(SR * DURATION_SEC), dtype=np.float32)
        stat = compute_statistical_features(w, feature_names=self.stat_features)
        return (mel, mfcc, torch.tensor(stat, dtype=torch.float32)), torch.tensor(self.labels[ri], dtype=torch.long)

# ── MODELS ────────────────────────────────────────────────────────────────────
class MelCNN(nn.Module):
    def __init__(self,num_classes=6):
        super().__init__()
        self.block1=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2))
        self.block2=nn.Sequential(nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2))
        self.block3=nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2))
        self.block4=nn.Sequential(nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.fc1=nn.Linear(256*4*4,256); self.dropout=nn.Dropout(0.5); self.fc2=nn.Linear(256,6)
    def extract_features(self,x):
        x=self.block1(x);x=self.block2(x);x=self.block3(x);x=self.block4(x)
        return self.dropout(F.relu(self.fc1(torch.flatten(x,1))))
    def forward(self,x): return self.fc2(self.extract_features(x))

class MFCCStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),nn.AdaptiveAvgPool2d((4,4)))
        self.fc=nn.Linear(64*4*4,128)
    def forward(self,x): return F.relu(self.fc(torch.flatten(self.features(x),1)))

class MelMFCCStatCNN(nn.Module):
    def __init__(self,num_classes=6,stat_dim=5):
        super().__init__()
        self.mel_stream=MelCNN(num_classes); self.mfcc_stream=MFCCStream()
        self.stat_branch=nn.Sequential(nn.Linear(stat_dim,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        # 256 (mel) + 128 (mfcc) + 32 (stat) = 416
        self.fc1=nn.Linear(256+128+32,256); self.dropout=nn.Dropout(0.4); self.fc2=nn.Linear(256,num_classes)
    def freeze_cnn(self):
        for m in [self.mel_stream,self.mfcc_stream]:
            for p in m.parameters(): p.requires_grad=False
    def unfreeze_cnn(self):
        for p in self.parameters(): p.requires_grad=True
    def forward(self,mel,mfcc,stat):
        f_mel=self.mel_stream.extract_features(mel); f_mfcc=self.mfcc_stream(mfcc); f_stat=self.stat_branch(stat)
        x=torch.cat([f_mel,f_mfcc,f_stat],dim=1)
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# ── TRAINING HELPERS ──────────────────────────────────────────────────────────
def train_epoch3(model,loader,optimizer,criterion,device,scaler_mean,scaler_std):
    model.train(); tl,cor,tot=0.0,0,0
    sm=torch.tensor(scaler_mean,dtype=torch.float32).to(device)
    ss=torch.tensor(scaler_std, dtype=torch.float32).to(device)
    for (mel,mfcc,stat),y in loader:
        mel,mfcc,stat,y=mel.to(device),mfcc.to(device),stat.to(device),y.to(device)
        stat=(stat-sm)/ss   # apply StandardScaler normalisation
        optimizer.zero_grad(); out=model(mel,mfcc,stat); loss=criterion(out,y)
        loss.backward(); optimizer.step()
        tl+=loss.item(); cor+=(out.argmax(1)==y).sum().item(); tot+=y.size(0)
    return tl/len(loader),cor/tot

def eval_epoch3(model,loader,criterion,device,scaler_mean,scaler_std):
    model.eval(); tl,cor,tot=0.0,0,0; ap,al=[],[]
    sm=torch.tensor(scaler_mean,dtype=torch.float32).to(device)
    ss=torch.tensor(scaler_std, dtype=torch.float32).to(device)
    with torch.no_grad():
        for (mel,mfcc,stat),y in loader:
            mel,mfcc,stat,y=mel.to(device),mfcc.to(device),stat.to(device),y.to(device)
            stat=(stat-sm)/ss
            out=model(mel,mfcc,stat); loss=criterion(out,y)
            tl+=loss.item(); preds=out.argmax(1)
            cor+=(preds==y).sum().item(); tot+=y.size(0)
            ap.extend(preds.cpu().numpy()); al.extend(y.cpu().numpy())
    return tl/len(loader),cor/tot,ap,al

def load_phase2_weights(model, ckpt_path, device):
    ckpt=torch.load(ckpt_path,map_location=device)
    sd=ckpt["model_state_dict"]
    # load mel_stream and mfcc_stream weights (keys match because class names are the same)
    model.mel_stream.load_state_dict(
        {k[len("mel_stream."):]:v for k,v in sd.items() if k.startswith("mel_stream.")})
    model.mfcc_stream.load_state_dict(
        {k[len("mfcc_stream."):]:v for k,v in sd.items() if k.startswith("mfcc_stream.")})
    print("Loaded Phase 2 CNN weights \u2713")

def fit_scaler_precomputed(dataset):
    """Compute mean & std from a PrecomputedDataset3 (stat features only)."""
    all_stat=[]
    for i in range(len(dataset)):
        (_,_,stat_t),_=dataset[i]
        all_stat.append(stat_t.numpy())
    arr=np.stack(all_stat,axis=0)
    return arr.mean(0), arr.std(0)+1e-8

# ── MAIN ──────────────────────────────────────────────────────────────────────
# Step A: scan all files + create/load the stratified split
_scan_ds   = MachineDataset3(ROOT_DIR,
                 AudioPreprocessor(PreprocessConfig(augmentation=AugmentationConfig(enabled=False))),
                 ["rms"], "train")
ALL_PATHS  = _scan_ds.paths
ALL_LABELS = _scan_ds.labels

# Step B: pre-compute mel + mfcc once (~20 min, then cached every run)
_infer_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=SR, default_duration_sec=DURATION_SEC, trim_silence=True, normalize_mode="peak",
    augmentation=AugmentationConfig(enabled=False),
))
precompute_all_dual(ALL_PATHS, FEATS_DIR_MEL, FEATS_DIR_MFCC, _infer_prep, n_workers=NUM_WORKERS)

# Step C: load split indices
_splits = json.load(open(pathlib.Path(MODELS_DIR) / "split_indices.json"))

ablation_results={}

for cfg_ablation in ABLATION_CONFIGS:
    name    = cfg_ablation["name"]
    feat    = cfg_ablation["features"]
    stat_dim= len(feat)
    print(f"\n{'='*60}")
    print(f"Ablation: {name}  (stat_dim={stat_dim})")
    print(f"{'='*60}")

    train_ds = PrecomputedDataset3(FEATS_DIR_MEL, FEATS_DIR_MFCC, ALL_LABELS, _splits["train"], feat, augment=True)
    val_ds   = PrecomputedDataset3(FEATS_DIR_MEL, FEATS_DIR_MFCC, ALL_LABELS, _splits["val"],   feat, augment=False)
    train_ds.set_paths(ALL_PATHS)
    val_ds.set_paths(ALL_PATHS)

    # Fit StandardScaler on training stat features
    print("Fitting scaler on training stat features...")
    scaler_mean, scaler_std = fit_scaler_precomputed(train_ds)

    tr_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True, collate_fn=collate3,
                         num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=True,prefetch_factor=2)
    vl_loader=DataLoader(val_ds,  batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate3,
                         num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=True,prefetch_factor=2)

    model=MelMFCCStatCNN(num_classes=6,stat_dim=stat_dim).to(DEVICE)
    load_phase2_weights(model, PHASE2_CKPT, DEVICE)
    model.freeze_cnn()   # only train stat branch + head during ablation

    # Only new parameters need optimising here
    new_params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(new_params,lr=5e-4,weight_decay=1e-4)

    label_counts=np.bincount([ALL_LABELS[i] for i in _splits["train"]],minlength=6)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(1.0/(label_counts+1),dtype=torch.float32).to(DEVICE))

    best_vl=0.0
    for epoch in range(1,ABLATION_EPOCHS+1):
        tr_loss,tr_acc=train_epoch3(model,tr_loader,optimizer,criterion,DEVICE,scaler_mean,scaler_std)
        vl_loss,vl_acc,_,_=eval_epoch3(model,vl_loader,criterion,DEVICE,scaler_mean,scaler_std)
        print(f"  Epoch {epoch:2d}/{ABLATION_EPOCHS}  train_acc={tr_acc:.4f}  val_acc={vl_acc:.4f}")
        if vl_acc>best_vl: best_vl=vl_acc

    ablation_results[name]={"val_acc":best_vl,"stat_dim":stat_dim,"features":feat}
    print(f"  Best val_acc for '{name}': {best_vl:.4f}")

# Print ablation comparison table
print("\n\n\u2500\u2500 Ablation Results \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
print(f"{'Config':<35} {'stat_dim':>8} {'val_acc':>8}")
print("-"*55)
for k,v in ablation_results.items():
    print(f"{k:<35} {v['stat_dim']:>8} {v['val_acc']:>8.4f}")

best_name = max(ablation_results, key=lambda k: ablation_results[k]["val_acc"])
best_feat = ablation_results[best_name]["features"]
print(f"\nBest config: '{best_name}' with features {best_feat}")

# ── FINAL FINE-TUNE with best config (unfreeze everything) ───────────────────
print(f"\n\u2500\u2500 Final fine-tune (unfreeze all, {FINETUNE_EPOCHS} epochs) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
stat_dim_final=len(best_feat)

train_ds_f = PrecomputedDataset3(FEATS_DIR_MEL, FEATS_DIR_MFCC, ALL_LABELS, _splits["train"], best_feat, augment=True)
val_ds_f   = PrecomputedDataset3(FEATS_DIR_MEL, FEATS_DIR_MFCC, ALL_LABELS, _splits["val"],   best_feat, augment=False)
test_ds_f  = PrecomputedDataset3(FEATS_DIR_MEL, FEATS_DIR_MFCC, ALL_LABELS, _splits["test"],  best_feat, augment=False)
train_ds_f.set_paths(ALL_PATHS)
val_ds_f.set_paths(ALL_PATHS)
test_ds_f.set_paths(ALL_PATHS)

scaler_mean_f,scaler_std_f=fit_scaler_precomputed(train_ds_f)
# Save scaler so infer.py can use it
pickle.dump({"mean":scaler_mean_f,"std":scaler_std_f,"features":best_feat},
            open(os.path.join(MODELS_DIR,"stat_scaler.pkl"),"wb"))
print("Scaler saved to stat_scaler.pkl \u2713")

tr_ldr=DataLoader(train_ds_f,batch_size=BATCH_SIZE,shuffle=True, collate_fn=collate3,
                  num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=True,prefetch_factor=2)
vl_ldr=DataLoader(val_ds_f,  batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate3,
                  num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=True,prefetch_factor=2)
te_ldr=DataLoader(test_ds_f, batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate3,
                  num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=True,prefetch_factor=2)

model_f=MelMFCCStatCNN(num_classes=6,stat_dim=stat_dim_final).to(DEVICE)
load_phase2_weights(model_f,PHASE2_CKPT,DEVICE)
model_f.unfreeze_cnn()

optimizer_f=torch.optim.AdamW([
    {"params": model_f.mel_stream.parameters(),   "lr":5e-5},
    {"params": model_f.mfcc_stream.parameters(),  "lr":5e-5},
    {"params": model_f.stat_branch.parameters(),  "lr":2e-4},
    {"params": model_f.fc1.parameters(),          "lr":2e-4},
    {"params": model_f.fc2.parameters(),          "lr":2e-4},
], weight_decay=1e-4)
label_counts=np.bincount([ALL_LABELS[i] for i in _splits["train"]],minlength=6)
criterion_f=nn.CrossEntropyLoss(weight=torch.tensor(1.0/(label_counts+1),dtype=torch.float32).to(DEVICE))
scheduler_f=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_f,T_max=FINETUNE_EPOCHS)

best_vl_f=0.0
for epoch in range(1,FINETUNE_EPOCHS+1):
    tr_loss,tr_acc=train_epoch3(model_f,tr_ldr,optimizer_f,criterion_f,DEVICE,scaler_mean_f,scaler_std_f)
    vl_loss,vl_acc,_,_=eval_epoch3(model_f,vl_ldr,criterion_f,DEVICE,scaler_mean_f,scaler_std_f)
    scheduler_f.step()
    print(f"Epoch {epoch:3d}/{FINETUNE_EPOCHS}  train_acc={tr_acc:.4f}  val_acc={vl_acc:.4f}",end="")
    if vl_acc>best_vl_f:
        best_vl_f=vl_acc; ckpt_path=os.path.join(MODELS_DIR,"phase3_best.pth")
        torch.save({"model_state_dict":model_f.state_dict(),"epoch":epoch,"val_acc":vl_acc,
                    "stat_features":best_feat,"stat_dim":stat_dim_final},ckpt_path); print("  \u2190 saved",end="")
    print()

print(f"\nBest val accuracy: {best_vl_f:.4f}")

# ┌──────────────────────────────────────────────────────────────────────┐
# │  CHECKPOINTS SAVED TO:  /kaggle/working/                             │
# │    phase3_best.pth   — model weights + stat_features + stat_dim      │
# │    stat_scaler.pkl   — {"mean", "std", "features"} for inference     │
# │  Both files are needed together at inference time.                   │
# └──────────────────────────────────────────────────────────────────────┘
ckpt=torch.load(os.path.join(MODELS_DIR,"phase3_best.pth"),map_location=DEVICE)
model_f.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_,ta,preds,labels=eval_epoch3(model_f,te_ldr,criterion_f,DEVICE,scaler_mean_f,scaler_std_f)
t_test = time.time() - t0
n_test = len(test_ds_f)
ms_per_sample = (t_test / n_test) * 1000   # milliseconds per sample

print(f"\n── Test Results ──────────────────────────────────────────")
print(f"Test accuracy : {ta:.4f}")
print(f"Macro F1      : {f1_score(labels,preds,average='macro'):.4f}")
print(f"Stat features : {best_feat}")
print("\n",classification_report(labels,preds,target_names=CLASS_NAMES))

print(f"\n── Inference Timing ──────────────────────────────────────")
print(f"Test set size         : {n_test} samples")
print(f"Total inference time  : {t_test:.2f} s")
print(f"Per-sample time       : {ms_per_sample:.3f} ms  →  {1000/ms_per_sample:.0f} samples/sec")
print(f"Estimated   100 files : {ms_per_sample *   100 / 1000:.2f} s")
print(f"Estimated 1 000 files : {ms_per_sample *  1000 / 1000:.2f} s")
print(f"Estimated 10 000 files: {ms_per_sample * 10000 / 1000:.2f} s")

cm=confusion_matrix(labels,preds)
plt.figure(figsize=(8,6)); sns.heatmap(cm,annot=True,fmt="d",cmap="Purples",xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES)
plt.ylabel("True Label"); plt.xlabel("Predicted Label"); plt.tight_layout(); plt.show()
print(f"\nFiles saved: phase3_best.pth, stat_scaler.pkl  → /kaggle/working/")
