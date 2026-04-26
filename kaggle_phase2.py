"""
  PHASE 2 — Dual-Stream CNN: Mel-Spectrogram + MFCC           
  Before running:                                             
    1. Add the machine-fault dataset (same as Phase 1)        
    2. Download phase1_best.pth from Phase 1's output         
       → upload it as a NEW Kaggle dataset (e.g. "phase1ckpt")
       → add that dataset to this notebook                    
    3. Set PHASE1_CKPT below to the correct path                                                                    
Output: phase2_best.pth saved to /kaggle/working/           

"""

import os, json, math, pathlib, random, time
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
from dataclasses import dataclass, field
import soundfile as sf
from scipy.signal import resample_poly

# ── PATHS ─────────────────────────────────────────────────────────────────────
# ROOT_DIR: same dataset as Phase 1.
# The code auto-detects whether 'Machine 1/2/3' folders sit directly here or
# inside a 'Students' subfolder, so you don't need to change this line.
_DATASET_BASE = "/kaggle/input/datasets/mostafaehab41/machine-fault-dataset"

def _find_machine_root(base: str) -> str:
    """Return the directory that directly contains 'Machine 1', 'Machine 2', 'Machine 3'."""
    base_p = pathlib.Path(base)
    # Case 1: machine folders are directly in base
    if any((base_p / f"Machine {i}").exists() for i in range(1, 4)):
        return str(base_p)
    # Case 2: one level deeper (common Kaggle packaging adds a 'Students' folder)
    for sub in sorted(base_p.rglob("Machine 1")):
        return str(sub.parent)   # parent of 'Machine 1' is what we want
    print(f"WARNING: Could not auto-detect machine folders under {base}. Using base path.")
    return str(base_p)

ROOT_DIR    = _find_machine_root(_DATASET_BASE)
# Phase 1 checkpoint — download phase1_best.pth from Phase 1's output, upload
# it as a Kaggle dataset, add it to this notebook, then set the path below.
# Example path after uploading a dataset named "phase1ckpt":
PHASE1_CKPT = "/kaggle/input/datasets/manarabdelshafy/phase1-best-pth/phase1_best.pth"  
MODELS_DIR  = "/kaggle/working"

print(f"ROOT_DIR    : {ROOT_DIR}")
print(f"PHASE1_CKPT : {PHASE1_CKPT}")
print(f"Checkpoint exists: {pathlib.Path(PHASE1_CKPT).exists()}")

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SR = 16000; DURATION_SEC = 2.75; BATCH_SIZE = 32; EPOCHS = 15
NUM_WORKERS  = 4

# ── FEATURE CACHE — AUTO-DETECT ──────────────────────────────────────────────
# Upload your features dataset under your account (manarabdelshafy), any name.
# The script finds feats_mel/ and feats_mfcc/ automatically — no config needed.
# If not found, features are computed fresh (~53 min) and archived at the end.

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

FEATS_DIR_MEL  = _feat_dir("feats_mel")
FEATS_DIR_MFCC = _feat_dir("feats_mfcc")
CLASS_NAMES = ["Machine1_Normal","Machine1_Abnormal","Machine2_Normal",
               "Machine2_Abnormal","Machine3_Normal","Machine3_Abnormal"]

EPSILON = 1e-8

@dataclass
class AugmentationConfig:
    enabled:bool=True; noise_prob:float=0.35; noise_snr_db_min:float=15.0
    noise_snr_db_max:float=35.0; time_shift_prob:float=0.30
    time_shift_max_sec:float=0.20; pitch_shift_prob:float=0.20
    pitch_shift_min_semitones:float=-1.0; pitch_shift_max_semitones:float=1.0
    random_crop_train:bool=True

@dataclass
class PreprocessConfig:
    target_sr:int=16000; default_duration_sec:float=2.75; trim_silence:bool=True
    silence_threshold_ratio:float=0.02; trim_frame_ms:int=20; trim_hop_ms:int=10
    min_retained_sec:float=0.25; denoise:bool=False; normalize_mode:str="peak"
    peak_target:float=0.95; rms_target:float=0.10; clip_value:float=1.0
    augmentation:AugmentationConfig=field(default_factory=AugmentationConfig)

class AudioPreprocessor:
    def __init__(self, config=None): self.config = config or PreprocessConfig()
    def preprocess(self, audio_path, duration_sec=None, target_sr=None, mode="inference", seed=None):
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
def _minmax(S):
    lo,hi=S.min(),S.max()
    return np.zeros_like(S) if hi-lo<1e-8 else (S-lo)/(hi-lo)

def compute_mel_spectrogram(waveform, sr=16000):
    mel=librosa.feature.melspectrogram(y=waveform,sr=sr,n_mels=128,n_fft=1024,hop_length=512,fmin=50,fmax=8000,center=False)
    return np.expand_dims(_minmax(librosa.power_to_db(mel,ref=np.max)).astype(np.float32),0)

def compute_mfcc(waveform, sr=16000):
    mfcc=librosa.feature.mfcc(y=waveform,sr=sr,n_mfcc=40,n_fft=1024,hop_length=512)
    d=librosa.feature.delta(mfcc); d2=librosa.feature.delta(mfcc,order=2)
    feats=np.stack([mfcc,d,d2],axis=0)
    for i in range(3): feats[i]=_minmax(feats[i])
    return feats.astype(np.float32)

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
    """Pre-compute mel + mfcc for every wav file. Skips already-saved files (resumable).
    If either dir is under /kaggle/input (uploaded dataset), skips that dir entirely."""
    import multiprocessing, tqdm as tqdm_module
    mel_dir  = pathlib.Path(mel_dir)
    mfcc_dir = pathlib.Path(mfcc_dir)
    # Dirs under /kaggle/input are read-only uploaded datasets — nothing to compute
    if str(mel_dir).startswith("/kaggle/input") and str(mfcc_dir).startswith("/kaggle/input"):
        print(f"[cache] mel+mfcc loaded from uploaded dataset  ✓"); return
    mel_dir.mkdir(parents=True, exist_ok=True)
    mfcc_dir.mkdir(parents=True, exist_ok=True)
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

# ── DATASET (supports tuple feature_fn) ──────────────────────────────────────
# Actual folder structure:
#   machine-fault-dataset/machine1/Normal/*.wav   → label 0
#   machine-fault-dataset/machine1/Abnormal/*.wav → label 1  ... etc.
# parent.name = "Normal"/"Abnormal",  parent.parent.name = "machine1"/"machine2"/"machine3"

class MachineDataset(Dataset):
    LABEL_MAP={("machine1","Normal"):0,("machine1","Abnormal"):1,
               ("machine2","Normal"):2,("machine2","Abnormal"):3,
               ("machine3","Normal"):4,("machine3","Abnormal"):5}
    def __init__(self,root_dir,preprocessor,feature_fn,split,augment=False):
        self.root_dir=pathlib.Path(root_dir); self.preprocessor=preprocessor
        self.feature_fn=feature_fn; self.split=split; self.augment=augment
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
        waveform=self.preprocessor.preprocess(path,mode=mode)
        feats=self.feature_fn(waveform)
        lbl_t=torch.tensor(lbl,dtype=torch.long)
        if isinstance(feats,tuple):
            return tuple(torch.tensor(f,dtype=torch.float32) for f in feats), lbl_t
        return torch.tensor(feats,dtype=torch.float32), lbl_t

# Custom collate to handle tuple features
def collate_tuple_features(batch):
    """batch = list of ((mel_t, mfcc_t), lbl_t)  — stack each part separately."""
    feats_list, labels = zip(*batch)
    mel_batch  = torch.stack([f[0] for f in feats_list])
    mfcc_batch = torch.stack([f[1] for f in feats_list])
    return (mel_batch, mfcc_batch), torch.stack(labels)

class PrecomputedDataset2(Dataset):
    """Loads pre-computed mel + mfcc .npy files. SpecAugment applied to mel at train time."""
    def __init__(self, mel_dir, mfcc_dir, labels, indices, augment=False):
        self.mel_dir  = pathlib.Path(mel_dir)
        self.mfcc_dir = pathlib.Path(mfcc_dir)
        self.labels   = labels
        self.indices  = indices
        self.augment  = augment
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        ri   = self.indices[idx]
        mel  = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        mfcc = torch.tensor(np.load(self.mfcc_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment:
            mel = spec_augment(mel)
        return (mel, mfcc), torch.tensor(self.labels[ri], dtype=torch.long)

# ── MODELS ────────────────────────────────────────────────────────────────────
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

class MFCCStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4,4)))
        self.fc=nn.Linear(64*4*4,128)
    def forward(self,x):
        return F.relu(self.fc(torch.flatten(self.features(x),1)))

class MelMFCCCNN(nn.Module):
    def __init__(self,num_classes=6):
        super().__init__()
        self.mel_stream=MelCNN(num_classes); self.mfcc_stream=MFCCStream()
        self.fc1=nn.Linear(256+128,256); self.dropout=nn.Dropout(0.4); self.fc2=nn.Linear(256,num_classes)
    def forward(self,mel,mfcc):
        x=torch.cat([self.mel_stream.extract_features(mel), self.mfcc_stream(mfcc)],dim=1)
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# ── TRAIN / EVAL ──────────────────────────────────────────────────────────────
def train_epoch_dual(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for (mel, mfcc), y in loader:
        mel,mfcc,y = mel.to(device),mfcc.to(device),y.to(device)
        optimizer.zero_grad()
        out=model(mel,mfcc); loss=criterion(out,y)
        loss.backward(); optimizer.step()
        total_loss+=loss.item(); correct+=(out.argmax(1)==y).sum().item(); total+=y.size(0)
    return total_loss/len(loader), correct/total

def eval_epoch_dual(model, loader, criterion, device):
    model.eval(); total_loss,correct,total=0.0,0,0; all_p,all_l=[],[]
    with torch.no_grad():
        for (mel,mfcc),y in loader:
            mel,mfcc,y=mel.to(device),mfcc.to(device),y.to(device)
            out=model(mel,mfcc); loss=criterion(out,y)
            total_loss+=loss.item(); preds=out.argmax(1)
            correct+=(preds==y).sum().item(); total+=y.size(0)
            all_p.extend(preds.cpu().numpy()); all_l.extend(y.cpu().numpy())
    return total_loss/len(loader), correct/total, all_p, all_l

def save_ckpt(model,optimizer,epoch,val_acc,path):
    torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                "epoch":epoch,"val_acc":val_acc},path)

def plot_cm(preds,labels,class_names):
    cm=confusion_matrix(labels,preds)
    plt.figure(figsize=(8,6)); sns.heatmap(cm,annot=True,fmt="d",cmap="Purples",
        xticklabels=class_names,yticklabels=class_names)
    plt.ylabel("True Label"); plt.xlabel("Predicted Label"); plt.tight_layout(); plt.show()

# ── MAIN ──────────────────────────────────────────────────────────────────────
# Step A: scan all files + create/load the stratified split
_scan_ds   = MachineDataset(ROOT_DIR,
                 AudioPreprocessor(PreprocessConfig(augmentation=AugmentationConfig(enabled=False))),
                 compute_mel_spectrogram, "train")
ALL_PATHS  = _scan_ds.paths
ALL_LABELS = _scan_ds.labels

# Step B: pre-compute mel + mfcc once (~20 min, then cached every run)
_infer_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=SR, default_duration_sec=DURATION_SEC, trim_silence=True, normalize_mode="peak",
    augmentation=AugmentationConfig(enabled=False),
))
precompute_all_dual(ALL_PATHS, FEATS_DIR_MEL, FEATS_DIR_MFCC, _infer_prep, n_workers=NUM_WORKERS)

# Step C: load split indices (created in Phase 1 or just now)
_splits = json.load(open(pathlib.Path(MODELS_DIR) / "split_indices.json"))

train_ds = PrecomputedDataset2(FEATS_DIR_MEL, FEATS_DIR_MFCC, ALL_LABELS, _splits["train"], augment=True)
val_ds   = PrecomputedDataset2(FEATS_DIR_MEL, FEATS_DIR_MFCC, ALL_LABELS, _splits["val"],   augment=False)
test_ds  = PrecomputedDataset2(FEATS_DIR_MEL, FEATS_DIR_MFCC, ALL_LABELS, _splits["test"],  augment=False)
print(f"Train:{len(train_ds)}  Val:{len(val_ds)}  Test:{len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_tuple_features,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_tuple_features,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_tuple_features,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)

# Build model and load Phase 1 mel_stream weights
model = MelMFCCCNN(num_classes=6).to(DEVICE)
p1_ckpt = torch.load(PHASE1_CKPT, map_location=DEVICE)
model.mel_stream.load_state_dict(p1_ckpt["model_state_dict"])
print(f"Loaded Phase 1 weights into mel_stream  (epoch {p1_ckpt['epoch']}, val_acc {p1_ckpt['val_acc']:.4f}) \u2713")

optimizer = torch.optim.AdamW([
    {"params": model.mel_stream.parameters(),  "lr": 1e-4},
    {"params": model.mfcc_stream.parameters(), "lr": 5e-4},
    {"params": model.fc1.parameters(),         "lr": 5e-4},
    {"params": model.fc2.parameters(),         "lr": 5e-4},
], weight_decay=1e-4)

label_counts  = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
class_weights = torch.tensor(1.0/(label_counts+1), dtype=torch.float32).to(DEVICE)
criterion     = nn.CrossEntropyLoss(weight=class_weights)
scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc=0.0; history={"loss":[],"acc":[],"val_loss":[],"val_acc":[]}

print("\n\u2500\u2500 Training Phase 2 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
for epoch in range(1,EPOCHS+1):
    tr_loss,tr_acc      = train_epoch_dual(model,train_loader,optimizer,criterion,DEVICE)
    vl_loss,vl_acc,_,_  = eval_epoch_dual( model,val_loader,  criterion,DEVICE)
    scheduler.step()
    history["loss"].append(tr_loss); history["acc"].append(tr_acc)
    history["val_loss"].append(vl_loss); history["val_acc"].append(vl_acc)
    print(f"Epoch {epoch:3d}/{EPOCHS}  train_acc={tr_acc:.4f}  val_acc={vl_acc:.4f}",end="")
    if vl_acc>best_val_acc:
        best_val_acc=vl_acc; ckpt_path=os.path.join(MODELS_DIR,"phase2_best.pth")
        save_ckpt(model,optimizer,epoch,vl_acc,ckpt_path); print("  \u2190 saved",end="")
    print()

print(f"\nBest val accuracy: {best_val_acc:.4f}")

# ┌─────────────────────────────────────────────────────────────────┐
# │  CHECKPOINT SAVED TO:  /kaggle/working/phase2_best.pth          │
# │  Keys stored: model_state_dict · optimizer_state_dict ·         │
# │               epoch · val_acc                                   │
# │  → Download from Kaggle Output tab, upload as a new dataset,    │
# │    then set  PHASE2_CKPT  in kaggle_phase3.py.                  │
# └─────────────────────────────────────────────────────────────────┘
ckpt=torch.load(os.path.join(MODELS_DIR,"phase2_best.pth"),map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_,test_acc,preds,labels_t=eval_epoch_dual(model,test_loader,criterion,DEVICE)
t_test = time.time() - t0
n_test = len(test_ds)
ms_per_sample = (t_test / n_test) * 1000   # milliseconds per sample

print(f"\n── Test Results ──────────────────────────────────────────")
print(f"Test accuracy : {test_acc:.4f}")
print(f"Macro F1      : {f1_score(labels_t,preds,average='macro'):.4f}")
print("\n",classification_report(labels_t,preds,target_names=CLASS_NAMES))

print(f"\n── Inference Timing ──────────────────────────────────────")
print(f"Test set size         : {n_test} samples")
print(f"Total inference time  : {t_test:.2f} s")
print(f"Per-sample time       : {ms_per_sample:.3f} ms  →  {1000/ms_per_sample:.0f} samples/sec")
print(f"Estimated   100 files : {ms_per_sample *   100 / 1000:.2f} s")
print(f"Estimated 1 000 files : {ms_per_sample *  1000 / 1000:.2f} s")
print(f"Estimated 10 000 files: {ms_per_sample * 10000 / 1000:.2f} s")

plot_cm(preds,labels_t,CLASS_NAMES)

# ── ARCHIVE MFCC FEATURES FOR REUSE IN PHASE 3 ───────────────────────────────
# After this runs, download BOTH archives and add them to your features dataset:
#   feats_mel_archive.zip  — if not already uploaded from Phase 1
#   feats_mfcc_archive.zip — new from this phase
# Re-upload the dataset with feats_mel/ + feats_mfcc/ inside.
import shutil
for folder, archive in [("feats_mel","feats_mel_archive"), ("feats_mfcc","feats_mfcc_archive")]:
    src = pathlib.Path("/kaggle/working") / folder
    if src.exists() and any(src.glob("*.npy")):   # only archive if freshly computed this session
        print(f"Archiving {folder} ...")
        shutil.make_archive(f"/kaggle/working/{archive}", "zip", "/kaggle/working", folder)
        sz = os.path.getsize(f"/kaggle/working/{archive}.zip") / 1e9
        print(f"  {archive}.zip  ({sz:.2f} GB)  → /kaggle/working/")

print(f"\nDownload phase2_best.pth from /kaggle/working/ and upload as dataset 'phase2ckpt' for Phase 3.")
