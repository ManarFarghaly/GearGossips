"""
Final Inference Script — infer.py
Usage: python infer.py <path_to_data_directory>

Reads all *.wav files from data/ in numeric order (1.wav, 2.wav, ...),
runs the Phase 3 model on each, and writes:
  results.txt  — one predicted label (0-5) per line
  time.txt     — one processing time (seconds, 3 dp) per line

The model file (phase3_best.pth) and scaler file (stat_scaler.pkl) must be
in the same directory as this script, or set MODEL_PATH / SCALER_PATH below.
"""

import sys, os, re, time, math, pathlib, json, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf
from scipy.signal import resample_poly
from dataclasses import dataclass, field

# ── PATHS ─────────────────────────────────────────────────────────────────────
HERE        = pathlib.Path(__file__).parent
MODEL_PATH  = HERE / "phase3_best.pth"
SCALER_PATH = HERE / "stat_scaler.pkl"

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
EPSILON = 1e-8

@dataclass
class AugmentationConfig:
    enabled: bool = False   # always off at inference

@dataclass
class PreprocessConfig:
    target_sr: int = 16000; default_duration_sec: float = 2.75
    trim_silence: bool = True; silence_threshold_ratio: float = 0.02
    trim_frame_ms: int = 20; trim_hop_ms: int = 10; min_retained_sec: float = 0.25
    denoise: bool = False; normalize_mode: str = "peak"; peak_target: float = 0.95
    clip_value: float = 1.0; augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

class AudioPreprocessor:
    def __init__(self, config=None): self.config = config or PreprocessConfig()
    def preprocess(self, audio_path, **_):
        cfg=self.config; eff_sr=cfg.target_sr; tgt_len=int(round(eff_sr*cfg.default_duration_sec))
        try: data,orig_sr=sf.read(str(audio_path),always_2d=False,dtype="float32")
        except: return np.zeros(tgt_len,dtype=np.float32)
        w=np.asarray(data,dtype=np.float32)
        if w.ndim>1: w=w.mean(axis=1)
        if orig_sr!=eff_sr:
            d=math.gcd(orig_sr,eff_sr); w=resample_poly(w,eff_sr//d,orig_sr//d).astype(np.float32)
        if cfg.trim_silence and w.size>0:
            peak=np.abs(w).max()
            if peak>EPSILON:
                thr=peak*cfg.silence_threshold_ratio; fl=max(1,int(eff_sr*cfg.trim_frame_ms/1000)); hl=max(1,int(eff_sr*cfg.trim_hop_ms/1000))
                aw=np.abs(w); active=[s for s in range(0,w.size-fl+1,hl) if aw[s:s+fl].max()>=thr]
                if active:
                    trimmed=w[active[0]:min(w.size,active[-1]+fl)]
                    if trimmed.size>=int(cfg.min_retained_sec*eff_sr): w=trimmed.astype(np.float32)
        p=np.abs(w).max()
        if p>EPSILON: w=w*(cfg.peak_target/p)
        cur=w.size
        if cur>tgt_len: w=w[(cur-tgt_len)//2:(cur-tgt_len)//2+tgt_len]
        elif cur<tgt_len: w=np.pad(w,(0,tgt_len-cur))
        np.clip(w,-cfg.clip_value,cfg.clip_value,out=w)
        return w.astype(np.float32)

# ── FEATURE FUNCTIONS ─────────────────────────────────────────────────────────
def _mm(S):
    lo,hi=S.min(),S.max()
    return np.zeros_like(S) if hi-lo<1e-8 else (S-lo)/(hi-lo)

def compute_mel_spectrogram(w,sr=16000):
    mel=librosa.feature.melspectrogram(y=w,sr=sr,n_mels=128,n_fft=1024,hop_length=512,fmin=50,fmax=8000,center=False)
    return np.expand_dims(_mm(librosa.power_to_db(mel,ref=np.max)).astype(np.float32),0)

def compute_mfcc(w,sr=16000):
    mfcc=librosa.feature.mfcc(y=w,sr=sr,n_mfcc=40,n_fft=1024,hop_length=512)
    d=librosa.feature.delta(mfcc); d2=librosa.feature.delta(mfcc,order=2)
    feats=np.stack([mfcc,d,d2],axis=0)
    for i in range(3): feats[i]=_mm(feats[i])
    return feats.astype(np.float32)

def compute_statistical_features(w, sr=16000, feature_names=None):
    if feature_names is None: feature_names=["rms","zcr","centroid","rolloff","bandwidth"]
    result=[]
    for name in feature_names:
        if name=="rms":        result.append(float(np.sqrt(np.mean(w**2))))
        elif name=="zcr":      result.append(float(librosa.feature.zero_crossing_rate(w).mean()))
        elif name=="centroid": result.append(float(librosa.feature.spectral_centroid(y=w,sr=sr).mean()))
        elif name=="rolloff":  result.append(float(librosa.feature.spectral_rolloff(y=w,sr=sr).mean()))
        elif name=="bandwidth":result.append(float(librosa.feature.spectral_bandwidth(y=w,sr=sr).mean()))
    return np.array(result,dtype=np.float32)

# ── MODELS (must match Phase 3 architecture exactly) ─────────────────────────
class MelCNN(nn.Module):
    def __init__(self,num_classes=6):
        super().__init__()
        self.block1=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2))
        self.block2=nn.Sequential(nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2))
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
        self.features=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),nn.AdaptiveAvgPool2d((4,4)))
        self.fc=nn.Linear(64*4*4,128)
    def forward(self,x): return F.relu(self.fc(torch.flatten(self.features(x),1)))

class MelMFCCStatCNN(nn.Module):
    def __init__(self,num_classes=6,stat_dim=5):
        super().__init__()
        self.mel_stream=MelCNN(num_classes); self.mfcc_stream=MFCCStream()
        self.stat_branch=nn.Sequential(nn.Linear(stat_dim,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.fc1=nn.Linear(256+128+32,256); self.dropout=nn.Dropout(0.4); self.fc2=nn.Linear(256,num_classes)
    def forward(self,mel,mfcc,stat):
        f_mel=self.mel_stream.extract_features(mel); f_mfcc=self.mfcc_stream(mfcc); f_stat=self.stat_branch(stat)
        return self.fc2(self.dropout(F.relu(self.fc1(torch.cat([f_mel,f_mfcc,f_stat],dim=1)))))

# ── NUMERIC FILE SORTING ──────────────────────────────────────────────────────
def numeric_key(path):
    m=re.search(r"\d+",path.stem)
    return int(m.group()) if m else 0

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py <data_directory>"); sys.exit(1)

    data_dir = pathlib.Path(sys.argv[1])
    if not data_dir.exists():
        raise SystemExit(f"Directory not found: {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load scaler
    scaler_data   = pickle.load(open(SCALER_PATH,"rb"))
    scaler_mean   = torch.tensor(scaler_data["mean"], dtype=torch.float32).to(device)
    scaler_std    = torch.tensor(scaler_data["std"],  dtype=torch.float32).to(device)
    stat_features = scaler_data["features"]
    stat_dim      = len(stat_features)

    # Load model
    ckpt  = torch.load(MODEL_PATH, map_location=device)
    model = MelMFCCStatCNN(num_classes=6, stat_dim=stat_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded (stat_dim={stat_dim}, features={stat_features})")

    preprocessor = AudioPreprocessor(PreprocessConfig())

    # Get files in numeric order: 1.wav, 2.wav, 3.wav ...
    wav_files = sorted(
        [f for f in data_dir.iterdir() if f.suffix.lower()==".wav"],
        key=numeric_key
    )
    print(f"Found {len(wav_files)} wav files")

    results, times = [], []

    for wav_path in wav_files:
        # Start timer AFTER reading the file (as per spec)
        waveform = preprocessor.preprocess(wav_path)
        t_start  = time.perf_counter()

        mel_t  = torch.tensor(compute_mel_spectrogram(waveform),  dtype=torch.float32).unsqueeze(0).to(device)
        mfcc_t = torch.tensor(compute_mfcc(waveform),             dtype=torch.float32).unsqueeze(0).to(device)
        stat_t = torch.tensor(compute_statistical_features(waveform, feature_names=stat_features), dtype=torch.float32).unsqueeze(0).to(device)
        stat_t = (stat_t - scaler_mean) / scaler_std

        with torch.no_grad():
            logits = model(mel_t, mfcc_t, stat_t)
            pred   = int(logits.argmax(1).item())

        t_end   = time.perf_counter()
        elapsed = t_end - t_start

        results.append(pred)
        times.append(elapsed)

    # Write results.txt
    out_results = data_dir.parent / "results.txt"
    with open(out_results, "w") as f:
        for r in results:
            f.write(f"{r}\n")

    # Write time.txt
    out_times = data_dir.parent / "time.txt"
    with open(out_times, "w") as f:
        for t in times:
            f.write(f"{t:.3f}\n")

    print(f"results.txt → {out_results}")
    print(f"time.txt    → {out_times}")
    print("Done.")
