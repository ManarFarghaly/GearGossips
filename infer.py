"""
infer.py — Machine Fault Recognition inference script
Usage: python infer.py <path_to_data_directory>

Reads all *.wav files from the given directory in numeric order (1.wav, 2.wav, ...),
runs the Phase 2b V4 model on each, and writes two output files:
  results.txt — one predicted label (0–5) per line
  time.txt    — one processing time (seconds, 3 dp) per line

The model checkpoint (phase2b_v4_best.pth) must be in the same directory as
this script. It embeds the per-machine scalers and feature list so no separate
scaler file is needed.

Labels:
  0 = Machine 1, Normal      1 = Machine 1, Abnormal
  2 = Machine 2, Normal      3 = Machine 2, Abnormal
  4 = Machine 3, Normal      5 = Machine 3, Abnormal
"""

import sys
import os
import re
import time
import math
import pathlib
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf
from scipy.signal import resample_poly
from scipy.stats import kurtosis as scipy_kurtosis

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE       = pathlib.Path(__file__).parent
MODEL_PATH = HERE / "phase2b_v4_best.pth"

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

EPSILON = 1e-8

TARGET_SR   = 16000
DURATION    = 2.75                          # seconds
TARGET_LEN  = int(round(TARGET_SR * DURATION))  # 44 000 samples

SILENCE_THRESHOLD_RATIO = 0.02
TRIM_FRAME_MS           = 20
TRIM_HOP_MS             = 10
MIN_RETAINED_SEC        = 0.25
PEAK_TARGET             = 0.95


def _load_audio(path):
    """Read a wav file and return (samples float32, original_sr)."""
    try:
        data, orig_sr = sf.read(str(path), always_2d=False, dtype="float32")
    except Exception:
        return np.zeros(TARGET_LEN, dtype=np.float32), TARGET_SR
    w = np.asarray(data, dtype=np.float32)
    if w.ndim > 1:
        w = w.mean(axis=1)
    return w, orig_sr


def _resample(w, orig_sr, target_sr):
    if orig_sr == target_sr:
        return w
    d = math.gcd(orig_sr, target_sr)
    return resample_poly(w, target_sr // d, orig_sr // d).astype(np.float32)


def _trim_silence(w, sr):
    if w.size == 0:
        return w
    peak = np.abs(w).max()
    if peak <= EPSILON:
        return w
    thr = peak * SILENCE_THRESHOLD_RATIO
    fl  = max(1, int(sr * TRIM_FRAME_MS / 1000))
    hl  = max(1, int(sr * TRIM_HOP_MS   / 1000))
    active = [s for s in range(0, w.size - fl + 1, hl)
              if np.abs(w[s:s + fl]).max() >= thr]
    if not active:
        return w
    trimmed = w[active[0]: min(w.size, active[-1] + fl)]
    if trimmed.size >= int(MIN_RETAINED_SEC * sr):
        return trimmed.astype(np.float32)
    return w


def _peak_normalize(w):
    p = np.abs(w).max()
    if p > EPSILON:
        w = w * (PEAK_TARGET / p)
    return w.astype(np.float32)


def _fix_length(w, tgt):
    cur = w.size
    if cur > tgt:
        start = (cur - tgt) // 2
        return w[start: start + tgt]
    if cur < tgt:
        return np.pad(w, (0, tgt - cur))
    return w


def preprocess(raw_w, orig_sr):
    """Full preprocessing pipeline: resample, trim, normalize, fix length."""
    w = _resample(raw_w, orig_sr, TARGET_SR)
    w = _trim_silence(w, TARGET_SR)
    w = _peak_normalize(w)
    w = _fix_length(w, TARGET_LEN)
    np.clip(np.nan_to_num(w, 0.0), -1.0, 1.0, out=w)
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _minmax(S):
    lo, hi = S.min(), S.max()
    return np.zeros_like(S) if hi - lo < 1e-8 else (S - lo) / (hi - lo)


def compute_mel(w, sr=TARGET_SR):
    """Returns (1, 128, 84) mel-spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=w, sr=sr, n_mels=128, n_fft=1024, hop_length=512,
        fmin=50, fmax=8000, center=False
    )
    return np.expand_dims(_minmax(librosa.power_to_db(mel, ref=np.max)).astype(np.float32), 0)


def compute_stat(w, sr=TARGET_SR):
    """Returns 19-d statistical feature vector used by Phase 2b V3/V4."""
    S    = np.abs(librosa.stft(w, n_fft=1024, hop_length=512))
    flux = float(np.mean(np.sum(np.diff(S, axis=1) ** 2, axis=0)))
    mfcc = librosa.feature.mfcc(y=w, sr=sr, n_mfcc=13).mean(axis=1)
    base = np.array([
        float(np.sqrt(np.mean(w ** 2))),
        float(librosa.feature.zero_crossing_rate(w).mean()),
        float(librosa.feature.spectral_rolloff(y=w, sr=sr).mean()),
        float(librosa.feature.spectral_bandwidth(y=w, sr=sr).mean()),
        flux,
        float(scipy_kurtosis(w, fisher=True)),
    ], dtype=np.float32)
    return np.concatenate([base, mfcc.astype(np.float32)])


# ---------------------------------------------------------------------------
# Model — must match the Phase 2b V4 checkpoint exactly
# ---------------------------------------------------------------------------

class AttentionPool2d(nn.Module):
    def __init__(self, in_channels, out_size=(4, 4)):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1), nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(out_size)

    def forward(self, x):
        w = torch.softmax(self.attn(x).flatten(2), dim=-1)
        return self.pool(x * w.view(x.shape[0], 1, x.shape[2], x.shape[3]))


class MelCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1,   32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                    AttentionPool2d(256, (4, 4)))
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.head_main    = nn.Linear(256, num_classes)
        self.head_machine = nn.Linear(256, 3)
        self.head_fault   = nn.Linear(256, 1)

    def extract_features(self, x):
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        return self.dropout(F.relu(self.fc1(torch.flatten(x, 1))))

    def forward(self, x):
        feat = self.extract_features(x)
        return self.head_main(feat), self.head_machine(feat), self.head_fault(feat)


class MelStatCNN(nn.Module):
    def __init__(self, num_classes=6, stat_dim=19):
        super().__init__()
        self.mel_stream  = MelCNN(num_classes)
        self.stat_branch = nn.Sequential(
            nn.Linear(stat_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.fc1     = nn.Linear(256 + 64, 256)
        self.dropout = nn.Dropout(0.5)
        self.head_main    = nn.Linear(256, num_classes)
        self.head_machine = nn.Linear(256, 3)
        self.head_fault   = nn.Linear(256, 1)

    def forward(self, mel, stat):
        f_mel  = self.mel_stream.extract_features(mel)
        f_stat = self.stat_branch(stat)
        fused  = self.dropout(F.relu(self.fc1(torch.cat([f_mel, f_stat], dim=1))))
        return self.head_main(fused), self.head_machine(fused), self.head_fault(fused)


# ---------------------------------------------------------------------------
# Numeric file sorting: 1.wav < 2.wav < 10.wav (not lexicographic)
# ---------------------------------------------------------------------------

def _numeric_key(path):
    m = re.search(r"\d+", path.stem)
    return int(m.group()) if m else 0


# ---------------------------------------------------------------------------
# Inference helper: two-pass per-machine stat normalization
# ---------------------------------------------------------------------------

def _predict(model, mel_t, stat_raw, scalers, device):
    """
    Two-pass prediction using per-machine scalers.

    Pass 1: normalize stat with the average of all three machine scalers,
            run the model to predict which machine the clip belongs to.
    Pass 2: normalize stat with the predicted machine's own scaler,
            run again to get the final class prediction.

    This avoids needing the true machine label at inference time while still
    benefiting from the per-machine normalization used during training.
    """
    # Build a global (averaged) scaler from the three machine scalers
    global_mean = np.mean([s[0] for s in scalers], axis=0)
    global_std  = np.mean([s[1] for s in scalers], axis=0)

    # Pass 1 — predict machine identity
    stat_g = torch.tensor(
        (stat_raw - global_mean) / global_std, dtype=torch.float32
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        _, machine_logits, _ = model(mel_t, stat_g)
    predicted_machine = int(machine_logits.argmax(1).item())

    # Pass 2 — predict class using the predicted machine's scaler
    mean, std = scalers[predicted_machine]
    stat_m = torch.tensor(
        (stat_raw - mean) / std, dtype=torch.float32
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        class_logits, _, _ = model(mel_t, stat_m)
    return int(class_logits.argmax(1).item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py <data_directory>")
        sys.exit(1)

    data_dir = pathlib.Path(sys.argv[1])
    if not data_dir.exists():
        raise SystemExit(f"Directory not found: {data_dir}")

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model file not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint — includes model weights, scalers, and feature metadata
    ckpt    = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    scalers = ckpt["scalers"]          # list of (mean, std) per machine
    stat_dim = ckpt.get("stat_dim", len(scalers[0][0]))

    model = MelStatCNN(num_classes=6, stat_dim=stat_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Collect wav files in numeric order: 1.wav, 2.wav, 3.wav ...
    wav_files = sorted(
        [f for f in data_dir.iterdir() if f.suffix.lower() == ".wav"],
        key=_numeric_key,
    )

    results = []
    times   = []

    for wav_path in wav_files:
        # ── Pure I/O: read file bytes into memory ──────────────────────────
        raw_w, orig_sr = _load_audio(wav_path)

        # ── Start timer AFTER file read, before any processing ─────────────
        t_start = time.perf_counter()

        # Preprocessing
        waveform = preprocess(raw_w, orig_sr)

        # Feature extraction
        mel_t = torch.tensor(compute_mel(waveform), dtype=torch.float32).unsqueeze(0).to(device)
        stat  = compute_stat(waveform)

        # Inference
        pred = _predict(model, mel_t, stat, scalers, device)

        t_end = time.perf_counter()
        # ── End timer ──────────────────────────────────────────────────────

        results.append(pred)
        times.append(t_end - t_start)

    # Write results.txt — one predicted label per line
    with open("results.txt", "w") as f:
        for r in results:
            f.write(f"{r}\n")

    # Write time.txt — one elapsed time per line, rounded to 3 decimal places
    with open("time.txt", "w") as f:
        for t in times:
            f.write(f"{t:.3f}\n")
