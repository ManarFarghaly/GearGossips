# GearGossips — Machine Fault Detection via Audio

**Dataset**: [Machine Fault Dataset (Kaggle)](https://www.kaggle.com/datasets/mostafaehab41/machine-fault-dataset)

A multi-phase deep learning pipeline that classifies industrial machine audio recordings into **6 classes** — Normal vs Abnormal for each of 3 machines — using mel-spectrograms, statistical features, and hierarchical CNN architectures.

---

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [Dataset Structure](#dataset-structure)
3. [Project Layout](#project-layout)
4. [Environment Setup](#environment-setup)
5. [Data Splitting Strategy](#data-splitting-strategy)
6. [Audio Preprocessing Pipeline](#audio-preprocessing-pipeline)
7. [Feature Extraction](#feature-extraction)
8. [Phase 1 — Mel-Spectrogram CNN Baseline](#phase-1--mel-spectrogram-cnn-baseline)
   - [Phase 1 V1 — Baseline](#phase-1-v1--baseline)
   - [Phase 1 V2 — Imbalance & Overfitting Fixes](#phase-1-v2--imbalance--overfitting-fixes)
   - [Phase 1 V3 — Machine3 Collapse Fix](#phase-1-v3--machine3-collapse-fix)
   - [Phase 1 V4 — Kaggle variant of V3](#phase-1-v4--kaggle-variant-of-v3)
9. [Phase 2b — Mel + Statistical Features](#phase-2b--mel--statistical-features)
   - [Phase 2b V1 — Flat head, global norm](#phase-2b-v1--flat-head-global-norm)
   - [Phase 2b V2 — Hierarchical heads](#phase-2b-v2--hierarchical-heads)
10. [Ablation Studies](#ablation-studies)
11. [Diagnostic Notebooks](#diagnostic-notebooks)
12. [Inference](#inference)
13. [How to Run on Kaggle (Step by Step)](#how-to-run-on-kaggle-step-by-step)
14. [Class Imbalance: Why SMOTE Was Not Used](#class-imbalance-why-smote-was-not-used)
15. [File Reference Table](#file-reference-table)

---

## Problem Overview

Three industrial machines are monitored via audio. Each recording is a short `.wav` clip. The task is to classify each clip into one of **6 classes**:

| Label | Class              |
|-------|--------------------|
| 0     | Machine1 — Normal  |
| 1     | Machine1 — Abnormal|
| 2     | Machine2 — Normal  |
| 3     | Machine2 — Abnormal|
| 4     | Machine3 — Normal  |
| 5     | Machine3 — Abnormal|

**Key challenge**: The dataset is **imbalanced** — Machine1 has ~2430 normal samples while Machine3 Abnormal has ~453. More critically, the model struggled to distinguish Normal vs Abnormal within Machine2 and Machine3 (it would predict nearly everything as "Abnormal" for those machines).

---

## Dataset Structure

Download the dataset from Kaggle and place it so the structure looks like:

```
Students/                          ← ROOT_DIR (the folder you point scripts at)
├── machine1/
│   ├── Normal/
│   │   ├── 1.wav
│   │   ├── 2.wav
│   │   └── ...
│   └── Abnormal/
│       ├── 1.wav
│       └── ...
├── machine2/
│   ├── Normal/
│   │   └── ...
│   └── Abnormal/
│       └── ...
└── machine3/
    ├── Normal/
    │   └── ...
    └── Abnormal/
        └── ...
```

> **On Kaggle**: Upload the dataset and set `ROOT_DIR` in each script to the `/kaggle/input/<dataset-name>/` path.

---

## Project Layout

```
GearGossips/
│
├── machine_listener/                   # Core library (shared across all phases)
│   └── src/
│       ├── preprocess.py               # Audio preprocessing pipeline
│       ├── dataset.py                  # PyTorch Dataset + label map
│       ├── split_utils.py              # Chronological, leak-free data splitting
│       ├── train_utils.py              # Train/eval loops, metrics, plotting
│       ├── features/
│       │   ├── mel_spectrogram.py      # Mel-spectrogram extractor → (1, 128, 84)
│       │   ├── MFCC.py                 # MFCC + deltas extractor → (3, 40, 84)
│       │   └── statistical.py          # Global stat features (RMS, ZCR, etc.)
│       └── models/
│           ├── cnn_baseline.py         # MelCNN (Phase 1) + MelCNNHier (Phase 1 V3)
│           ├── cnn_MFCC.py             # MFCCStream sub-network
│           ├── cnn_MelMFCC.py          # Dual-stream Mel+MFCC (Phase 2, not used in final)
│           ├── cnn_mel_stat.py         # MelStatCNN + MelStatCNNHier (Phase 2b)
│           └── cnn_statistical.py      # Mel+MFCC+Stat (Phase 3, used by infer.py)
│
├── train_phase1.py                     # Phase 1 V1 — local training
├── train_phase1V2.py                   # Phase 1 V2 — local training
├── train_phase1V3.py                   # Phase 1 V3 — local training
├── train_phase1_ablation.py            # Phase 1 ablation (local)
├── train_phase2b.py                    # Phase 2b V1 — local training
├── train_phase2bV2.py                  # Phase 2b V2 — local training
├── train_phase2b_ablation.py           # Phase 2b ablation (local)
│
├── kaggle_phase1V1.py                  # ⬆ Self-contained Kaggle notebook scripts
├── kaggle_phase1V2.py                  #   (inline all code — no imports from
├── kaggle_phase1V3.py                  #    machine_listener; copy-paste into
├── kaggle_phase1V4.py                  #    a Kaggle notebook cell and run)
├── kaggle_phase1_ablation.py           #
├── kaggle_phase2b.py                   #
├── kaggle_phase2b_v2.py                #
├── kaggle_phase2b_ablation.py          #
├── kaggle_diagnose_leakage.py          # Diagnostic: temporal leakage detection
├── kaggle_diagnose_overfit_bias.py     # Diagnostic: overfitting & bias checks
│
├── infer.py                            # Final inference script
├── how_to_use.py                       # Quick demo of the preprocessing pipeline
├── requirements.txt                    # Python dependencies
│
├── ResultsFiles/                       # Saved results, checkpoints, screenshots
│   ├── PhaseOne/                       #   Phase 1 V1 checkpoint + screenshots
│   ├── phase2b/                        #   Phase 2b checkpoint + scaler + split
│   ├── PhaseOneTuning.md               #   Detailed diagnosis notes
│   └── ...
└── NewResults/                         # Results from each version iteration
    ├── Phase1V1/ Phase1V2/ ...
    ├── phase2b/ phase2bv2/
    └── ...
```

---

## Environment Setup

### Local

```bash
pip install -r requirements.txt
# Also needed (not in requirements.txt):
pip install torch torchvision torchaudio
pip install scikit-learn matplotlib seaborn tqdm
```

### Kaggle

The `kaggle_*.py` scripts are **self-contained** — they inline all the code from `machine_listener/src/` so you don't need to upload the library. Just:

1. Create a new Kaggle notebook
2. Enable **GPU** (Settings → Accelerator → GPU T4 ×2)
3. Add the [Machine Fault Dataset](https://www.kaggle.com/datasets/mostafaehab41/machine-fault-dataset) as input
4. Paste the contents of the relevant `kaggle_*.py` file into a cell
5. Update `ROOT_DIR` to match your Kaggle input path
6. Run

---

## Data Splitting Strategy

**File**: `machine_listener/src/split_utils.py`

The split is **chronological** (not random!) to prevent data leakage:

| Concern                 | Solution                                                                 |
|------------------------|--------------------------------------------------------------------------|
| **Temporal leakage**    | Files are sorted numerically within each class, then split 70/15/15. Adjacent clips from the same recording session always stay in the same split. |
| **Duplicate leakage**   | Exact duplicate files (detected via file-size + MD5 hash) are forced into the same split. |
| **Ratios**              | 70% train / 15% validation / 15% test, applied per-class.               |

The split is created **once** by Phase 1 (saved as `split_indices_clean.json`) and **loaded** by every subsequent phase. This ensures all phases use the exact same train/val/test samples.

---

## Audio Preprocessing Pipeline

**File**: `machine_listener/src/preprocess.py`

Every `.wav` file goes through this pipeline:

```
Raw .wav → Load → Mono → Resample to 16kHz → (Optional denoise)
        → Trim silence → Peak normalize → (Training: augmentations)
        → Fix length to 2.75s (44,000 samples) → Clip & sanitize
```

| Step              | Detail                                                                   |
|-------------------|--------------------------------------------------------------------------|
| **Sample rate**   | Resample to 16 kHz using polyphase filtering                             |
| **Duration**      | Fixed at 2.75 seconds = 44,000 samples                                   |
| **Silence trim**  | Frame-wise energy detection (20ms frames, 10ms hop) removes leading/trailing silence |
| **Normalization** | Peak normalization to 0.95 amplitude                                      |
| **Training augmentations** | Gaussian noise (35% prob, 15–35 dB SNR), time shift (30% prob, ±0.2s), pitch shift (20% prob, ±1 semitone), random crop |
| **Inference**     | Deterministic: center crop, no augmentation                               |

---

## Feature Extraction

### 1. Mel-Spectrogram (all phases)

**File**: `machine_listener/src/features/mel_spectrogram.py`

```
Waveform (44000,) → librosa.melspectrogram → power_to_dB → min-max normalize → (1, 128, 84)
```

- 128 mel frequency bins, FFT window = 1024, hop = 512
- Frequency range: 50–8000 Hz
- Output shape: `(1, 128, 84)` — 1 channel (grayscale), 128 freq bins, 84 time frames

### 2. MFCC + Deltas (Phase 2 / Phase 3 only)

**File**: `machine_listener/src/features/MFCC.py`

```
Waveform → 40 MFCCs + delta + delta-delta → (3, 40, 84)
```

- Channel 0: Raw MFCCs (where energy is)
- Channel 1: Delta (velocity of change)
- Channel 2: Delta-delta (acceleration of change)
- Normal machines have steady patterns → small deltas; faulty machines → large deltas

### 3. Statistical Features (Phase 2b and later)

**File**: `machine_listener/src/features/statistical.py`

Global scalar descriptors per clip:

| Feature    | What it captures                                | Version |
|------------|------------------------------------------------|---------|
| RMS        | Overall energy/loudness                         | v1, v2, v3 |
| ZCR        | Zero-crossing rate — noisiness indicator         | v1, v2, v3 |
| Centroid   | Spectral "center of mass" (dropped in v2 — redundant with mel CNN) | v1 only |
| Rolloff    | Frequency below which 85% of energy lies         | v1, v2, v3 |
| Bandwidth  | Spread of the spectrum                           | v1, v2, v3 |
| Kurtosis   | "Peakiness" — detects impulsive fault bursts     | v2, v3 |
| Spectral flux | Frame-to-frame spectral change               | v3 only |

Statistical features are **StandardScaler-normalized** (fit on training set only, saved as `stat_scaler_2b.pkl`).

---

## Phase 1 — Mel-Spectrogram CNN Baseline

### Architecture: MelCNN

```
Input (B, 1, 128, 84)
  → Conv2d(1→32) + BN + ReLU + MaxPool2d    → (B, 32, 64, 42)
  → Conv2d(32→64) + BN + ReLU + MaxPool2d   → (B, 64, 32, 21)
  → Conv2d(64→128) + BN + ReLU + MaxPool2d  → (B, 128, 16, 10)
  → Conv2d(128→256) + BN + ReLU + AdaptiveAvgPool2d(4,4) → (B, 256, 4, 4)
  → Flatten → FC(4096→256) + ReLU + Dropout(0.5) → FC(256→6)
```

### Phase 1 V1 — Baseline

**Train**: `train_phase1.py` | **Kaggle**: `kaggle_phase1V1.py`

- Simple `1/(count+1)` class weights
- SpecAugment (freq_mask=30, time_mask=15) — **too aggressive**
- CosineAnnealingLR, 20 epochs
- **Result**: Machine1 fine, but Machine2 and Machine3 Normal/Abnormal collapsed. The model predicted almost everything as "Abnormal" for M2/M3.

> **This is the only script that CREATES `split_indices_clean.json`**. All subsequent phases load it.

### Phase 1 V2 — Imbalance & Overfitting Fixes

**Train**: `train_phase1V2.py` | **Kaggle**: `kaggle_phase1V2.py`

Research-backed fixes applied:

| Fix | Technique | Paper | What it does |
|-----|-----------|-------|-------------|
| **ENS class weights** | `w = (1−β^n)/(1−β)`, β=0.9999 | Cui et al., CVPR 2019 | Principled weighting that accounts for sample overlap — much stronger than `1/√n` for minority classes |
| **Label smoothing** | ε = 0.1 in CrossEntropyLoss | Müller et al., NeurIPS 2019 | Prevents overconfidence on easy Machine1 classes, keeps gradients flowing to M2/M3 |
| **SpecAugment calibrated** | freq_mask=27, n_freq=2 | Park et al., 2019 | Reduced from 93% to ~42% max frequency masking — V1 was destroying too much signal for minority classes |
| **Mixup** | α = 0.3 | Zhang et al., ICLR 2018 | Interpolates between samples, forcing smooth decision boundaries between M2/M3 Normal vs Abnormal |
| **Warmup + Cosine LR** | 2-epoch linear warmup | Goyal et al., 2017 | Prevents memorizing Machine1 in early epochs before seeing enough M2/M3 |
| **Early stopping** | patience=4 on val_loss | Prechelt, 1998 | Stops before overfitting — val_loss was diverging after epoch 7 in V1 |

### Phase 1 V3 — Machine3 Collapse Fix

**Train**: `train_phase1V3.py` | **Kaggle**: `kaggle_phase1V3.py`

**Problem**: V2 fixed M2 but Machine3_Normal still had near-zero recall (the model predicted M3 Abnormal for almost everything).

**Root cause**: A flat 6-class softmax head can satisfy the loss by learning machine identity but ignoring the Normal/Abnormal distinction within M3.

| Fix | Technique | What it does |
|-----|-----------|-------------|
| **Hierarchical dual-head** | MelCNNHier: head_main (6-class) + head_machine (3-class) + head_fault (binary) | Forces the backbone to learn *both* machine-discriminative AND fault-discriminative features |
| **Focal loss on fault head** | γ = 2.0 | Down-weights easy samples so hard M3 Normal/Abnormal pairs get larger gradients |
| **ReduceLROnPlateau** | Replaces cosine annealing | Holds peak LR while M3 is still learning; reduces only when val_loss stagnates |
| **Mixup α reduced** | 0.3 → 0.1 | Aggressive mixing created ambiguous M3 samples; 0.1 keeps λ > 0.85 in 90% of samples |
| **Gradient clipping** | max_norm=1.0 | Stabilizes hierarchical training |

**Loss function**: `L = L_main + 0.4 * L_machine + 0.6 * L_fault`

### Phase 1 V4 — Kaggle variant of V3

**Kaggle**: `kaggle_phase1V4.py`

Same as V3 but inlined for Kaggle with minor tuning adjustments.

---

## Phase 2b — Mel + Statistical Features

A lightweight alternative to Phase 2 (which used Mel + MFCC dual-stream). Instead of the expensive MFCC CNN stream, Phase 2b adds a tiny FC branch for statistical features — same accuracy boost at nearly the same inference speed as Phase 1.

### Architecture: MelStatCNN / MelStatCNNHier

```
Mel Stream (from Phase 1):  (B, 1, 128, 84) → MelCNN.extract_features() → (B, 256)
Stat Branch:                (B, 5) → FC(5→64) → ReLU → FC(64→32) → ReLU → (B, 32)
Concat:                     (B, 288) → FC(288→256) → Dropout(0.4) → FC(256→6)
```

### Phase 2b V1 — Flat head, global norm

**Train**: `train_phase2b.py` | **Kaggle**: `kaggle_phase2b.py`

- Loads Phase 1 V1 checkpoint into `mel_stream`
- **Differential learning rates**: mel_stream at 1e-4 (preserve learned features), stat branch + head at 5e-4
- Features: `[rms, zcr, rolloff, bandwidth, kurtosis]` — centroid dropped (redundant with mel CNN)
- Global StandardScaler normalization
- ENS class weights + label smoothing

### Phase 2b V2 — Hierarchical heads

**Train**: `train_phase2bV2.py` | **Kaggle**: `kaggle_phase2b_v2.py`

- Loads Phase 1 **V3** checkpoint (MelCNNHier backbone)
- Uses `MelStatCNNHier` with 3 output heads (main, machine, fault)
- Same hierarchical loss as Phase 1 V3
- Focal loss on fault head, Mixup α=0.1, gradient clipping

---

## Ablation Studies

### Phase 1 Ablation

**Kaggle**: `kaggle_phase1_ablation.py` | **Local**: `train_phase1_ablation.py`

Incrementally adds improvements to measure each one's contribution:

| Config | Name | Additions |
|--------|------|-----------|
| A | baseline | Flat head, CE with uniform weights, cosine LR |
| B | +ens+ls | + ENS class weights + label smoothing + warmup |
| C | +mixup | + Mixup α=0.1 |
| D | +hier+focal | + Hierarchical heads + focal loss on fault head |
| E | +rlrop | + ReduceLROnPlateau replacing cosine |

### Phase 2b Ablation

**Kaggle**: `kaggle_phase2b_ablation.py` | **Local**: `train_phase2b_ablation.py`

Tests which statistical features matter:

| Config | Name | Features |
|--------|------|----------|
| A | mel_only | No stat branch — Phase 1 V2 fine-tuned (baseline) |
| B | basic_4 | rms, zcr, rolloff, bandwidth |
| C | +kurtosis | basic_4 + kurtosis |
| D | +flux | basic_4 + spectral_flux |
| E | all_6 | rms, zcr, rolloff, bandwidth, kurtosis, spectral_flux |

---

## Diagnostic Notebooks

### `kaggle_diagnose_leakage.py`

Run **before** training to check for:
- **Temporal leakage**: adjacent sequential clips landing in different splits
- **Near-duplicates**: cosine similarity between train/test neighbors
- **Path audit**: verify file structure

### `kaggle_diagnose_overfit_bias.py`

Run **after** Phase 2b training to check:
- Generalization gap (train vs test accuracy)
- Confidence calibration (overconfident predictions = overfitting)
- Noise robustness (genuine learning degrades gracefully)
- Per-class bias analysis

---

## Inference

**File**: `infer.py`

Standalone inference script — reads `.wav` files, predicts labels 0–5, writes `results.txt` and `time.txt`.

```bash
python infer.py <path_to_data_directory>
```

Requires `phase3_best.pth` and `stat_scaler.pkl` in the same directory.

---

## How to Run on Kaggle (Step by Step)

### Step 1: Phase 1 (creates the split + trains mel-spectrogram CNN)

1. Create a new Kaggle notebook with GPU enabled
2. Add the Machine Fault Dataset as input
3. Paste `kaggle_phase1V1.py` (or V2/V3/V4 depending on which version)
4. Set `ROOT_DIR` to your Kaggle input path, e.g.:
   ```python
   ROOT_DIR = "/kaggle/input/machine-fault-dataset/Students"
   ```
5. Run the notebook
6. **Download outputs**: `phase1_best.pth` (or `phase1_v2_best.pth` / `phase1_v3_best.pth`) + `split_indices_clean.json`
7. Upload these as a new Kaggle dataset for Phase 2b to consume

### Step 2: Phase 2b (adds statistical features)

1. Create a new Kaggle notebook with GPU
2. Add the Machine Fault Dataset as input
3. Add the Phase 1 outputs dataset as a second input
4. Paste `kaggle_phase2b.py` (or `kaggle_phase2b_v2.py`)
5. Update paths:
   ```python
   ROOT_DIR    = "/kaggle/input/machine-fault-dataset/Students"
   PHASE1_CKPT = "/kaggle/input/<your-phase1-dataset>/phase1_v3_best.pth"
   SPLIT_FILE  = "/kaggle/input/<your-phase1-dataset>/split_indices_clean.json"
   ```
6. Run
7. Download: `phase2b_best.pth` + `stat_scaler_2b.pkl`

### Step 3: Ablation Studies (optional)

Same pattern — paste `kaggle_phase1_ablation.py` or `kaggle_phase2b_ablation.py`, point at the right input datasets, run.

---

## Class Imbalance: Why SMOTE Was Not Used

The class imbalance (especially M2/M3 where Normal vs Abnormal is hard to distinguish) was addressed via **three complementary techniques** instead of SMOTE:

| Technique | Mechanism | Paper |
|-----------|-----------|-------|
| **ENS class weights** | Weights loss inversely proportional to "effective number of samples" — accounts for diminishing information per additional sample | Cui et al., CVPR 2019 |
| **Label smoothing** (ε=0.1) | Prevents model from becoming 100% confident on easy classes, preserving gradient signal for hard minority classes | Müller et al., NeurIPS 2019 |
| **Mixup augmentation** | Creates virtual training samples by interpolating between real samples, which implicitly augments minority classes and smooths decision boundaries | Zhang et al., ICLR 2018 |

**Why not SMOTE?** These approaches were found to be more effective for audio spectrogram data in the research literature. SMOTE generates synthetic samples by interpolating in feature space, which can create unrealistic spectrograms. ENS weights + label smoothing + Mixup address imbalance at the loss/training level without fabricating potentially noisy synthetic data.

---

## File Reference Table

| File | Purpose | Run Where | Creates |
|------|---------|-----------|---------|
| `train_phase1.py` | Phase 1 V1 — baseline | Local | `split_indices_clean.json`, `phase1_best.pth` |
| `train_phase1V2.py` | Phase 1 V2 — ENS+LS+Mixup | Local | `phase1_v2_best.pth` |
| `train_phase1V3.py` | Phase 1 V3 — Hier+Focal | Local | `phase1_v3_best.pth` |
| `train_phase2b.py` | Phase 2b V1 — Mel+Stat | Local | `phase2b_best.pth`, `stat_scaler_2b.pkl` |
| `train_phase2bV2.py` | Phase 2b V2 — Hier Mel+Stat | Local | `phase2b_v2_best.pth`, `stat_scaler_2b.pkl` |
| `kaggle_phase1V1.py` | Phase 1 V1 (self-contained) | Kaggle | Same as above |
| `kaggle_phase1V2.py` | Phase 1 V2 (self-contained) | Kaggle | Same as above |
| `kaggle_phase1V3.py` | Phase 1 V3 (self-contained) | Kaggle | Same as above |
| `kaggle_phase1V4.py` | Phase 1 V4 (self-contained) | Kaggle | Same as above |
| `kaggle_phase2b.py` | Phase 2b V1 (self-contained) | Kaggle | Same as above |
| `kaggle_phase2b_v2.py` | Phase 2b V2 (self-contained) | Kaggle | Same as above |
| `kaggle_phase1_ablation.py` | Phase 1 ablation study | Kaggle | Comparison table |
| `kaggle_phase2b_ablation.py` | Phase 2b feature ablation | Kaggle | Comparison table |
| `kaggle_diagnose_leakage.py` | Leakage diagnostic | Kaggle | Report (run before training) |
| `kaggle_diagnose_overfit_bias.py` | Overfit diagnostic | Kaggle | Report (run after training) |
| `infer.py` | Production inference | Local | `results.txt`, `time.txt` |
| `how_to_use.py` | Preprocessing demo | Local | Demo wav files |

---

## Training Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1 V1 (baseline)                                          │
│  ├── Creates split_indices_clean.json (used by ALL phases)      │
│  ├── Precomputes mel-spectrograms (cached as .npy)              │
│  ├── Trains MelCNN (4-block CNN, 6-class softmax)               │
│  └── Saves phase1_best.pth                                      │
│       ↓ Problem: M2/M3 Normal/Abnormal collapse                 │
│                                                                  │
│  Phase 1 V2 (imbalance fixes)                                   │
│  ├── +ENS weights, +Label smoothing, +Mixup, +Warmup            │
│  ├── +Calibrated SpecAugment, +Early stopping                   │
│  └── Saves phase1_v2_best.pth                                   │
│       ↓ Problem: M3_Normal still near-zero recall                │
│                                                                  │
│  Phase 1 V3 (hierarchical fix)                                  │
│  ├── MelCNNHier: 3 output heads (main + machine + fault)        │
│  ├── +Focal loss on fault head, +RLROP, +Grad clipping          │
│  └── Saves phase1_v3_best.pth                                   │
│       ↓                                                          │
│  Phase 2b (add statistical features)                             │
│  ├── Loads Phase 1 checkpoint into mel_stream                    │
│  ├── Adds stat branch (RMS, ZCR, rolloff, bandwidth, kurtosis)  │
│  ├── Differential LRs: mel_stream=1e-4, stat+head=5e-4          │
│  └── Saves phase2b_best.pth + stat_scaler_2b.pkl                │
└─────────────────────────────────────────────────────────────────┘
```
