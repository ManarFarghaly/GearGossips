# GearGossips — Machine Fault Detection via Audio

**Dataset**: [Machine Fault Dataset on Kaggle](https://www.kaggle.com/datasets/mostafaehab41/machine-fault-dataset)

A multi-phase deep learning pipeline that classifies industrial machine audio recordings into **6 classes** — Normal vs Abnormal for each of 3 machines — using mel-spectrograms, statistical features, and CNN architectures trained with hierarchical losses.

---

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [Dataset Structure](#dataset-structure)
3. [Project Layout](#project-layout)
4. [Environment Setup](#environment-setup)
5. [Running Inference](#running-inference)
6. [Training Pipeline](#training-pipeline)
7. [Phase 1 — Mel-Spectrogram CNN](#phase-1--mel-spectrogram-cnn)
8. [Phase 2b — Mel + Statistical Features](#phase-2b--mel--statistical-features)
9. [Ablation Studies](#ablation-studies)
10. [Results Summary](#results-summary)
11. [File Reference Table](#file-reference-table)

---

## Problem Overview

Three industrial machines are monitored via audio. Each recording is a short `.wav` clip. The task is to classify each clip into one of **6 classes**:

| Label | Class               |
|-------|---------------------|
| 0     | Machine 1 — Normal  |
| 1     | Machine 1 — Abnormal |
| 2     | Machine 2 — Normal  |
| 3     | Machine 2 — Abnormal |
| 4     | Machine 3 — Normal  |
| 5     | Machine 3 — Abnormal |

The dataset is class-imbalanced (Machine 1 Normal: ~2430 samples; Machine 3 Abnormal: ~453). Early models collapsed — predicting nearly everything as "Abnormal" for M2/M3 while getting M1 right. Each training version addresses a specific observed failure.

---

## Dataset Structure

```
Students/                        ← ROOT_DIR
├── machine1/
│   ├── Normal/   *.wav
│   └── Abnormal/ *.wav
├── machine2/
│   ├── Normal/   *.wav
│   └── Abnormal/ *.wav
└── machine3/
    ├── Normal/   *.wav
    └── Abnormal/ *.wav
```

---

## Project Layout

```
GearGossips/
│
├── machine_listener/                     # Core library — shared across all phases
│   └── src/
│       ├── preprocess.py                 # Audio preprocessing (resample, trim, normalize, augment)
│       ├── dataset.py                    # PyTorch Dataset + label map
│       ├── split_utils.py                # Chronological, leak-free train/val/test split
│       ├── train_utils.py                # Shared train/eval loops, metrics, plotting
│       ├── features/
│       │   ├── mel_spectrogram.py        # Mel-spectrogram → (1, 128, 84)
│       │   ├── MFCC.py                   # MFCC + deltas → (3, 40, 84)
│       │   └── statistical.py            # Scalar feature extractors (v1–v4)
│       └── models/
│           ├── cnn_baseline.py           # MelCNN (Phase 1 V1/V2) + MelCNNHier (V3/V4)
│           ├── cnn_MFCC.py               # MFCC CNN sub-network
│           ├── cnn_MelMFCC.py            # Dual-stream Mel+MFCC (Phase 2)
│           ├── cnn_mel_stat.py           # MelStatCNN + MelStatCNNHier (Phase 2b V1/V2)
│           ├── cnn_mel_stat_v4.py        # MelStatCNNV4 with AttentionPool2d (Phase 2b V3/V4)
│           └── cnn_statistical.py        # Mel+MFCC+Stat architecture (Phase 3)
│
├── train_phase1.py                       # Phase 1 V1 — creates split, trains baseline CNN
├── train_phase1V2.py                     # Phase 1 V2 — ENS weights, Mixup, label smoothing
├── train_phase1V3.py                     # Phase 1 V3 — hierarchical heads, focal loss
├── train_phase1V4.py                     # Phase 1 V4 — V3 with RLROP + tuned HPs
├── train_phase1_ablation.py              # Phase 1 ablation study
├── train_phase2b.py                      # Phase 2b V1 — flat head, global stat norm
├── train_phase2bV2.py                    # Phase 2b V2 — hierarchical heads
├── train_phase2bV3.py                    # Phase 2b V3 — 19 features, attention pooling
├── train_phase2bV4.py                    # Phase 2b V4 — best model (lower LR, early unfreeze)
├── train_phase2b_ablation.py             # Phase 2b feature ablation
│
├── kaggle_phase1V1.py                    # Self-contained Kaggle scripts
├── kaggle_phase1V2.py                    #   (no machine_listener imports — paste into
├── kaggle_phase1V3.py                    #    a Kaggle notebook cell and run)
├── kaggle_phase1V4.py                    #
├── kaggle_phase1_ablation.py             #
├── kaggle_phase2b.py                     #
├── kaggle_phase2b_v2.py                  #
├── kaggle_phase2b_v3.py                  #
├── kaggle_v2b_v4.py                      # Phase 2b V4 — best model (Kaggle version)
├── kaggle_phase2b_ablation.py            #
├── kaggle_diagnose_leakage.py            # Data leakage diagnostic
├── kaggle_diagnose_overfit_bias.py       # Overfitting / bias diagnostic
│
├── infer.py                              # Inference script — reads wav folder, writes results
├── Makefile                              # Build targets: infer, train-best, install, clean
├── requirements.txt                      # Python dependencies
└── NewResults/                           # Screenshots and result images per version
```

---

## Environment Setup

```bash
pip install -r requirements.txt
```

For GPU training (recommended), install PyTorch with CUDA support from [pytorch.org](https://pytorch.org) before running the above.

### Kaggle

The `kaggle_*.py` scripts are self-contained — they inline all preprocessing and model code. No library upload is needed. Create a Kaggle notebook, paste the script contents, set `ROOT_DIR` to your input path, and run with GPU enabled.

---

## Running Inference

The best model is **Phase 2b V4** (`MelStatCNNV4` — mel CNN + 19 statistical features with attention pooling).

Place the checkpoint `phase2b_v4_best.pth` in the same directory as `infer.py`, then:

```bash
python infer.py <path_to_wav_directory>
```

Or using Make:

```bash
make infer DATA_DIR=/path/to/wav/directory
```

**Input**: A directory of `.wav` files named numerically (`1.wav`, `2.wav`, ...).

**Output** (written to current directory):
- `results.txt` — one predicted label (0–5) per line, one line per input file
- `time.txt` — processing time in seconds per file (3 decimal places), file I/O excluded

**Timing**: The timer starts after loading the wav file and stops after the model prediction — it measures preprocessing + feature extraction + inference, not disk I/O.

---

## Training Pipeline

Phase 2 training requires a Phase 1 checkpoint. Run in order:

```bash
# 1. Train Phase 1 (creates split file + mel backbone)
make train-phase1

# 2. Train best model (Phase 2b V4) — uses phase1_best.pth
make train-phase2b-v4

# Or: train both end-to-end
make train-best
```

**Checkpoints are saved to**: `machine_listener/outputs/saved_models/`

**Cached features**: `machine_listener/outputs/features/`

---

## Phase 1 — Mel-Spectrogram CNN

**Architecture** — 4-block CNN:

```
Input (B, 1, 128, 84)
→ Conv2d(1→32)   + BN + ReLU + MaxPool2d
→ Conv2d(32→64)  + BN + ReLU + MaxPool2d
→ Conv2d(64→128) + BN + ReLU + MaxPool2d
→ Conv2d(128→256)+ BN + ReLU + Pool
→ FC(4096→256) + Dropout(0.5) → 6-class head
```

Phase 1 V3/V4 add a machine-ID head (3-class) and fault-status head (binary) alongside the main head, with loss `L = L_main + 0.4·L_machine + 0.6·L_fault_focal`.

### Phase 1 V1 — Baseline

**Files**: `train_phase1.py` / `kaggle_phase1V1.py`

Simple 1/(count+1) class weights, SpecAugment (too aggressive at freq_mask=30), CosineAnnealingLR.

> This script **creates** `split_indices_clean.json`. All later phases load this same split.

**Confusion Matrix:**
<!-- TODO: insert Phase1V1_confusion_matrix.png -->

| Metric    | Value |
|-----------|-------|
| Accuracy  | —     |
| Macro F1  | —     |
| Infer (ms/sample) | — |

### Phase 1 V2 — Imbalance Fixes

**Files**: `train_phase1V2.py` / `kaggle_phase1V2.py`

Added ENS class weights (Cui et al., CVPR 2019), label smoothing ε=0.1, calibrated SpecAugment (freq_mask=27), Mixup α=0.3, warmup LR, early stopping.

**Confusion Matrix:**
<!-- TODO: insert Phase1V2_confusion_matrix.png -->

| Metric    | Value |
|-----------|-------|
| Accuracy  | —     |
| Macro F1  | —     |
| Infer (ms/sample) | — |

### Phase 1 V3 — Machine3 Fix

**Files**: `train_phase1V3.py` / `kaggle_phase1V3.py`

Added hierarchical dual-head (machine + fault), focal loss on fault head γ=2.0, ReduceLROnPlateau, Mixup α reduced to 0.1, gradient clipping.

**Confusion Matrix:**
<!-- TODO: insert Phase1V3_confusion_matrix.png -->

| Metric    | Value |
|-----------|-------|
| Accuracy  | —     |
| Macro F1  | —     |
| Infer (ms/sample) | — |

### Phase 1 V4

**Files**: `train_phase1V4.py` / `kaggle_phase1V4.py`

V3 architecture with ReduceLROnPlateau replacing cosine annealing, ES patience=4, Mixup α=0.1.

**Confusion Matrix:**
<!-- TODO: insert Phase1V4_confusion_matrix.png -->

| Metric    | Value |
|-----------|-------|
| Accuracy  | —     |
| Macro F1  | —     |
| Infer (ms/sample) | — |

---

## Phase 2b — Mel + Statistical Features

The mel backbone from Phase 1 is frozen and a statistical feature branch is fused at the embedding level.

### Feature Sets

| Version | Stat features | Dim |
|---------|--------------|-----|
| V1      | rms, zcr, rolloff, bandwidth, kurtosis | 5 |
| V2      | same as V1 | 5 |
| V3/V4   | rms, zcr, rolloff, bandwidth, spectral_flux, kurtosis, mfcc_1..13 | 19 |

### Phase 2b V1

**Files**: `train_phase2b.py` / `kaggle_phase2b.py`

Loads Phase 1 V1 checkpoint. 5 stat features, global StandardScaler, flat 6-class head.

**Confusion Matrix:**
<!-- TODO: insert Phase2bV1_confusion_matrix.png -->

| Metric    | Value |
|-----------|-------|
| Accuracy  | —     |
| Macro F1  | —     |
| Infer (ms/sample) | — |

### Phase 2b V2

**Files**: `train_phase2bV2.py` / `kaggle_phase2b_v2.py`

Loads Phase 1 V3 checkpoint. Hierarchical heads (main + machine + fault), focal loss, Mixup α=0.1.

**Confusion Matrix:**
<!-- TODO: insert Phase2bV2_confusion_matrix.png -->

| Metric    | Value |
|-----------|-------|
| Accuracy  | —     |
| Macro F1  | —     |
| Infer (ms/sample) | — |

### Phase 2b V3

**Files**: `train_phase2bV3.py` / `kaggle_phase2b_v3.py`

Loads Phase 1 best checkpoint. 19 stat features (adds spectral_flux + mfcc_1..13), AttentionPool2d in CNN block4, per-machine scalers (3 separate normalisation scalers), stat branch Dropout=0.3.

Key HPs: LR_new=3e-4, unfreeze at epoch 7, Mixup α=0.3.

**Confusion Matrix:**
<!-- TODO: insert Phase2bV3_confusion_matrix.png -->

| Metric    | Value |
|-----------|-------|
| Accuracy  | —     |
| Macro F1  | —     |
| Infer (ms/sample) | — |

### Phase 2b V4 — Best Model

**Files**: `train_phase2bV4.py` / `kaggle_v2b_v4.py`

Same architecture as V3 but with lower LR for new layers (5e-5 vs 3e-4), earlier backbone unfreeze (epoch 2 vs 7), and heavier dropout in stat branch (0.5 vs 0.3).

**Architecture**:

```
Mel Stream: (B, 1, 128, 84)
  → Conv2d blocks 1–3 with MaxPool2d
  → Conv2d block4 + AttentionPool2d(4,4)   ← learnable spatial attention
  → FC(4096→256) + Dropout(0.5)            → (B, 256)

Stat Branch: (B, 19)
  → FC(19→128) + BN + ReLU + Dropout(0.5)
  → FC(128→64) + ReLU                      → (B, 64)

Fusion: cat(256, 64) → FC(320→256) + Dropout(0.5)
  → head_main (B, 6), head_machine (B, 3), head_fault (B, 1)
```

**Two-pass inference** (no ground-truth machine label needed at test time):
1. Normalise stat with global averaged scaler → predict machine identity
2. Normalise stat with predicted machine's scaler → predict final class

**Confusion Matrix:**
<!-- TODO: insert Phase2bV4_confusion_matrix.png -->

| Metric    | Value |
|-----------|-------|
| Accuracy  | —     |
| Macro F1  | —     |
| Infer (ms/sample) | — |

---

## Ablation Studies

### Phase 1 Ablation

**Files**: `train_phase1_ablation.py` / `kaggle_phase1_ablation.py`

Adds improvements one at a time to isolate each contribution:

| Config | Additions over baseline |
|--------|------------------------|
| A      | Baseline — flat CE, cosine LR |
| B      | + ENS weights + label smoothing + warmup |
| C      | + Mixup α=0.1 |
| D      | + Hierarchical heads + focal loss |
| E      | + ReduceLROnPlateau |

### Phase 2b Ablation

**Files**: `train_phase2b_ablation.py` / `kaggle_phase2b_ablation.py`

Tests which statistical features matter:

| Config | Features |
|--------|---------|
| A      | Mel only (no stat branch) |
| B      | rms, zcr, rolloff, bandwidth |
| C      | + kurtosis |
| D      | + spectral_flux |
| E      | All 6: rms, zcr, rolloff, bandwidth, kurtosis, spectral_flux |

---

## Results Summary

| Version       | Accuracy | Macro F1 | Infer (ms/sample) |
|---------------|----------|----------|-------------------|
| Phase 1 V1    | —        | —        | —                 |
| Phase 1 V2    | —        | —        | —                 |
| Phase 1 V3    | —        | —        | —                 |
| Phase 1 V4    | —        | —        | —                 |
| Phase 2b V1   | —        | —        | —                 |
| Phase 2b V2   | —        | —        | —                 |
| Phase 2b V3   | —        | —        | —                 |
| Phase 2b V4   | —        | —        | —                 |

---

## File Reference Table

| File | Purpose | Run Where | Requires | Produces |
|------|---------|-----------|----------|----------|
| `train_phase1.py` | Phase 1 V1 — baseline | Local | — | `phase1_best.pth`, `split_indices_clean.json` |
| `train_phase1V2.py` | Phase 1 V2 — ENS+Mixup | Local | split file | `phase1_v2_best.pth` |
| `train_phase1V3.py` | Phase 1 V3 — Hier+Focal | Local | split file | `phase1_v3_best.pth` |
| `train_phase1V4.py` | Phase 1 V4 — RLROP+tuned | Local | split file | `phase1_best.pth` |
| `train_phase2b.py` | Phase 2b V1 — flat head | Local | `phase1_best.pth` | `phase2b_best.pth` |
| `train_phase2bV2.py` | Phase 2b V2 — Hier | Local | `phase1_v3_best.pth` | `phase2b_v2_best.pth` |
| `train_phase2bV3.py` | Phase 2b V3 — 19 features | Local | `phase1_best.pth` | `phase2b_v3_best.pth` |
| `train_phase2bV4.py` | Phase 2b V4 — **best model** | Local | `phase1_best.pth` | `phase2b_v4_best.pth` |
| `kaggle_phase1V1.py` | Phase 1 V1 self-contained | Kaggle | dataset | same as local |
| `kaggle_phase1V2.py` | Phase 1 V2 self-contained | Kaggle | dataset | same as local |
| `kaggle_phase1V3.py` | Phase 1 V3 self-contained | Kaggle | dataset | same as local |
| `kaggle_phase1V4.py` | Phase 1 V4 self-contained | Kaggle | dataset | same as local |
| `kaggle_phase2b.py` | Phase 2b V1 self-contained | Kaggle | phase1 outputs | same as local |
| `kaggle_phase2b_v2.py` | Phase 2b V2 self-contained | Kaggle | phase1 outputs | same as local |
| `kaggle_phase2b_v3.py` | Phase 2b V3 self-contained | Kaggle | phase1 outputs | same as local |
| `kaggle_v2b_v4.py` | Phase 2b V4 self-contained | Kaggle | phase1 outputs | `phase2b_v4_best.pth` |
| `kaggle_phase1_ablation.py` | Phase 1 ablation | Kaggle | dataset | comparison table |
| `kaggle_phase2b_ablation.py` | Phase 2b feature ablation | Kaggle | phase1 outputs | comparison table |
| `kaggle_diagnose_leakage.py` | Leakage check (run before training) | Kaggle | dataset | report |
| `kaggle_diagnose_overfit_bias.py` | Overfitting check (run after training) | Kaggle | checkpoint | report |
| `infer.py` | **Inference — final delivery** | Local | `phase2b_v4_best.pth` | `results.txt`, `time.txt` |
| `Makefile` | Build targets | Local | — | — |
| `requirements.txt` | Python dependencies | — | — | — |
