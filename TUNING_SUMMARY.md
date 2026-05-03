# GearGossips — Machine Fault Detection: Full Tuning Summary

**Project**: Multi-class audio fault detection across 3 machines (Normal / Abnormal per machine = 6 classes)  
**Notebook**: `firsttrial.ipynb` → iteratively fixed → new `phase2b_v3.ipynb`  
**Dataset**: Chronological train/val/test split (Train=39365, Val=8435, Test=8436)  
**Classes**: Machine1_Normal, Machine1_Abnormal, Machine2_Normal, Machine2_Abnormal, Machine3_Normal, Machine3_Abnormal

---

## The Starting Problem

The original `firsttrial.ipynb` used a **MelStatCNN** model (mel spectrogram CNN + statistical features branch).
Machine 2 (M2) was being predicted at ~50/50 accuracy despite an oversampling attempt already in place.
The root causes turned out to be multiple stacked bugs — not just a data imbalance problem.

---

## Architecture Overview

### Phase 1 Backbone (`phase1_best.pth`)
- V3 hierarchical CNN: `block1–4 → fc1(256) → head_main(6) + head_machine(3) + head_fault(1)`
- Loaded into `MelStatCNN.mel_stream` with key remapping and `strict=False`
- New attention layers (in V3 notebook) initialize randomly and are fine-tuned

### MelStatCNN (V2 — firsttrial.ipynb)
- `mel_stream`: MelCNN (256-d embedding via AdaptiveAvgPool2d)
- `stat_branch`: 6 features → 64 → 32
- Fusion: 256 + 32 = 288-d → fc1(256) → heads

### MelStatCNN (V3 — phase2b_v3.ipynb, new)
- `mel_stream`: MelCNN with **AttentionPool2d** replacing AdaptiveAvgPool2d in block4
- `stat_branch`: **19 features** → 128 → 64 (wider, dropout=0.3)
- Fusion: 256 + 64 = **320-d** → fc1(256) → heads
- New features: 6 spectral stats + 13 MFCC means = 19-dim stat vector
- Cache directory: `feats_stat_v4` (new; `feats_mel` reused)

---

## Bug Fixes (Round by Round)

### Round 1 — Initial Diagnosis: 4 bugs found

| # | Location | Bug | Fix |
|---|----------|-----|-----|
| 1 | Cell 17 | `requires_grad=False` was set to `True` — backbone was NOT frozen | Changed to `False` |
| 2 | Cell 17 | `mel_stream` missing from optimizer — backbone never updated even when unfrozen | Added `mel_stream.parameters()` as separate param group |
| 3 | Cell 15 | `WeightedRandomSampler` only boosted M2 samples (3×) — too narrow, ignored other imbalances | Replaced with full inverse-frequency sampler |
| 4 | Cell 17 | No focal loss on main classification head | Added `FocalCrossEntropy` class and used it for `crit_main` |
| 5 | Cell 19 | Unfreeze set `requires_grad=False` instead of `True` | Fixed to `True` |

---

### Round 2 — Results Got Worse (Model predicts almost all Abnormal)

**Observed**: All Normal classes had precision=1.0 but very low recall (~33–39%)
**Root cause**: Triple over-compensation stacked simultaneously:
- Full inverse-frequency sampler (strongly boosting Abnormal)
- ENS class weights (also boosting Abnormal)
- FocalCrossEntropy (further downweighting easy Normal gradients)

**Fix (Cell 15)**: Removed `WeightedRandomSampler` entirely — ENS weights alone are enough.  
Changed to `shuffle=True` (no sampler).

---

### Round 3 — Overfitting: Best epoch was Epoch 1

**Training output**:
```
Epoch 1: val_acc=0.6988  <- saved
Epoch 2: val_acc=0.5695  (patience 1/15)
...
Epoch 16: [early stop]
Best val loss: 0.1278  (val acc: 0.6988)
```
**Root cause 1**: `MIXUP_ALPHA=0.0` — mixup was disabled, training overfit fast  
**Root cause 2**: `LR_NEW_LAYERS=1e-3` — too high, unstable training

**Fix (Cell 2)**:
- `MIXUP_ALPHA: 0.0 → 0.3`
- `LR_NEW_LAYERS: 1e-3 → 3e-4`

---

### Round 4 — Poor Accuracy (~57%): Contradictory signals between branches

**Observed**:
```
Accuracy: 0.5671 | Macro F1: 0.5504
Machine1_Normal:    precision=1.00, recall=0.39
Machine1_Abnormal:  precision=0.24, recall=1.00
```
**Root cause**: Mixup was applied to mel only — stat features were not mixed.  
Two branches received contradictory training signals for the same sample.

**Fix (Cell 12)**: `train_epoch()` now applies mixup to both mel AND stat using the **same** `lam` and `perm`:
```python
mel_mix  = lam * mel  + (1-lam) * mel[perm]
stat_mix = lam * stat + (1-lam) * stat[perm]
```

---

### Round 5 — Results Regressed Again (Macro F1=0.55)

**Observed**: Same Abnormal-bias pattern returned after adding consistent mixup  
**Root cause**: `FocalCrossEntropy` on `crit_main` downweights gradients from easy Normal samples, causing the model to focus excessively on Abnormal classes.  
Also: early stopping was by `val_loss` — best val_loss was at epoch 1 (low loss, bad accuracy), while epoch 5 had better val_acc (0.819 vs 0.795).

**Fix (Cell 17)**: Reverted `crit_main` to plain `CrossEntropyLoss` with ENS weights  
**Fix (Cell 19)**: Changed early stopping criterion from `val_loss` to **`val_acc`**

---

### Round 6 — Post-Unfreeze Degradation (M2 still bad)

**Training output**:
```
Epoch 5:  val_acc=0.8190  <- saved
[unfreeze] Mel backbone unfrozen with low LR
Epoch 6:  val_acc=0.7898  (patience 5)
Epoch 7:  val_acc=0.7401  (patience 6)
...
```
**Root cause**: `LR_MEL_STREAM=1e-5` was too high — unfreezing the backbone destabilized the already-trained heads.  
Also: unfreeze at epoch 6 was too early (heads not yet stable).

**Fix (Cell 2)**:
- `LR_MEL_STREAM: 1e-5 → 1e-6`
- `UNFREEZE_EPOCH: 6 → 10`

---

## Results Progression

| Round | Key Change | Accuracy | Macro F1 | M2_Norm F1 | M2_Abn F1 | Notes |
|-------|-----------|----------|----------|------------|-----------|-------|
| Baseline | Original notebook | ~50% | ~0.33 | ~0.33 | ~0.33 | M2 at chance |
| Round 1 | Fix freeze/unfreeze + sampler + focal | ~57% | ~0.55 | ~0.49 | ~0.36 | Abnormal bias |
| Round 2 | Remove sampler (sqrt-balanced) | ~57% | ~0.55 | ~0.49 | ~0.36 | Same bias |
| Round 3 | Mixup=0.3, LR_NEW=3e-4 | ~69% | ~0.65 | ~0.62 | ~0.32 | Improving |
| Round 4 | Consistent mixup (mel+stat same perm) | ~72% | ~0.68 | ~0.62 | ~0.20 | M2_Abn worse |
| Round 5 | Plain CE loss + early stop by val_acc | ~78% | ~0.75 | ~0.60 | ~0.27 | Big jump |
| Round 6 | LR_MEL=1e-6, unfreeze@10 | **~78%** | **~0.75** | **~0.60** | **~0.27–0.28** | Stable |

### Best Result (firsttrial.ipynb, V2 architecture)
```
-- Test Results --
Accuracy : 0.7792
Macro F1 : 0.7498

Per-class F1:
  Machine1_Normal:    0.9344
  Machine1_Abnormal:  0.7614
  Machine2_Normal:    0.5969
  Machine2_Abnormal:  0.2815   <-- still the problem
  Machine3_Normal:    0.9865
  Machine3_Abnormal:  0.9384

Precision / Recall:
  Machine1_Normal:    precision=1.00, recall=0.88
  Machine1_Abnormal:  precision=0.61, recall=1.00
  Machine2_Normal:    precision=0.85, recall=0.46
  Machine2_Abnormal:  precision=0.18, recall=0.61
  Machine3_Normal:    precision=0.99, recall=0.98
  Machine3_Abnormal:  precision=0.90, recall=0.98

Per-sample inference: ~0.40 ms
```

---

## Final Configuration (firsttrial.ipynb Cell 2)

```python
SR = 16000; DURATION_SEC = 4.0; BATCH_SIZE = 32; TRAIN_EPOCHS = 40; NUM_WORKERS = 2
LABEL_SMOOTH = 0.05; ENS_BETA = 0.9999; MIXUP_ALPHA = 0.3
FOCAL_GAMMA  = 1.0    # BinaryFocalLoss on fault head only
HIER_ALPHA   = 0.2    # weight of machine-ID auxiliary loss
WEIGHT_DECAY = 5e-4
LR_MEL_STREAM = 1e-6  # very low -- gentle backbone fine-tuning
LR_NEW_LAYERS = 3e-4  # stat_branch, fusion fc1, all heads
WARMUP_EPOCHS = 0; RLROP_PATIENCE = 4; RLROP_FACTOR = 0.5
ES_PATIENCE   = 15; UNFREEZE_EPOCH = 10
```

---

## Key Principles Learned

1. **Don't stack imbalance fixes** — sampler + ENS weights + FocalCE all fighting imbalance simultaneously causes the model to collapse to always-Abnormal. Use ENS weights in loss only.

2. **Mixup must be consistent across all branches** — mel and stat must use the same `lam` and `perm`. Inconsistent mixup sends contradictory signals between branches.

3. **Early stopping by val_acc, not val_loss** — with CrossEntropyLoss, best val_loss (epoch 1) and best val_acc can diverge significantly. Always stop by accuracy.

4. **Backbone LR must be very small after unfreeze** — `LR_MEL_STREAM=1e-6` prevents the fine-tuned heads from being destabilized when the backbone is unfrozen. RLROP also reduces this over time.

5. **Unfreeze timing matters** — unfreezing too early (epoch 6) before heads are stable causes catastrophic forgetting. Wait until epoch ~10.

6. **M2 is genuinely harder** — M2 Normal/Abnormal sounds are more acoustically similar than M1/M3. It's not purely a class imbalance issue. Better features (MFCCs) and spatial attention are needed.

---

## New Architecture: phase2b_v3.ipynb

Created to address M2's fundamental difficulty with two structural improvements:

### AttentionPool2d (replaces AdaptiveAvgPool2d)
```python
class AttentionPool2d(nn.Module):
    def __init__(self, in_channels, out_size=(4,4)):
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, 1), nn.ReLU(),
            nn.Conv2d(in_channels//8, 1, 1))
        self.pool = nn.AdaptiveAvgPool2d(out_size)
    def forward(self, x):
        w = torch.softmax(self.attn(x).flatten(2), dim=-1)
        return self.pool(x * w.view(x.shape[0], 1, x.shape[2], x.shape[3]))
```
Learns which time-frequency regions are most discriminative instead of averaging everything.

### MFCC Features (13 means added to stat branch)
```python
STAT_FEATURES = ["rms", "zcr", "rolloff", "bandwidth", "spectral_flux", "kurtosis",
                  "mfcc_1", ..., "mfcc_13"]
STAT_DIM = 19  # was 6
```
MFCCs capture timbral texture — critical for distinguishing M2 Normal vs Abnormal.

### V3 Notebook Structure (11 cells, down from 25)
| Cell | Contents |
|------|----------|
| 0 | Markdown: description + setup |
| 1 | Imports + `wandb.login()` |
| 2 | **Config** — edit 4 paths + all hyperparams |
| 3 | Data pipeline (label maps, split, preprocessor, mel+MFCC features, dataset, scalers) |
| 4 | Model + training code (AttentionPool2d, MelStatCNN, losses, train/eval) |
| 5 | Scan files + precompute features + fit scalers |
| 6 | Class weights + DataLoaders |
| 7 | Model init + optimizer + loss functions |
| 8 | Training loop (wandb logging, early stop by val_acc) |
| 9 | Evaluation + confusion matrix + learning curves |
| 10 | Save checkpoint + scaler + zip outputs |

---

## Next Steps

1. **Run `phase2b_v3.ipynb` on Kaggle** to get the attention+MFCC baseline  
   - Check M2_Abnormal F1 vs current 0.28
   - Ensure `LR_NEW_LAYERS=3e-4` (not 1e-6) in Cell 2 before running

2. **W&B Sweep** (after V3 baseline is established)  
   Sweep config:
   ```yaml
   method: bayes
   metric:
     name: val_acc
     goal: maximize
   parameters:
     LR_NEW_LAYERS:  {values: [1e-4, 3e-4, 5e-4]}
     MIXUP_ALPHA:    {values: [0.2, 0.3, 0.4]}
     HIER_ALPHA:     {values: [0.1, 0.2, 0.3]}
     FOCAL_GAMMA:    {values: [0.5, 1.0, 2.0]}
     ENS_BETA:       {values: [0.999, 0.9999, 0.99999]}
   ```

3. **If M2 still below F1=0.40 after V3 + sweep**, consider:
   - Per-machine classification heads (separate fc+head per machine)
   - GRU over temporal windows (captures temporal fault patterns)
   - Longer audio clips for M2 (DURATION_SEC per machine)

---

## Files

| File | Description |
|------|-------------|
| `firsttrial.ipynb` | Original notebook with all V2 bug fixes applied |
| `phase2b_v3.ipynb` | New compact notebook (11 cells) with attention pooling + MFCCs |
| `phase1_best.pth` | Phase 1 V3 pretrained checkpoint (hierarchical heads) |
| `split_indices_clean.json` | Chronological train/val/test split (must be consistent) |
| `requirements.txt` | Python dependencies |
