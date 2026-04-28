"""
Overfitting & Bias Diagnostic — Phase 2b
=========================================
Run this AFTER kaggle_phase2b.py has finished.
Needs:  phase2b_best.pth  +  stat_scaler_2b.pkl  +  feats_mel/  +  feats_stat_v2/
        split_indices.json  (or split_indices_clean.json if you already ran the
        leakage fix)

Seven checks:

  1. GENERALIZATION GAP     — eval on train set with the saved checkpoint.
                              A gap > 1.5 % between train and test = overfit signal.

  2. CONFIDENCE CALIBRATION — are predictions well-calibrated?
                              Overfit models are overconfident: high softmax scores
                              even on wrong predictions.

  3. NOISE ROBUSTNESS       — add Gaussian noise at 5 SNR levels to test mel specs.
                              Genuine learning degrades gracefully; memorization
                              collapses suddenly at low noise.

  4. MC-DROPOUT             — run inference 20 times with dropout ON.
                              High variance across runs on the same sample = model
                              is uncertain despite appearing confident without dropout.

  5. PER-CLASS BIAS         — precision / recall / F1 per class, plus a
                              Normal-vs-Abnormal breakdown.
                              Recall(Abnormal) < Recall(Normal) = bias toward Normal.

  6. CROSS-MACHINE BIAS     — train split contains all three machines.
                              We test on each machine separately to see if accuracy
                              is consistent, or if the model leans on machine identity
                              rather than fault state.

  7. PREDICTION DISTRIBUTION — compare the distribution of predicted labels vs.
                               true labels on the test set.
                               Mismatch = the model systematically over-predicts
                               certain classes.

All results are printed and saved to /kaggle/working/diagnostic_report.txt.
"""

import os, json, pathlib, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── CONFIG ─────────────────────────────────────────────────────────────────────
ROOT_DIR     = "/kaggle/input/datasets/mostafaehab41/machine-fault-dataset"
MODELS_DIR   = "/kaggle/working"
CKPT_PATH    = os.path.join(MODELS_DIR, "phase2b_best.pth")
SCALER_PATH  = os.path.join(MODELS_DIR, "stat_scaler_2b.pkl")
FEATS_MEL    = "/kaggle/working/feats_mel"
FEATS_STAT   = "/kaggle/working/feats_stat_v2"
SPLIT_FILE   = os.path.join(MODELS_DIR, "split_indices.json")   # swap to _clean if desired
BATCH_SIZE   = 64
NUM_WORKERS  = 2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPORT_PATH  = os.path.join(MODELS_DIR, "diagnostic_report.txt")

CLASS_NAMES  = ["M1_Normal","M1_Abnormal","M2_Normal","M2_Abnormal","M3_Normal","M3_Abnormal"]
NORMAL_IDS   = {0, 2, 4}
ABNORMAL_IDS = {1, 3, 5}
MACHINE_IDS  = {0: [0,1], 1: [2,3], 2: [4,5]}   # machine_idx → class ids

# Noise SNR levels (dB) for the robustness test
NOISE_SNRS   = [40, 30, 20, 10, 5]   # 40 dB = barely noticeable, 5 dB = harsh

lines = []   # collected for the report file

def log(msg=""):
    print(msg)
    lines.append(msg)

# ── MODEL (copy of MelStatCNN from phase2b) ────────────────────────────────────
class MelCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2))
        self.block4 = nn.Sequential(nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.fc1 = nn.Linear(256*4*4,256); self.dropout = nn.Dropout(0.5); self.fc2 = nn.Linear(256,6)
    def extract_features(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        return self.dropout(F.relu(self.fc1(torch.flatten(x,1))))
    def forward(self, x): return self.fc2(self.extract_features(x))

class MelStatCNN(nn.Module):
    def __init__(self, num_classes=6, stat_dim=5):
        super().__init__()
        self.mel_stream  = MelCNN(num_classes)
        self.stat_branch = nn.Sequential(nn.Linear(stat_dim,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.fc1 = nn.Linear(288,256); self.dropout = nn.Dropout(0.4); self.fc2 = nn.Linear(256,num_classes)
    def forward(self, mel, stat):
        f = torch.cat([self.mel_stream.extract_features(mel), self.stat_branch(stat)], dim=1)
        return self.fc2(self.dropout(F.relu(self.fc1(f))))

# ── DATASET ────────────────────────────────────────────────────────────────────
def scan_files(root):
    LABEL_MAP = {("machine1","Normal"):0,("machine1","Abnormal"):1,
                 ("machine2","Normal"):2,("machine2","Abnormal"):3,
                 ("machine3","Normal"):4,("machine3","Abnormal"):5}
    paths, labels = [], []
    for f in pathlib.Path(root).rglob("*.wav"):
        lbl = LABEL_MAP.get((f.parent.parent.name, f.parent.name))
        if lbl is not None: paths.append(f); labels.append(lbl)
    return paths, labels

class NpyDataset(Dataset):
    def __init__(self, mel_dir, stat_dir, labels, indices):
        self.mel_dir  = pathlib.Path(mel_dir)
        self.stat_dir = pathlib.Path(stat_dir)
        self.labels   = labels
        self.indices  = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        ri   = self.indices[i]
        mel  = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        stat = torch.tensor(np.load(self.stat_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        return (mel, stat), torch.tensor(self.labels[ri], dtype=torch.long)

def collate(batch):
    feats, labels = zip(*batch)
    return (torch.stack([f[0] for f in feats]),
            torch.stack([f[1] for f in feats])), torch.stack(labels)

def make_loader(indices):
    ds = NpyDataset(FEATS_MEL, FEATS_STAT, ALL_LABELS, indices)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate,
                      num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False,
                      prefetch_factor=2)

# ── EVAL HELPERS ───────────────────────────────────────────────────────────────
def eval_loader(model, loader, sm, ss, mc_dropout=False):
    """Returns (preds, true_labels, max_probs, all_probs)."""
    if mc_dropout:
        model.train()   # keep dropout active
    else:
        model.eval()
    sm_t = torch.tensor(sm, dtype=torch.float32, device=DEVICE)
    ss_t = torch.tensor(ss, dtype=torch.float32, device=DEVICE)
    all_preds, all_true, all_maxp, all_probs = [], [], [], []
    with torch.no_grad():
        for (mel, stat), y in loader:
            mel, stat, y = mel.to(DEVICE), stat.to(DEVICE), y.to(DEVICE)
            stat = (stat - sm_t) / ss_t
            logits = model(mel, stat)
            probs  = F.softmax(logits, dim=1)
            preds  = probs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())
            all_maxp.extend(probs.max(1).values.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return (np.array(all_preds), np.array(all_true),
            np.array(all_maxp), np.array(all_probs))

def accuracy(preds, true): return (preds == true).mean()

# ── NOISE HELPER ───────────────────────────────────────────────────────────────
def add_mel_noise(mel_tensor, snr_db):
    """Add white noise to mel spec at a given SNR (dB)."""
    sig_power  = mel_tensor.pow(2).mean()
    noise      = torch.randn_like(mel_tensor)
    noise_power = noise.pow(2).mean()
    scale = torch.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10) + 1e-12))
    return mel_tensor + scale * noise

# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 68)
log("OVERFITTING & BIAS DIAGNOSTIC  —  Phase 2b")
log("=" * 68)

for path, label in [(CKPT_PATH,"checkpoint"), (SCALER_PATH,"scaler"),
                    (FEATS_MEL,"mel cache"), (FEATS_STAT,"stat cache"),
                    (SPLIT_FILE,"split file")]:
    exists = pathlib.Path(path).exists()
    log(f"  {'✓' if exists else '✗'}  {label}: {path}")
    if not exists:
        raise FileNotFoundError(f"Required file missing: {path}")

log()

# Load everything
scaler     = pickle.load(open(SCALER_PATH, "rb"))
sm, ss     = scaler["mean"], scaler["std"]
splits     = json.load(open(SPLIT_FILE))
ALL_PATHS, ALL_LABELS = scan_files(ROOT_DIR)

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
stat_dim = ckpt.get("stat_dim", 5)
model = MelStatCNN(num_classes=6, stat_dim=stat_dim).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])

log(f"Checkpoint loaded  (epoch {ckpt['epoch']},  saved val_acc={ckpt['val_acc']:.4f})")
log(f"Device: {DEVICE}  |  stat_dim={stat_dim}  |  features={ckpt.get('stat_features')}")
log()

te_ldr = make_loader(splits["test"])
tr_ldr = make_loader(splits["train"])   # for generalization gap

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 0 — DURATION AUDIT
# ══════════════════════════════════════════════════════════════════════════════
log("─" * 68)
log("CHECK 0 — FILE DURATION AUDIT")
log("─" * 68)
log("  DURATION_SEC=2.75 crops files longer than 2.75 s (center crop).")
log("  Files shorter than 2.75 s would get zero-padded — stats diluted.")
log()

import soundfile as _sf
_dur_sample = random.sample(list(range(len(ALL_PATHS))), min(300, len(ALL_PATHS)))
_durations  = []
for i in _dur_sample:
    try:
        info = _sf.info(str(ALL_PATHS[i]))
        _durations.append(info.duration)
    except Exception:
        pass

if _durations:
    _durations = sorted(_durations)
    _p5  = float(np.percentile(_durations, 5))
    _p50 = float(np.percentile(_durations, 50))
    _p95 = float(np.percentile(_durations, 95))
    log(f"  Sampled {len(_durations)} files")
    log(f"  Min / P5 / Median / P95 / Max : "
        f"{min(_durations):.2f} / {_p5:.2f} / {_p50:.2f} / {_p95:.2f} / {max(_durations):.2f}  sec")
    log()
    if _p5 >= 2.75:
        log(f"  VERDICT: ✓  All files are ≥ 2.75 sec. Preprocessor crops, not pads.")
        log("            Features are computed on real audio — no dilution.")
    elif _p50 >= 2.75:
        log(f"  VERDICT: ⚠️  Some files (< P5) are shorter than 2.75 sec → padded.")
        log(f"            These are a small fraction; most files are fine.")
    else:
        log(f"  VERDICT: ❌  Median file is {_p50:.2f} sec < 2.75 sec → majority padded.")
        log("            Lower DURATION_SEC to match actual clip length.")
else:
    log("  Could not read file durations — soundfile may not be installed.")
log()

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — GENERALIZATION GAP
# ══════════════════════════════════════════════════════════════════════════════
log("─" * 68)
log("CHECK 1 — GENERALIZATION GAP  (train acc vs test acc)")
log("─" * 68)

te_preds, te_true, te_conf, te_probs = eval_loader(model, te_ldr, sm, ss)
tr_preds, tr_true, _,       _        = eval_loader(model, tr_ldr, sm, ss)

test_acc  = accuracy(te_preds, te_true)
train_acc = accuracy(tr_preds, tr_true)
gap       = train_acc - test_acc

log(f"  Train accuracy : {train_acc:.4f}  ({train_acc*100:.2f}%)")
log(f"  Test  accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
log(f"  Generalization gap (train - test): {gap*100:+.2f}%")
log()
if gap > 0.02:
    log("  VERDICT: ⚠️  Gap > 2% — overfit signal. Model fits training set")
    log("            noticeably better than unseen data.")
elif gap > 0.005:
    log("  VERDICT: ⚠️  Mild gap (0.5–2%). Could be normal variance or mild overfit.")
elif gap < -0.005:
    log("  VERDICT: ⚠️  Negative gap (test > train). Training uses heavy augmentation")
    log("            so train accuracy is artificially suppressed — expected.")
    log("            However, if this is LARGE, it may indicate the test set is")
    log("            easier than train (temporal leakage). Run leakage diagnostic.")
else:
    log("  VERDICT: ✓  Gap < 0.5%. Generalization looks healthy.")
log()

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — CONFIDENCE CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
log("─" * 68)
log("CHECK 2 — CONFIDENCE CALIBRATION")
log("─" * 68)
log("  (overfit models are overconfident: near-100% softmax even when wrong)")
log()

correct_mask   = (te_preds == te_true)
mean_conf_all  = te_conf.mean()
mean_conf_cor  = te_conf[correct_mask].mean()
mean_conf_wr   = te_conf[~correct_mask].mean() if (~correct_mask).any() else float("nan")
n_wrong        = int((~correct_mask).sum())

log(f"  Mean confidence on ALL  predictions : {mean_conf_all:.4f}")
log(f"  Mean confidence on CORRECT preds    : {mean_conf_cor:.4f}")
log(f"  Mean confidence on WRONG   preds    : {mean_conf_wr:.4f}  (n={n_wrong})")

# Bucket accuracy vs. confidence (reliability diagram data)
buckets = np.linspace(0.5, 1.0, 11)
log()
log(f"  {'Conf range':<18} {'n samples':>10} {'actual acc':>12}")
for lo, hi in zip(buckets[:-1], buckets[1:]):
    mask = (te_conf >= lo) & (te_conf < hi)
    if mask.sum() > 0:
        acc_b = accuracy(te_preds[mask], te_true[mask])
        log(f"  [{lo:.2f} – {hi:.2f})       {mask.sum():>10}  {acc_b:>12.4f}")
log()

if not np.isnan(mean_conf_wr) and mean_conf_wr > 0.85:
    log("  VERDICT: ⚠️  Wrong predictions still carry >85% confidence.")
    log("            This is a sign of overconfidence / poor calibration.")
elif not np.isnan(mean_conf_wr):
    log(f"  VERDICT: ✓  Wrong predictions have lower confidence ({mean_conf_wr:.3f}).")
    log("            Calibration looks reasonable.")
log()

# Reliability diagram
fig, ax = plt.subplots(figsize=(5, 4))
bucket_accs, bucket_mids = [], []
for lo, hi in zip(buckets[:-1], buckets[1:]):
    mask = (te_conf >= lo) & (te_conf < hi)
    if mask.sum() > 0:
        bucket_accs.append(accuracy(te_preds[mask], te_true[mask]))
        bucket_mids.append((lo + hi) / 2)
ax.plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.4, label="Perfect calibration")
ax.plot(bucket_mids, bucket_accs, "o-", color="purple", label="Model")
ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Actual accuracy")
ax.set_title("Reliability diagram (Phase 2b)"); ax.legend(); ax.set_xlim(0.5, 1); ax.set_ylim(0.5, 1)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "diag_calibration.png"), dpi=120)
plt.close()
log("  Reliability diagram → diag_calibration.png")
log()

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — NOISE ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════
log("─" * 68)
log("CHECK 3 — NOISE ROBUSTNESS  (accuracy vs SNR on test mel)")
log("─" * 68)
log("  Genuine learning: accuracy degrades gradually with more noise.")
log("  Memorisation: accuracy collapses suddenly at low noise.")
log()

model.eval()
sm_t = torch.tensor(sm, dtype=torch.float32, device=DEVICE)
ss_t = torch.tensor(ss, dtype=torch.float32, device=DEVICE)

noise_results = {}
for snr in NOISE_SNRS:
    all_preds_n, all_true_n = [], []
    with torch.no_grad():
        for (mel, stat), y in te_ldr:
            mel  = add_mel_noise(mel, snr)
            mel, stat, y = mel.to(DEVICE), stat.to(DEVICE), y.to(DEVICE)
            stat = (stat - sm_t) / ss_t
            preds = model(mel, stat).argmax(1)
            all_preds_n.extend(preds.cpu().numpy())
            all_true_n.extend(y.cpu().numpy())
    acc_n = accuracy(np.array(all_preds_n), np.array(all_true_n))
    noise_results[snr] = acc_n
    log(f"  SNR = {snr:2d} dB  →  test acc = {acc_n:.4f}  ({acc_n*100:.2f}%)")

log(f"\n  Baseline (no noise) : {test_acc:.4f}")
drop_at_10dB = test_acc - noise_results[10]
log(f"  Drop at 10 dB SNR   : {drop_at_10dB*100:.2f}%")
log()
if drop_at_10dB > 0.10:
    log("  VERDICT: ⚠️  Accuracy drops >10% at moderate noise (10 dB).")
    log("            The model may have memorised recording-specific acoustics.")
elif drop_at_10dB > 0.03:
    log("  VERDICT: ⚠️  Mild degradation at 10 dB. Acceptable but watch closely.")
else:
    log("  VERDICT: ✓  Accuracy stays robust under noise. Model learned signal,")
    log("            not recording artefacts.")

# Plot robustness curve
fig, ax = plt.subplots(figsize=(5, 4))
snrs_sorted = sorted(noise_results.keys(), reverse=True)
ax.plot(snrs_sorted, [noise_results[s] for s in snrs_sorted], "o-", color="purple")
ax.axhline(test_acc, color="gray", linestyle="--", alpha=0.6, label="No noise")
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Test accuracy")
ax.set_title("Noise Robustness (Phase 2b)"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "diag_noise_robustness.png"), dpi=120)
plt.close()
log("  Noise robustness plot → diag_noise_robustness.png")
log()

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — MC-DROPOUT UNCERTAINTY
# ══════════════════════════════════════════════════════════════════════════════
log("─" * 68)
log("CHECK 4 — MC-DROPOUT UNCERTAINTY  (20 stochastic forward passes)")
log("─" * 68)
log("  High variance = model is uncertain (good, honest).")
log("  Near-zero variance = model is always confident — could be overfit.")
log()

N_MC = 20
# Collect all test samples once into tensors (small enough to hold in memory)
all_mel_list, all_stat_list, all_true_list = [], [], []
for (mel, stat), y in te_ldr:
    all_mel_list.append(mel); all_stat_list.append(stat); all_true_list.append(y)
all_mel  = torch.cat(all_mel_list)
all_stat = torch.cat(all_stat_list)
all_true_mc = torch.cat(all_true_list).numpy()

mc_probs = []   # (N_MC, n_test, 6)
with torch.no_grad():
    model.train()   # dropout ON
    for _ in range(N_MC):
        batch_probs = []
        for start in range(0, len(all_mel), BATCH_SIZE):
            mel_b  = all_mel[start:start+BATCH_SIZE].to(DEVICE)
            stat_b = ((all_stat[start:start+BATCH_SIZE].to(DEVICE) - sm_t) / ss_t)
            p = F.softmax(model(mel_b, stat_b), dim=1)
            batch_probs.append(p.cpu().numpy())
        mc_probs.append(np.concatenate(batch_probs, axis=0))
model.eval()

mc_probs   = np.array(mc_probs)          # (20, n_test, 6)
mean_probs = mc_probs.mean(axis=0)       # (n_test, 6)
std_probs  = mc_probs.std(axis=0)        # (n_test, 6) — uncertainty per class
max_std    = std_probs.max(axis=1)       # (n_test,) — worst-case uncertainty per sample

mc_preds = mean_probs.argmax(axis=1)
mc_acc   = accuracy(mc_preds, all_true_mc)

log(f"  MC-Dropout accuracy          : {mc_acc:.4f}")
log(f"  Mean max-class std (all test): {max_std.mean():.4f}")
log(f"  Samples with std > 0.05      : {(max_std > 0.05).sum()} / {len(max_std)}")
log(f"  Samples with std > 0.10      : {(max_std > 0.10).sum()} / {len(max_std)}")
log()

correct_mask_mc = (mc_preds == all_true_mc)
std_correct = max_std[correct_mask_mc].mean()
std_wrong   = max_std[~correct_mask_mc].mean() if (~correct_mask_mc).any() else float("nan")
log(f"  Mean uncertainty on CORRECT predictions : {std_correct:.4f}")
log(f"  Mean uncertainty on WRONG   predictions : {std_wrong:.4f}")
log()

if not np.isnan(std_wrong) and std_wrong < 0.05:
    log("  VERDICT: ⚠️  Wrong predictions have very low MC uncertainty.")
    log("            The model is confidently wrong — classic overfit symptom.")
elif max_std.mean() < 0.01:
    log("  VERDICT: ⚠️  Near-zero variance across all runs. Dropout is not")
    log("            providing meaningful regularisation signal.")
else:
    log("  VERDICT: ✓  MC-Dropout shows reasonable uncertainty on hard samples.")
log()

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — PER-CLASS BIAS
# ══════════════════════════════════════════════════════════════════════════════
log("─" * 68)
log("CHECK 5 — PER-CLASS BIAS  (precision / recall / F1)")
log("─" * 68)

prec, rec, f1, sup = precision_recall_fscore_support(
    te_true, te_preds, labels=list(range(6)), zero_division=0)

log(f"\n  {'Class':<14} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
log("  " + "-" * 52)
for i, name in enumerate(CLASS_NAMES):
    flag = ""
    if rec[i] < 0.97:  flag = "  ← LOW RECALL"
    if prec[i] < 0.97: flag = "  ← LOW PRECISION"
    log(f"  {name:<14} {prec[i]:>10.4f} {rec[i]:>8.4f} {f1[i]:>8.4f} {int(sup[i]):>9}{flag}")

# Normal vs Abnormal aggregate
norm_mask = np.isin(te_true, list(NORMAL_IDS))
abn_mask  = ~norm_mask
log(f"\n  Normal   classes — accuracy: {accuracy(te_preds[norm_mask], te_true[norm_mask]):.4f}")
log(f"  Abnormal classes — accuracy: {accuracy(te_preds[abn_mask],  te_true[abn_mask]):.4f}")

norm_pred_count = np.isin(te_preds, list(NORMAL_IDS)).sum()
abn_pred_count  = np.isin(te_preds, list(ABNORMAL_IDS)).sum()
log(f"\n  Predicted Normal   : {norm_pred_count}  (true Normal   : {norm_mask.sum()})")
log(f"  Predicted Abnormal : {abn_pred_count}  (true Abnormal : {abn_mask.sum()})")

bias_pct = abs(norm_pred_count - norm_mask.sum()) / max(norm_mask.sum(), 1) * 100
if bias_pct > 2:
    log(f"\n  VERDICT: ⚠️  Prediction counts differ from ground truth by {bias_pct:.1f}%.")
    log("            Model has a Normal/Abnormal prediction bias.")
else:
    log(f"\n  VERDICT: ✓  Normal/Abnormal prediction balance matches ground truth")
    log(f"            within {bias_pct:.1f}%.")
log()

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6 — CROSS-MACHINE BIAS
# ══════════════════════════════════════════════════════════════════════════════
log("─" * 68)
log("CHECK 6 — CROSS-MACHINE BIAS  (per-machine test accuracy)")
log("─" * 68)
log("  If accuracy is very uneven across machines, the model may rely on")
log("  machine identity rather than fault signatures.")
log()

machine_labels = ["Machine 1", "Machine 2", "Machine 3"]
machine_accs   = []
for m_idx, cls_ids in MACHINE_IDS.items():
    mask = np.isin(te_true, cls_ids)
    if mask.sum() == 0: continue
    m_acc = accuracy(te_preds[mask], te_true[mask])
    machine_accs.append(m_acc)

    # Normal/Abnormal within this machine
    norm_m = np.isin(te_true, [cls_ids[0]])
    abn_m  = np.isin(te_true, [cls_ids[1]])
    m_norm_acc = accuracy(te_preds[norm_m & mask], te_true[norm_m & mask]) if (norm_m & mask).any() else float("nan")
    m_abn_acc  = accuracy(te_preds[abn_m  & mask], te_true[abn_m  & mask]) if (abn_m  & mask).any() else float("nan")

    log(f"  {machine_labels[m_idx]}:  overall={m_acc:.4f}   Normal={m_norm_acc:.4f}   Abnormal={m_abn_acc:.4f}   (n={mask.sum()})")

if len(machine_accs) >= 2:
    spread = max(machine_accs) - min(machine_accs)
    log()
    if spread > 0.02:
        log(f"  VERDICT: ⚠️  Machine accuracy spread = {spread*100:.2f}%. Model is not")
        log("            equally reliable across machines.")
    else:
        log(f"  VERDICT: ✓  Machine accuracy spread = {spread*100:.2f}%. Consistent")
        log("            performance across all three machines.")
log()

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 7 — PREDICTION DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
log("─" * 68)
log("CHECK 7 — PREDICTION DISTRIBUTION vs GROUND TRUTH")
log("─" * 68)
log()

true_counts = np.bincount(te_true, minlength=6)
pred_counts = np.bincount(te_preds, minlength=6)

log(f"  {'Class':<14} {'True count':>12} {'Pred count':>12} {'Δ':>8}")
log("  " + "-" * 50)
for i, name in enumerate(CLASS_NAMES):
    delta = int(pred_counts[i]) - int(true_counts[i])
    flag  = "  ← over-predicted" if delta > 10 else ("  ← under-predicted" if delta < -10 else "")
    log(f"  {name:<14} {true_counts[i]:>12} {pred_counts[i]:>12} {delta:>+8}{flag}")

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(6)
w = 0.35
ax.bar(x - w/2, true_counts, w, label="Ground truth", color="steelblue", alpha=0.8)
ax.bar(x + w/2, pred_counts, w, label="Predictions",  color="purple",    alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
ax.set_ylabel("Count"); ax.set_title("Prediction distribution vs Ground truth")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "diag_pred_distribution.png"), dpi=120)
plt.close()
log("\n  Distribution plot → diag_pred_distribution.png")
log()

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 68)
log("OVERALL DIAGNOSTIC SUMMARY")
log("=" * 68)
log(f"  Generalization gap      : {gap*100:+.2f}%  (train={train_acc:.4f}, test={test_acc:.4f})")
log(f"  Confidence on errors    : {mean_conf_wr:.4f}")
log(f"  MC-Dropout mean std     : {max_std.mean():.4f}")
log(f"  Accuracy drop @ 10dB    : {drop_at_10dB*100:.2f}%")
log(f"  Machine accuracy spread : {(max(machine_accs)-min(machine_accs))*100:.2f}%")
log()
log("  Files written to /kaggle/working/:")
log("    diag_calibration.png")
log("    diag_noise_robustness.png")
log("    diag_pred_distribution.png")
log("    diagnostic_report.txt")
log()
log("  NEXT STEP: run kaggle_diagnose_leakage.py to check for temporal")
log("  leakage.  If leakage is confirmed, swap to split_indices_clean.json")
log("  and re-run training — compare test accuracy before and after.")
log("=" * 68)

# Write full report to file
with open(REPORT_PATH, "w") as f:
    f.write("\n".join(lines))
print(f"\nFull report written → {REPORT_PATH}")
