import os, random, pathlib, time, pickle
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
import tqdm
from concurrent.futures import ThreadPoolExecutor
from machine_listener.src.preprocess import AudioPreprocessor, PreprocessConfig, AugmentationConfig
from machine_listener.src.dataset import scan_wav_files, SPLIT_DIR
from machine_listener.src.features.mel_spectrogram import compute_mel_spectrogram
from machine_listener.src.features.statistical import compute_stat_features_v3, ALL_FEATURE_NAMES_V3
from machine_listener.src.models.cnn_baseline import MelCNNHier
from machine_listener.src.models.cnn_mel_stat import MelStatCNNHier
from machine_listener.src.split_utils import load_clean_split
import machine_listener.src.train_utils as utils


ROOT_DIR    = "Students"
MODELS_DIR  = "machine_listener/outputs/saved_models"
PHASE1_CKPT = os.path.join(MODELS_DIR, "phase1_v2_best.pth")
FEATS_DIR_MEL  = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mel"))
FEATS_DIR_STAT = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "stat_v3"))
os.makedirs(MODELS_DIR,     exist_ok=True)
os.makedirs(FEATS_DIR_MEL,  exist_ok=True)
os.makedirs(FEATS_DIR_STAT, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 32
MAX_EPOCHS  = 20
NUM_WORKERS = 4
WEIGHT_DECAY = 5e-4
LABEL_SMOOTH = 0.1
ENS_BETA     = 0.9999
MIXUP_ALPHA  = 0.1
FOCAL_GAMMA  = 2.0
HIER_ALPHA   = 0.4
ES_PATIENCE  = 5
LR_MEL       = 3e-4
LR_NEW       = 5e-4

CLASS_NAMES = ["Machine1_Normal","Machine1_Abnormal",
               "Machine2_Normal","Machine2_Abnormal",
               "Machine3_Normal","Machine3_Abnormal"]

_MACHINE_FROM_CLASS = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2}
_FAULT_FROM_CLASS   = {0:0, 1:1, 2:0, 3:1, 4:0, 5:1}

# Full 6-feature list — subsets are selected by index
ALL_STAT = ALL_FEATURE_NAMES_V3   # ["rms", "zcr", "rolloff", "bandwidth", "kurtosis", "spectral_flux"]

print(f"Device: {DEVICE}")

if not os.path.exists(PHASE1_CKPT):
    raise FileNotFoundError(
        f"Phase 1 V2 checkpoint not found at {PHASE1_CKPT}. Run train_phase1V2.py first.")

@dataclass
class AblCfg:
    name:     str
    features: list

CONFIGS = [
    AblCfg("A_mel_only",  features=[]),
    AblCfg("B_basic4",    features=["rms","zcr","rolloff","bandwidth"]),
    AblCfg("C_kurtosis",  features=["rms","zcr","rolloff","bandwidth","kurtosis"]),
    AblCfg("D_flux",      features=["rms","zcr","rolloff","bandwidth","spectral_flux"]),
    AblCfg("E_all6",      features=["rms","zcr","rolloff","bandwidth","kurtosis","spectral_flux"]),
]

def feat_indices(features):
    return [ALL_STAT.index(f) for f in features]

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


def precompute_all(paths, mel_dir, stat_dir, preprocessor, n_workers=4):
    mel_dir  = pathlib.Path(mel_dir);  mel_dir.mkdir(parents=True, exist_ok=True)
    stat_dir = pathlib.Path(stat_dir); stat_dir.mkdir(parents=True, exist_ok=True)

    done_mel  = sum(1 for i in range(len(paths)) if (mel_dir  / f"{i:06d}.npy").exists())
    done_stat = sum(1 for i in range(len(paths)) if (stat_dir / f"{i:06d}.npy").exists())

    def _one(args):
        idx, wav = args
        mo = mel_dir  / f"{idx:06d}.npy"
        so = stat_dir / f"{idx:06d}.npy"
        if mo.exists() and so.exists(): return
        try:
            w = preprocessor.preprocess(str(wav), mode="inference")
            if not mo.exists(): np.save(mo, compute_mel_spectrogram(w))
            if not so.exists(): np.save(so, compute_stat_features_v3(w))
        except Exception:
            if not mo.exists(): np.save(mo, np.zeros((1, 128, 84), dtype=np.float32))
            if not so.exists(): np.save(so, np.zeros(6, dtype=np.float32))

    if done_mel < len(paths) or done_stat < len(paths):
        print(f"Pre-computing features for {len(paths)} files ...")
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(tqdm.tqdm(ex.map(_one, enumerate(paths)), total=len(paths), desc="mel+stat"))
        print("Done.")
    else:
        print(f"All {len(paths)} mel+stat features cached.")


def fit_scaler(stat_dir, indices, fidx):
    arr = np.stack([np.load(pathlib.Path(stat_dir) / f"{i:06d}.npy")[fidx] for i in indices])
    return arr.mean(0), arr.std(0) + 1e-8


def load_v2_weights(model, ckpt_path, device):
    """Load Phase 1 V2 backbone (flat head) — backbone keys load, head keys are ignored."""
    ckpt     = torch.load(ckpt_path, map_location=device)
    p1_state = ckpt["model_state_dict"]

    if hasattr(model, "mel_stream"):
        remapped = {"mel_stream." + k: v for k, v in p1_state.items()}
        missing, _ = model.load_state_dict(remapped, strict=False)
        loaded = len(remapped) - len([k for k in missing if k.startswith("mel_stream.")])
        print(f"[ckpt] Loaded {loaded}/{len(remapped)} keys into mel_stream.")
    else:
        missing, _ = model.load_state_dict(p1_state, strict=False)
        print(f"[ckpt] Loaded {len(p1_state)-len(missing)}/{len(p1_state)} keys.")


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma; self.pos_weight = pos_weight

    def forward(self, logits, targets):
        t   = targets.float().unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, t,
                                                  pos_weight=self.pos_weight, reduction="none")
        return ((1.0 - torch.exp(-bce)) ** self.gamma * bce).mean()

def effective_num_weights(label_counts, beta=0.9999, n=6):
    eff = (1.0 - np.power(beta, label_counts)) / (1.0 - beta)
    w   = 1.0 / eff
    return torch.tensor(w / w.sum() * n, dtype=torch.float32)

def build_losses(label_counts, device):
    cw = effective_num_weights(label_counts).to(device)
    mc = np.array([label_counts[0]+label_counts[1],
                   label_counts[2]+label_counts[3],
                   label_counts[4]+label_counts[5]], dtype=np.float64)
    me = (1.0 - np.power(ENS_BETA, mc)) / (1.0 - ENS_BETA)
    mw = torch.tensor(1.0/me / (1.0/me).sum() * 3, dtype=torch.float32).to(device)
    fc  = np.array([sum(label_counts[i] for i in [0,2,4]),
                    sum(label_counts[i] for i in [1,3,5])], dtype=np.float64)
    fpw = torch.tensor([fc[0]/(fc[1]+1e-6)], dtype=torch.float32).to(device)
    return (nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTH),
            nn.CrossEntropyLoss(weight=mw),
            BinaryFocalLoss(gamma=FOCAL_GAMMA, pos_weight=fpw))


def mixup_batch(x, y, device):
    lam  = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
    lam  = max(lam, 1.0 - lam)
    perm = torch.randperm(x.size(0), device=device)
    return lam*x + (1-lam)*x[perm], y, y[perm], lam

def mixup_loss(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1-lam) * crit(pred, yb)


class MelOnlyDataset(Dataset):
    def __init__(self, mel_dir, labels, indices, augment=False):
        self.mel_dir = pathlib.Path(mel_dir)
        self.labels  = labels; self.indices = indices; self.augment = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri  = self.indices[idx]
        lbl = self.labels[ri]
        mel = torch.tensor(np.load(self.mel_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment: mel = spec_augment(mel)
        return (mel,
                torch.tensor(lbl,                      dtype=torch.long),
                torch.tensor(_MACHINE_FROM_CLASS[lbl], dtype=torch.long),
                torch.tensor(_FAULT_FROM_CLASS[lbl],   dtype=torch.long))


class MelStatDataset(Dataset):
    def __init__(self, mel_dir, stat_dir, labels, indices, feat_idx, augment=False):
        self.mel_dir  = pathlib.Path(mel_dir)
        self.stat_dir = pathlib.Path(stat_dir)
        self.labels   = labels; self.indices = indices
        self.feat_idx = feat_idx; self.augment = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri  = self.indices[idx]
        lbl = self.labels[ri]
        mel  = torch.tensor(np.load(self.mel_dir  / f"{ri:06d}.npy"), dtype=torch.float32)
        stat = torch.tensor(np.load(self.stat_dir / f"{ri:06d}.npy")[self.feat_idx], dtype=torch.float32)
        if self.augment: mel = spec_augment(mel)
        return (mel, stat,
                torch.tensor(lbl,                      dtype=torch.long),
                torch.tensor(_MACHINE_FROM_CLASS[lbl], dtype=torch.long),
                torch.tensor(_FAULT_FROM_CLASS[lbl],   dtype=torch.long))


def train_epoch_mel(model, loader, optimizer, crit_main, crit_machine, crit_fault, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for mel, y_main, y_machine, y_fault in loader:
        mel, y_main = mel.to(device), y_main.to(device)
        y_machine, y_fault = y_machine.to(device), y_fault.to(device)
        mel_mix, ya, yb, lam = mixup_batch(mel, y_main, device)
        optimizer.zero_grad()
        out_main, out_mach, out_fault = model(mel_mix)
        loss = (mixup_loss(crit_main, out_main, ya, yb, lam)
                + HIER_ALPHA * crit_machine(out_mach, y_machine)
                + (1-HIER_ALPHA) * crit_fault(out_fault, y_fault))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += (out_main.argmax(1) == ya).sum().item()
        total      += y_main.size(0)
    return total_loss / len(loader), correct / total


def train_epoch_stat(model, loader, optimizer, crit_main, crit_machine, crit_fault,
                     sm, ss, device):
    model.train()
    sm_t = torch.tensor(sm, dtype=torch.float32, device=device)
    ss_t = torch.tensor(ss, dtype=torch.float32, device=device)
    total_loss, correct, total = 0.0, 0, 0
    for mel, stat, y_main, y_machine, y_fault in loader:
        mel, stat = mel.to(device), stat.to(device)
        y_main, y_machine, y_fault = y_main.to(device), y_machine.to(device), y_fault.to(device)
        stat = (stat - sm_t) / ss_t
        mel_mix, ya, yb, lam = mixup_batch(mel, y_main, device)
        optimizer.zero_grad()
        out_main, out_mach, out_fault = model(mel_mix, stat)
        loss = (mixup_loss(crit_main, out_main, ya, yb, lam)
                + HIER_ALPHA * crit_machine(out_mach, y_machine)
                + (1-HIER_ALPHA) * crit_fault(out_fault, y_fault))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += (out_main.argmax(1) == ya).sum().item()
        total      += y_main.size(0)
    return total_loss / len(loader), correct / total


def eval_epoch_mel(model, loader, crit_main, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, labels = [], []
    with torch.no_grad():
        for mel, y_main, y_machine, y_fault in loader:
            mel, y_main = mel.to(device), y_main.to(device)
            out_main, _, _ = model(mel)
            loss = crit_main(out_main, y_main)
            p = out_main.argmax(1)
            total_loss += loss.item()
            correct    += (p == y_main).sum().item()
            total      += y_main.size(0)
            preds.extend(p.cpu().numpy())
            labels.extend(y_main.cpu().numpy())
    return total_loss / len(loader), correct / total, preds, labels


def eval_epoch_stat(model, loader, crit_main, sm, ss, device):
    model.eval()
    sm_t = torch.tensor(sm, dtype=torch.float32, device=device)
    ss_t = torch.tensor(ss, dtype=torch.float32, device=device)
    total_loss, correct, total = 0.0, 0, 0
    preds, labels = [], []
    with torch.no_grad():
        for mel, stat, y_main, y_machine, y_fault in loader:
            mel, stat, y_main = mel.to(device), stat.to(device), y_main.to(device)
            stat = (stat - sm_t) / ss_t
            out_main, _, _ = model(mel, stat)
            loss = crit_main(out_main, y_main)
            p = out_main.argmax(1)
            total_loss += loss.item()
            correct    += (p == y_main).sum().item()
            total      += y_main.size(0)
            preds.extend(p.cpu().numpy())
            labels.extend(y_main.cpu().numpy())
    return total_loss / len(loader), correct / total, preds, labels


def run_training(model, tr_ldr, vl_ldr, optimizer,
                 crit_main, crit_machine, crit_fault,
                 warmup_sched, plateau_sched, device,
                 ckpt_path, sm=None, ss=None, use_stat=False):
    best_loss = float("inf")
    es = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        lr = optimizer.param_groups[0]["lr"]
        if use_stat:
            tr_l, tr_a = train_epoch_stat(model, tr_ldr, optimizer,
                                          crit_main, crit_machine, crit_fault, sm, ss, device)
            vl_l, vl_a, _, _ = eval_epoch_stat(model, vl_ldr, crit_main, sm, ss, device)
        else:
            tr_l, tr_a = train_epoch_mel(model, tr_ldr, optimizer,
                                         crit_main, crit_machine, crit_fault, device)
            vl_l, vl_a, _, _ = eval_epoch_mel(model, vl_ldr, crit_main, device)

        if epoch <= 2:
            warmup_sched.step()
        else:
            plateau_sched.step(vl_l)

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

_infer_prep = AudioPreprocessor(PreprocessConfig(
    target_sr=16000, default_duration_sec=2.75,
    augmentation=AugmentationConfig(enabled=False),
))
ALL_PATHS, ALL_LABELS = scan_wav_files(ROOT_DIR)
print(f"Found {len(ALL_PATHS)} files")

_splits      = load_clean_split(SPLIT_DIR)
label_counts = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
crit_main, crit_machine, crit_fault = build_losses(label_counts, DEVICE)

precompute_all(ALL_PATHS, FEATS_DIR_MEL, FEATS_DIR_STAT, _infer_prep, n_workers=NUM_WORKERS)

print(f"\nClass counts: {dict(zip(CLASS_NAMES, label_counts))}")

results = []

for cfg in CONFIGS:
    use_stat = len(cfg.features) > 0
    fidx     = feat_indices(cfg.features) if use_stat else []
    stat_dim = len(fidx)

    print(f"\n{'='*60}")
    print(f"Config: {cfg.name}  features={cfg.features or 'none (mel only)'}")
    print(f"{'='*60}")

    if use_stat:
        sm, ss = fit_scaler(FEATS_DIR_STAT, _splits["train"], fidx)
        model  = MelStatCNNHier(num_classes=6, stat_dim=stat_dim).to(DEVICE)
        load_v2_weights(model, PHASE1_CKPT, DEVICE)
        optimizer = torch.optim.AdamW([
            {"params": model.mel_stream.parameters(),  "lr": LR_MEL},
            {"params": model.stat_branch.parameters(), "lr": LR_NEW},
            {"params": model.fc1.parameters(),         "lr": LR_NEW},
            {"params": model.head_main.parameters(),   "lr": LR_NEW},
            {"params": model.head_machine.parameters(),"lr": LR_NEW},
            {"params": model.head_fault.parameters(),  "lr": LR_NEW},
        ], weight_decay=WEIGHT_DECAY)
        tr_ds = MelStatDataset(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS,
                               _splits["train"], fidx, augment=True)
        vl_ds = MelStatDataset(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS,
                               _splits["val"],   fidx, augment=False)
        te_ds = MelStatDataset(FEATS_DIR_MEL, FEATS_DIR_STAT, ALL_LABELS,
                               _splits["test"],  fidx, augment=False)
    else:
        sm = ss = None
        model = MelCNNHier(num_classes=6).to(DEVICE)
        load_v2_weights(model, PHASE1_CKPT, DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=LR_MEL, weight_decay=WEIGHT_DECAY)
        tr_ds = MelOnlyDataset(FEATS_DIR_MEL, ALL_LABELS, _splits["train"], augment=True)
        vl_ds = MelOnlyDataset(FEATS_DIR_MEL, ALL_LABELS, _splits["val"],   augment=False)
        te_ds = MelOnlyDataset(FEATS_DIR_MEL, ALL_LABELS, _splits["test"],  augment=False)

    tr_ldr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True)
    vl_ldr = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    warmup_sched  = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: (e+1)/2 if e < 2 else 1.0)
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    ckpt = os.path.join(MODELS_DIR, f"phase2b_abl_{cfg.name}.pth")
    run_training(model, tr_ldr, vl_ldr, optimizer,
                 crit_main, crit_machine, crit_fault,
                 warmup_sched, plateau_sched, DEVICE,
                 ckpt, sm=sm, ss=ss, use_stat=use_stat)

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    if use_stat:
        _, _, test_preds, test_labels = eval_epoch_stat(model, te_ldr, crit_main, sm, ss, DEVICE)
    else:
        _, _, test_preds, test_labels = eval_epoch_mel(model, te_ldr, crit_main, DEVICE)

    f1s   = f1_score(test_labels, test_preds, average=None, zero_division=0)
    macro = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    acc   = np.mean(np.array(test_preds) == np.array(test_labels))
    results.append({"name": cfg.name, "f1": f1s, "macro": macro, "acc": acc})

    print(f"\n  Test  macro_f1={macro:.4f}  acc={acc:.4f}")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))

    if use_stat:
        pkl_path = os.path.join(MODELS_DIR, f"stat_scaler_abl_{cfg.name}.pkl")
        with open(pkl_path, "wb") as fh:
            import pickle
            pickle.dump({"mean": sm, "std": ss, "features": cfg.features}, fh)
        print(f"  Scaler saved → {pkl_path}")


print(f"\n{'Config':<16} {'M1N':>5} {'M1A':>5} {'M2N':>5} {'M2A':>5} {'M3N':>5} {'M3A':>5} {'Macro':>7} {'Acc':>6}")
print("-" * 68)
for r in results:
    f = r["f1"]
    print(f"{r['name']:<16} "
          f"{f[0]:>5.3f} {f[1]:>5.3f} {f[2]:>5.3f} {f[3]:>5.3f} "
          f"{f[4]:>5.3f} {f[5]:>5.3f} {r['macro']:>7.3f} {r['acc']:>6.3f}")
