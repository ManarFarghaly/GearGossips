import os, random, pathlib, time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import tqdm
from concurrent.futures import ThreadPoolExecutor
from machine_listener.src.preprocess import AudioPreprocessor, PreprocessConfig, AugmentationConfig
from machine_listener.src.dataset import scan_wav_files, SPLIT_DIR
from machine_listener.src.features.mel_spectrogram import compute_mel_spectrogram
from machine_listener.src.models.cnn_baseline import MelCNN, MelCNNHier
from machine_listener.src.split_utils import create_clean_split, load_clean_split
import machine_listener.src.train_utils as utils

ROOT_DIR   = "Students"
MODELS_DIR = "machine_listener/outputs/saved_models"
FEATS_DIR  = os.path.normpath(os.path.join(MODELS_DIR, "..", "features", "mel"))
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FEATS_DIR,  exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 32
MAX_EPOCHS  = 20
LR          = 1e-3
WEIGHT_DECAY = 5e-4
ES_PATIENCE = 5
NUM_WORKERS = 4

CLASS_NAMES = ["Machine1_Normal","Machine1_Abnormal",
               "Machine2_Normal","Machine2_Abnormal",
               "Machine3_Normal","Machine3_Abnormal"]

_MACHINE_FROM_CLASS = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2}
_FAULT_FROM_CLASS   = {0:0, 1:1, 2:0, 3:1, 4:0, 5:1}

print(f"Device: {DEVICE}")

@dataclass
class AblCfg:
    name:         str
    use_hier:     bool
    use_ens:      bool
    label_smooth: float
    mixup_alpha:  float
    use_focal:    bool
    use_rlrop:    bool
    use_warmup:   bool

CONFIGS = [
    AblCfg("A_baseline",    use_hier=False, use_ens=False, label_smooth=0.0,
            mixup_alpha=0.0, use_focal=False, use_rlrop=False, use_warmup=False),
    AblCfg("B_ens_ls",      use_hier=False, use_ens=True,  label_smooth=0.1,
            mixup_alpha=0.0, use_focal=False, use_rlrop=False, use_warmup=True),
    AblCfg("C_mixup",       use_hier=False, use_ens=True,  label_smooth=0.1,
            mixup_alpha=0.1, use_focal=False, use_rlrop=False, use_warmup=True),
    AblCfg("D_hier_focal",  use_hier=True,  use_ens=True,  label_smooth=0.1,
            mixup_alpha=0.1, use_focal=True,  use_rlrop=False, use_warmup=True),
    AblCfg("E_rlrop",       use_hier=True,  use_ens=True,  label_smooth=0.1,
            mixup_alpha=0.1, use_focal=True,  use_rlrop=True,  use_warmup=True),
]


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

def precompute_mel(paths, feats_dir, preprocessor, n_workers=4):
    feats_dir = pathlib.Path(feats_dir)
    feats_dir.mkdir(parents=True, exist_ok=True)
    done = sum(1 for i in range(len(paths)) if (feats_dir / f"{i:06d}.npy").exists())
    if done == len(paths):
        print(f"All {len(paths)} mel-specs cached."); return
    print(f"Pre-computing {len(paths) - done} mel-specs ...")
    def _one(args):
        idx, wav = args
        out = feats_dir / f"{idx:06d}.npy"
        if out.exists(): return
        try: np.save(out, compute_mel_spectrogram(preprocessor.preprocess(str(wav), mode="inference")))
        except Exception: np.save(out, np.zeros((1, 128, 84), dtype=np.float32))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(tqdm.tqdm(ex.map(_one, enumerate(paths)), total=len(paths), desc="mel"))
    print("Done.")

class MelDataset(Dataset):
    def __init__(self, feats_dir, labels, indices, augment=False):
        self.feats_dir = pathlib.Path(feats_dir)
        self.labels = labels; self.indices = indices; self.augment = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ri  = self.indices[idx]
        lbl = self.labels[ri]
        mel = torch.tensor(np.load(self.feats_dir / f"{ri:06d}.npy"), dtype=torch.float32)
        if self.augment: mel = spec_augment(mel)
        return (mel,
                torch.tensor(lbl,                      dtype=torch.long),
                torch.tensor(_MACHINE_FROM_CLASS[lbl], dtype=torch.long),
                torch.tensor(_FAULT_FROM_CLASS[lbl],   dtype=torch.long))

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma; self.pos_weight = pos_weight

    def forward(self, logits, targets):
        t   = targets.float().unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, t,pos_weight=self.pos_weight, reduction="none")
        return ((1.0 - torch.exp(-bce)) ** self.gamma * bce).mean()

def effective_num_weights(label_counts, beta=0.9999, num_classes=6):
    eff = (1.0 - np.power(beta, label_counts)) / (1.0 - beta)
    w   = 1.0 / eff
    return torch.tensor(w / w.sum() * num_classes, dtype=torch.float32)

def build_losses(cfg, label_counts, device):
    cw = effective_num_weights(label_counts).to(device) if cfg.use_ens \
         else torch.ones(6, dtype=torch.float32).to(device)
    crit_main    = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.label_smooth)
    crit_machine = crit_fault = None

    if cfg.use_hier:
        mc = np.array([label_counts[0]+label_counts[1],
                       label_counts[2]+label_counts[3],
                       label_counts[4]+label_counts[5]], dtype=np.float64)
        me = (1.0 - np.power(0.9999, mc)) / (1.0 - 0.9999)
        mw = torch.tensor(1.0/me / (1.0/me).sum() * 3, dtype=torch.float32).to(device)
        crit_machine = nn.CrossEntropyLoss(weight=mw)
        fc  = np.array([sum(label_counts[i] for i in [0,2,4]),
                        sum(label_counts[i] for i in [1,3,5])], dtype=np.float64)
        fpw = torch.tensor([fc[0]/(fc[1]+1e-6)], dtype=torch.float32).to(device)
        crit_fault = (BinaryFocalLoss(gamma=2.0, pos_weight=fpw) if cfg.use_focal
                      else nn.BCEWithLogitsLoss(pos_weight=fpw))
    return crit_main, crit_machine, crit_fault

def build_schedulers(cfg, optimizer):
    warmup = (torch.optim.lr_scheduler.LambdaLR(
                  optimizer, lambda e: (e+1)/2 if e < 2 else 1.0)
              if cfg.use_warmup else None)
    main   = (torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)
              if cfg.use_rlrop else
              torch.optim.lr_scheduler.CosineAnnealingLR(
                  optimizer, T_max=MAX_EPOCHS, eta_min=1e-5))
    return warmup, main

def mixup_batch(x, y, alpha, device):
    if alpha <= 0: return x, y, y, 1.0
    lam  = float(np.random.beta(alpha, alpha))
    lam  = max(lam, 1.0 - lam)
    perm = torch.randperm(x.size(0), device=device)
    return lam*x + (1-lam)*x[perm], y, y[perm], lam

def mixup_loss(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1-lam) * crit(pred, yb)


def train_epoch(model, loader, optimizer, cfg,
                crit_main, crit_machine, crit_fault, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for mel, y_main, y_machine, y_fault in loader:
        mel, y_main = mel.to(device), y_main.to(device)
        y_machine, y_fault = y_machine.to(device), y_fault.to(device)
        mel_mix, ya, yb, lam = mixup_batch(mel, y_main, cfg.mixup_alpha, device)
        optimizer.zero_grad()

        if cfg.use_hier:
            out_main, out_mach, out_fault = model(mel_mix)
            loss = (mixup_loss(crit_main, out_main, ya, yb, lam)
                    + 0.4 * crit_machine(out_mach, y_machine)
                    + 0.6 * crit_fault(out_fault, y_fault))
        else:
            out_main = model(mel_mix)
            loss = mixup_loss(crit_main, out_main, ya, yb, lam)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += (out_main.argmax(1) == ya).sum().item()
        total      += y_main.size(0)
    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, crit_main, device, use_hier):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, labels = [], []
    with torch.no_grad():
        for mel, y_main, y_machine, y_fault in loader:
            mel, y_main = mel.to(device), y_main.to(device)
            out = model(mel)
            out_main = out[0] if use_hier else out
            loss = crit_main(out_main, y_main)
            p    = out_main.argmax(1)
            total_loss += loss.item()
            correct    += (p == y_main).sum().item()
            total      += y_main.size(0)
            preds.extend(p.cpu().numpy())
            labels.extend(y_main.cpu().numpy())
    return total_loss / len(loader), correct / total, preds, labels


def run_training(model, tr_ldr, vl_ldr, optimizer, cfg,
                 crit_main, crit_machine, crit_fault,
                 warmup_sched, main_sched, device, ckpt_path):
    best_loss = float("inf")
    es = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        lr = optimizer.param_groups[0]["lr"]
        tr_l, tr_a = train_epoch(model, tr_ldr, optimizer, cfg,
                                  crit_main, crit_machine, crit_fault, device)
        vl_l, vl_a, _, _ = eval_epoch(model, vl_ldr, crit_main, device, cfg.use_hier)

        if warmup_sched and epoch <= 2:
            warmup_sched.step()
        elif cfg.use_rlrop:
            main_sched.step(vl_l)
        else:
            main_sched.step()

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

split_file = pathlib.Path(SPLIT_DIR) / "split_indices_clean.json"
if split_file.exists():
    _splits = load_clean_split(SPLIT_DIR)
    print(f"Loaded existing split from {split_file}")
else:
    _splits = create_clean_split(ALL_PATHS, ALL_LABELS, SPLIT_DIR)

precompute_mel(ALL_PATHS, FEATS_DIR, _infer_prep, n_workers=NUM_WORKERS)

label_counts = np.bincount([ALL_LABELS[i] for i in _splits["train"]], minlength=6)
print(f"\nClass counts: {dict(zip(CLASS_NAMES, label_counts))}")

tr_ds = MelDataset(FEATS_DIR, ALL_LABELS, _splits["train"], augment=True)
vl_ds = MelDataset(FEATS_DIR, ALL_LABELS, _splits["val"],   augment=False)
te_ds = MelDataset(FEATS_DIR, ALL_LABELS, _splits["test"],  augment=False)
print(f"Train: {len(tr_ds)}  Val: {len(vl_ds)}  Test: {len(te_ds)}")

tr_ldr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=True)
vl_ldr = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)
te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)

results = []

for cfg in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Config: {cfg.name}")
    print(f"  hier={cfg.use_hier}  ens={cfg.use_ens}  ls={cfg.label_smooth}"
          f"  mixup={cfg.mixup_alpha}  focal={cfg.use_focal}"
          f"  rlrop={cfg.use_rlrop}")
    print(f"{'='*60}")

    model = (MelCNNHier if cfg.use_hier else MelCNN)(num_classes=6).to(DEVICE)
    crit_main, crit_machine, crit_fault = build_losses(cfg, label_counts, DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup_sched, main_sched = build_schedulers(cfg, optimizer)
    ckpt = os.path.join(MODELS_DIR, f"phase1_abl_{cfg.name}.pth")

    run_training(model, tr_ldr, vl_ldr, optimizer, cfg,
                 crit_main, crit_machine, crit_fault,
                 warmup_sched, main_sched, DEVICE, ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    _, _, test_preds, test_labels = eval_epoch(model, te_ldr, crit_main, DEVICE, cfg.use_hier)

    f1s   = f1_score(test_labels, test_preds, average=None, zero_division=0)
    macro = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    acc   = np.mean(np.array(test_preds) == np.array(test_labels))
    results.append({"name": cfg.name, "f1": f1s, "macro": macro, "acc": acc})

    print(f"\n  Test  macro_f1={macro:.4f}  acc={acc:.4f}")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))


print(f"\n{'Config':<16} {'M1N':>5} {'M1A':>5} {'M2N':>5} {'M2A':>5} {'M3N':>5} {'M3A':>5} {'Macro':>7} {'Acc':>6}")
print("-" * 68)
for r in results:
    f = r["f1"]
    print(f"{r['name']:<16} "
          f"{f[0]:>5.3f} {f[1]:>5.3f} {f[2]:>5.3f} {f[3]:>5.3f} "
          f"{f[4]:>5.3f} {f[5]:>5.3f} {r['macro']:>7.3f} {r['acc']:>6.3f}")
