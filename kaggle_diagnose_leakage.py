"""
Leakage & Bias Diagnostic for the Machine-Fault Dataset
========================================================

Run this as a standalone Kaggle notebook cell (or locally) BEFORE training.
It does four things:

  1. PATH AUDIT        — prints sample file paths per class so you can see
                         the naming convention and detect sequential segments.

  2. TEMPORAL LEAKAGE  — sorts files by name within each class, then checks
                         what fraction of adjacent pairs span the train/test
                         boundary.  High fraction → segments from the same
                         recording land in both splits → leakage.

  3. NEAR-DUPLICATE    — loads a random sample of mel .npy files and checks
                         cosine similarity between train and test samples that
                         are neighbors in sorted order.  High similarity
                         (>0.99) confirms the clips are almost identical.

  4. RECORDING-LEVEL SPLIT  — if leakage is detected, this replaces the
                         random stratified split with a session-level split:
                         consecutive files are grouped into "recordings" of
                         SESSION_SIZE clips; entire recordings go to either
                         train, val, or test — never split across boundaries.
                         A new split_indices_clean.json is written so you can
                         swap it in without changing any training script.

Usage
-----
Run all four sections.  Read the printed verdicts.
If Section 2 reports > 5% boundary pairs → use the clean split.
"""

import os, json, pathlib, random, math
import numpy as np
from sklearn.model_selection import train_test_split

# ── CONFIG ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = "/kaggle/input/datasets/mostafaehab41/machine-fault-dataset"
MODELS_DIR = "/kaggle/working"

# How many consecutive clips we consider "one recording session".
# E.g. if original recordings are ~60 s and clips are ~2.75 s → 60/2.75 ≈ 22.
# Use a round number; the exact value only affects group boundaries.
# Chronological split keeps the first 70% as train, next 15% as val, last 15% as test.
# This guarantees 0 boundary pairs for sequential recordings.

# Directory of precomputed mel .npy (needed for Section 3 only)
FEATS_MEL = "/kaggle/working/feats_mel"   # change if yours is elsewhere

LABEL_MAP = {
    ("machine1","Normal"):0, ("machine1","Abnormal"):1,
    ("machine2","Normal"):2, ("machine2","Abnormal"):3,
    ("machine3","Normal"):4, ("machine3","Abnormal"):5,
}
CLASS_NAMES = ["M1_N","M1_A","M2_N","M2_A","M3_N","M3_A"]


# ── HELPERS ────────────────────────────────────────────────────────────────────

def scan_files(root):
    paths, labels = [], []
    for f in sorted(pathlib.Path(root).rglob("*.wav")):
        lbl = LABEL_MAP.get((f.parent.parent.name, f.parent.name))
        if lbl is not None:
            paths.append(f); labels.append(lbl)
    return paths, labels

def _num_sort(f):
    """Sort by integer stem (2.wav < 10.wav < 100.wav).
    Alphabetical sort gives wrong order: 10.wav < 2.wav < 20.wav."""
    try: return int(pathlib.Path(f).stem)
    except ValueError: return pathlib.Path(f).name


def load_split(models_dir):
    p = pathlib.Path(models_dir) / "split_indices.json"
    if not p.exists():
        raise FileNotFoundError(f"split_indices.json not found at {p}. Run phase2b first.")
    return json.load(open(p))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PATH AUDIT
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1 — FILE PATH AUDIT")
print("=" * 70)

paths, labels = scan_files(ROOT_DIR)
print(f"Total files found: {len(paths)}\n")

# Show first 5 and last 5 files per class (sorted by path)
from collections import defaultdict
by_class = defaultdict(list)
for p, l in zip(paths, labels):
    by_class[l].append(p)

for cls_id in range(6):
    fps = sorted(by_class[cls_id])
    print(f"{CLASS_NAMES[cls_id]}  ({len(fps)} files)")
    for f in fps[:3]:  print(f"  first: {f}")
    print(f"  ...")
    for f in fps[-2:]: print(f"  last : {f}")
    print()

print("""
WHAT TO LOOK FOR:
  - Sequential numeric names (00000001.wav, 00000002.wav …) → likely segments
    from the same continuous recording.  Random split = temporal leakage.
  - Descriptive names with run/session IDs → safer, but still check Section 2.
  - Identical filename stems across classes → independent per-class recordings.
""")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TEMPORAL LEAKAGE CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 2 — TEMPORAL LEAKAGE (adjacent file pairs across train/test)")
print("=" * 70)

splits = load_split(MODELS_DIR)
train_set = set(splits["train"])
test_set  = set(splits["test"])
val_set   = set(splits["val"])

# Build a global index map: path → global index
path_to_idx = {p: i for i, p in enumerate(paths)}

leakage_counts = {}
for cls_id in range(6):
    # Numeric sort: 1.wav, 2.wav, ..., 10.wav not 1.wav, 10.wav, 100.wav
    fps_sorted = sorted(by_class[cls_id], key=_num_sort)
    global_ids = [path_to_idx[f] for f in fps_sorted]

    boundary_train_test = 0
    same_split_pairs    = 0
    for a, b in zip(global_ids[:-1], global_ids[1:]):
        a_split = "train" if a in train_set else ("val" if a in val_set else "test")
        b_split = "train" if b in train_set else ("val" if b in val_set else "test")
        if {a_split, b_split} in ({"train","test"}, {"val","test"}):
            boundary_train_test += 1
        else:
            same_split_pairs += 1

    total_pairs = len(global_ids) - 1
    pct = 100.0 * boundary_train_test / max(total_pairs, 1)
    leakage_counts[CLASS_NAMES[cls_id]] = (boundary_train_test, total_pairs, pct)

print(f"\n{'Class':<8}  {'Boundary pairs':>15}  {'Total pairs':>12}  {'% leaking':>10}")
print("-" * 52)
for cls, (bp, tp, pct) in leakage_counts.items():
    flag = "  ← HIGH" if pct > 5 else ""
    print(f"{cls:<8}  {bp:>15}  {tp:>12}  {pct:>9.1f}%{flag}")

avg_pct = np.mean([v[2] for v in leakage_counts.values()])
print(f"\nAverage leakage rate: {avg_pct:.1f}%")

# Expected rate under a CLEAN split: 0% (all adjacent pairs same split)
# Expected rate under a RANDOM split: ~2 * 0.15 * 0.70 ≈ 21% (train 70%, test 15%)
random_expected = 2 * 0.15 * 0.70 * 100
print(f"Random-split expected rate (theoretical): ~{random_expected:.0f}%")
print(f"Recording-level split expected rate     :  ~0%")

if avg_pct > 5:
    print(f"""
VERDICT: ⚠️  TEMPORAL LEAKAGE DETECTED  ({avg_pct:.1f}% of adjacent pairs span
  train/test).  Files that are neighbors in sorted order land in different
  splits, which means clips from the SAME recording appear in both training
  and test.  The model has seen nearly-identical examples → inflated accuracy.
  → Run Section 4 to create a recording-level split.
""")
else:
    print(f"""
VERDICT: ✓  No significant temporal leakage detected ({avg_pct:.1f}%).
  Adjacent files in sorted order mostly end up in the same split.
  The high accuracy likely reflects genuine dataset structure.
""")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NEAR-DUPLICATE SIMILARITY CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 3 — NEAR-DUPLICATE SIMILARITY (cosine sim of mel features)")
print("=" * 70)

feats_mel_path = pathlib.Path(FEATS_MEL)
if not feats_mel_path.exists() or not any(feats_mel_path.glob("*.npy")):
    print("Mel features not found — skipping Section 3.")
    print(f"(Expected at {FEATS_MEL}; run phase2b first to populate it.)\n")
else:
    # For each class, take 10 adjacent sorted pairs that cross train/test boundary
    # and 10 that stay within the same split.  Compare cosine similarity.
    def cos_sim(a, b):
        a, b = a.flatten().astype(np.float32), b.flatten().astype(np.float32)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    cross_sims, same_sims = [], []
    rng = random.Random(42)

    for cls_id in range(6):
        fps_sorted = sorted(by_class[cls_id], key=lambda f: f.name)
        global_ids = [path_to_idx[f] for f in fps_sorted]

        cross_pairs = [(a, b) for a, b in zip(global_ids[:-1], global_ids[1:])
                       if (a in train_set and b in test_set)
                       or (a in test_set  and b in train_set)]
        same_pairs  = [(a, b) for a, b in zip(global_ids[:-1], global_ids[1:])
                       if (a in train_set and b in train_set)
                       or (a in test_set  and b in test_set)]

        # Sample up to 20 of each to keep it fast
        for a, b in rng.sample(cross_pairs, min(20, len(cross_pairs))):
            try:
                fa = np.load(feats_mel_path / f"{a:06d}.npy")
                fb = np.load(feats_mel_path / f"{b:06d}.npy")
                cross_sims.append(cos_sim(fa, fb))
            except FileNotFoundError:
                pass

        for a, b in rng.sample(same_pairs, min(20, len(same_pairs))):
            try:
                fa = np.load(feats_mel_path / f"{a:06d}.npy")
                fb = np.load(feats_mel_path / f"{b:06d}.npy")
                same_sims.append(cos_sim(fa, fb))
            except FileNotFoundError:
                pass

    if cross_sims and same_sims:
        print(f"Adjacent pairs ACROSS train/test boundary  — mean cosine sim: {np.mean(cross_sims):.4f}  (n={len(cross_sims)})")
        print(f"Adjacent pairs WITHIN the same split       — mean cosine sim: {np.mean(same_sims):.4f}  (n={len(same_sims)})")
        print()
        if np.mean(cross_sims) > 0.95:
            print("VERDICT: ⚠️  Cross-boundary pairs are nearly IDENTICAL (sim > 0.95).")
            print("  The model has seen audio almost indistinguishable from the test set.")
        elif np.mean(cross_sims) > 0.80:
            print("VERDICT: ⚠️  Cross-boundary pairs are HIGHLY SIMILAR (sim > 0.80).")
            print("  Moderate leakage — recording-level split is strongly recommended.")
        else:
            print("VERDICT: ✓  Cross-boundary pairs are not unusually similar.")
    else:
        print("Not enough cross-boundary pairs found for similarity analysis.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CHRONOLOGICAL SPLIT (the fix)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 4 — CHRONOLOGICAL SPLIT  (split_indices_clean.json)")
print("=" * 70)
print("""
Files are sorted NUMERICALLY (1.wav, 2.wav, ..., 9999.wav) within each class.
The first 70% go to train, the next 15% to val, the last 15% to test.

This gives EXACTLY 0 cross-boundary adjacent pairs because consecutive files
always land in the same split — the boundary only happens once per class at
the 70% and 85% marks.

Why not random stratified?  Because adjacent files come from the same
recording session and are nearly identical.  Randomly mixing them puts
clips 100, 101, 102 in train/test/train respectively, leaking information.
The chronological split respects the recording timeline.
""")

all_train, all_val, all_test = [], [], []

for cls_id in range(6):
    # Numeric sort — critical.  Alphabetical gives 1, 10, 100, 1000, 2, 20 ...
    fps_sorted = sorted(by_class[cls_id], key=_num_sort)
    global_ids = [path_to_idx[f] for f in fps_sorted]
    n          = len(global_ids)

    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    # test gets the remainder to ensure total == n

    all_train.extend(global_ids[:n_train])
    all_val.extend(  global_ids[n_train : n_train + n_val])
    all_test.extend( global_ids[n_train + n_val :])

    print(f"  {CLASS_NAMES[cls_id]:<12}  n={n}  "
          f"train={n_train}  val={n_val}  test={n - n_train - n_val}")

clean_split = {"train": all_train, "val": all_val, "test": all_test}
out_path = pathlib.Path(MODELS_DIR) / "split_indices_clean.json"
json.dump(clean_split, open(out_path, "w"))

print(f"\nChronological split written → {out_path}")
print(f"  Train : {len(all_train):>6} samples  (first 70% of each class)")
print(f"  Val   : {len(all_val):>6} samples  (next  15%)")
print(f"  Test  : {len(all_test):>6} samples  (last  15%)")
print(f"  Total : {len(all_train)+len(all_val)+len(all_test):>6}")

# Verify — should be exactly 0 because splits are contiguous blocks
new_train_set = set(all_train)
new_test_set  = set(all_test)
new_boundary = sum(
    1 for cls_id in range(6)
    for fps_num_sorted in [sorted(by_class[cls_id], key=_num_sort)]
    for a, b in zip(
        [path_to_idx[f] for f in fps_num_sorted][:-1],
        [path_to_idx[f] for f in fps_num_sorted][1:])
    if (a in new_train_set and b in new_test_set)
    or (a in new_test_set  and b in new_train_set)
)
print(f"\nVerification — boundary pairs in clean split: {new_boundary}  (should be 0)")

print("""
HOW TO USE THE CLEAN SPLIT
──────────────────────────
All phase scripts now auto-detect split_indices_clean.json if it exists.
Just run this script first, then run the phase script — it will pick up the
clean split automatically.

No code changes needed in the phase scripts.  If you want to force the old
random split for comparison, delete split_indices_clean.json temporarily.
""")

print("=" * 70)
print("OVERALL SUMMARY")
print("=" * 70)
print(f"Files total            : {len(paths)}")
print(f"Original split leakage : {avg_pct:.1f}% of adjacent pairs cross train/test")
print(f"Clean split leakage    : {new_boundary} boundary pairs  (should be 0)")
print()
print("Next steps:")
print("  1. Run kaggle_phase2b.py — it will auto-use split_indices_clean.json")
print("  2. Compare test accuracy: if it stays near 99.9% → accuracy is real")
print("  3. If it drops significantly → original results were inflated by leakage")
print()
print("ALSO: files are ~0.11 sec but DURATION_SEC=2.75 → 96% zero-padding.")
print("  All phase scripts have been updated to DURATION_SEC=0.5 to fix this.")
print("  You MUST delete the old feats_mel/ cache before retraining (it's invalid).")
