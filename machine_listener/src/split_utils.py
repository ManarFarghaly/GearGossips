"""
split_utils.py — Single source of truth for all data splitting.

Prevents two categories of leakage:

  1. Temporal leakage  — sequential files (1.wav, 2.wav …) from the same
     recording session randomly assigned to different splits.  Adjacent clips
     share nearly-identical audio content, so the model trains on audio it
     will effectively "see again" at test time.

  2. Duplicate leakage — files whose content is byte-for-byte identical
     landing in both train and test.

Algorithm
---------
1. Sort files NUMERICALLY within each class  (1 < 2 < 10 < 100, not 1,10,100,2).
2. Detect exact duplicates via file-size pre-filter + MD5 hash.
3. Assign first 70 % → train, next 15 % → val, last 15 % → test.
4. If a file is in a known duplicate group, force it to the same split as
   the first-seen member of that group.

Public API
----------
create_clean_split(paths, labels, split_dir, ...)
    Build the split, verify it, save split_indices_clean.json, return it.
    Call this ONCE from train_phase1.py.

load_clean_split(split_dir)
    Load an existing split_indices_clean.json.
    Raises FileNotFoundError if the file is missing — run Phase 1 first.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple


# ── Sort key ─────────────────────────────────────────────────────────────────

def num_sort_key(f) -> tuple:
    """
    Numeric-first sort for files with integer stems.
    1.wav → (0, 1, '1')  <  2.wav → (0, 2)  <  10.wav → (0, 10) …
    Non-integer names fall back to alphabetical order.
    """
    p = pathlib.Path(f)
    try:
        return (0, int(p.stem), p.stem.lower())
    except ValueError:
        return (1, 0, p.stem.lower())

# ── Duplicate detection ───────────────────────────────────────────────────────

def _md5(path) -> str:
    h = hashlib.md5()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicate_groups(paths: List) -> Dict[str, List[int]]:
    """
    Find groups of files with identical content.

    Uses file-size as a cheap pre-filter: MD5 is only computed for files that
    share the same byte count (collisions are rare in real audio datasets).

    Returns
    -------
    { "size:md5": [global_idx, …] }  — only groups with ≥ 2 members.
    """
    by_size: Dict[int, List[int]] = defaultdict(list)
    for i, p in enumerate(paths):
        by_size[pathlib.Path(p).stat().st_size].append(i)

    groups: Dict[str, List[int]] = defaultdict(list)
    for size, idxs in by_size.items():
        if len(idxs) < 2:
            continue                          # unique size → definitely unique
        for i in idxs:
            key = f"{size}:{_md5(paths[i])}"
            groups[key].append(i)

    return {k: v for k, v in groups.items() if len(v) > 1}

# ── Main split builder ────────────────────────────────────────────────────────

def build_chronological_split(
    paths:            List,
    labels:           List[int],
    train_ratio:      float = 0.70,
    val_ratio:        float = 0.15,
    check_duplicates: bool  = True,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Build a chronological, duplicate-safe split.

    Parameters
    ----------
    paths            : flat list of all .wav Paths (all classes combined)
    labels           : matching integer label per path
    train_ratio      : fraction of each class to put in train
    val_ratio        : fraction for val (test = 1 − train − val)
    check_duplicates : detect and fix duplicate-content files

    Returns
    -------
    (train_indices, val_indices, test_indices) — global positions in `paths`.
    """
    # ── Build duplicate index map ─────────────────────────────────────────────
    idx_to_key: Dict[int, str] = {}
    if check_duplicates:
        dup_groups = find_duplicate_groups(paths)
        if dup_groups:
            n_dup = sum(len(v) for v in dup_groups.values())
            print(f"[split] ⚠️  {len(dup_groups)} duplicate group(s) "
                  f"({n_dup} files with identical content) — "
                  "all copies will be placed in the same split")
            for key, idxs in list(dup_groups.items())[:5]:
                names = ", ".join(pathlib.Path(paths[i]).name for i in idxs[:4])
                print(f"         {names}" + ("  …" if len(idxs) > 4 else ""))
            for key, idxs in dup_groups.items():
                for i in idxs:
                    idx_to_key[i] = key
        else:
            print("[split] ✓  No duplicate files detected")

    # ── Per-class chronological assignment ────────────────────────────────────
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_class[lbl].append(i)

    dup_assigned: Dict[str, str] = {}   # group key → "train" / "val" / "test"
    all_train: List[int] = []
    all_val:   List[int] = []
    all_test:  List[int] = []

    for cls_id in sorted(by_class):
        cls_idxs = sorted(by_class[cls_id], key=lambda i: num_sort_key(paths[i]))
        n        = len(cls_idxs)
        n_train  = int(train_ratio * n)
        n_val    = int(val_ratio   * n)

        for rank, gidx in enumerate(cls_idxs):
            if   rank < n_train:           natural = "train"
            elif rank < n_train + n_val:   natural = "val"
            else:                          natural = "test"

            key = idx_to_key.get(gidx)
            if key is not None:
                assigned = dup_assigned.setdefault(key, natural)
                if assigned != natural:
                    print(f"[split]   dup-fix: "
                          f"{pathlib.Path(paths[gidx]).name} "
                          f"moved {natural} → {assigned} "
                          "(same content as a previously-assigned file)")
            else:
                assigned = natural

            if   assigned == "train": all_train.append(gidx)
            elif assigned == "val":   all_val.append(gidx)
            else:                     all_test.append(gidx)

    return all_train, all_val, all_test


# ── Verification ──────────────────────────────────────────────────────────────

def verify_split(
    paths:         List,
    labels:        List[int],
    train_indices: List[int],
    val_indices:   List[int],
    test_indices:  List[int],
) -> None:
    """Print split sizes and count temporal boundary pairs (target: 0)."""
    train_set = set(train_indices)
    test_set  = set(test_indices)

    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_class[lbl].append(i)

    boundary = 0
    for idxs in by_class.values():
        sorted_idxs = sorted(idxs, key=lambda i: num_sort_key(paths[i]))
        for a, b in zip(sorted_idxs[:-1], sorted_idxs[1:]):
            if (a in train_set and b in test_set) or (a in test_set and b in train_set):
                boundary += 1

    print(f"[split]  Train : {len(train_indices)}")
    print(f"[split]  Val   : {len(val_indices)}")
    print(f"[split]  Test  : {len(test_indices)}")
    print(f"[split]  Boundary pairs (temporal leakage) : {boundary}  (target = 0)")
    if boundary == 0:
        print("[split] ✓  Clean — zero temporal leakage")
    else:
        print(f"[split] ⚠️  {boundary} boundary pairs detected "
              "(likely from duplicate-fix; check logs above)")


# ── Primary public API ────────────────────────────────────────────────────────

def load_clean_split(split_dir, filename: str = "split_indices_clean.json") -> dict:
    split_dir  = pathlib.Path(split_dir)
    clean_path = split_dir / filename
    if not clean_path.exists():
        raise FileNotFoundError(
            f"{filename} not found in {split_dir} — "
            "run train_phase1.py first to build it"
        )
    print(f"[split] Loaded {filename}  (chronological, no leakage)")
    return json.load(open(clean_path))


def create_clean_split(
    paths:            List,
    labels:           List[int],
    split_dir,
    train_ratio:      float = 0.70,
    val_ratio:        float = 0.15,
    check_duplicates: bool  = True,
    filename:         str   = "split_indices_clean.json",
) -> dict:
    """
    Build the split, verify it, save split_indices_clean.json, and return the dict.
    Called once from train_phase1.py.  All other phases call load_clean_split().

    paths            : full list of all .wav Paths (all classes combined)
    labels           : matching integer labels
    split_dir        : directory where the JSON file is written
    train_ratio      : fraction for train  (default 0.70)
    val_ratio        : fraction for val    (default 0.15)
    check_duplicates : run duplicate detection (default True)
    filename         : JSON filename (default "split_indices_clean.json")
    Returns: {"train": [...], "val": [...], "test": [...]}
    """
    split_dir  = pathlib.Path(split_dir)
    clean_path = split_dir / filename
    tr, va, te = build_chronological_split(
        paths, labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        check_duplicates=check_duplicates,
    )
    verify_split(paths, labels, tr, va, te)
    split_dir.mkdir(parents=True, exist_ok=True)
    result = {"train": tr, "val": va, "test": te}
    json.dump(result, open(clean_path, "w"))
    print(f"[split] Saved → {clean_path}")
    return result
