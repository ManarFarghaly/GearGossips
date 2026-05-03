"""
PyTorch Dataset for the Machine-Fault dataset.

__getitem__ flow:
  path  →  preprocessor.preprocess()  →  waveform (44000,)
        →  feature_fn(waveform)        →  np.ndarray  or  tuple of np.ndarray
        →  torch.tensor(...)           →  returned to DataLoader

The split file (split_indices_clean.json) must exist before instantiating this
dataset.  Run train_phase1.py once to create it via create_clean_split().
All phases share the same file because every script scans with sorted(rglob).

feature_fn can return:
    a single np.ndarray   (Phase 1: mel-spec)
    a tuple of np.ndarray (Phase 2: (mel, mfcc),  Phase 3: (mel, mfcc, stat))
"""

import pathlib
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from machine_listener.src.split_utils import load_clean_split


LABEL_MAP = {
    ("machine1", "Normal"):   0,
    ("machine1", "Abnormal"): 1,
    ("machine2", "Normal"):   2,
    ("machine2", "Abnormal"): 3,
    ("machine3", "Normal"):   4,
    ("machine3", "Abnormal"): 5,
}

# Where split_indices_clean.json lives — relative to project root
SPLIT_DIR = pathlib.Path("machine_listener/outputs/saved_models")


def scan_wav_files(root_dir: str) -> Tuple[List, List[int]]:
    """
    Walk root_dir and return (paths, labels) for all labelled .wav files.

    Expected layout: root_dir/machineX/Normal/*.wav
                     root_dir/machineX/Abnormal/*.wav

    Uses sorted(rglob) so the order is deterministic across all phases.
    """
    paths, labels = [], []
    for wav in sorted(pathlib.Path(root_dir).rglob("*.wav")):
        state   = wav.parent.name
        machine = wav.parent.parent.name
        label   = LABEL_MAP.get((machine, state))
        if label is not None:
            paths.append(wav)
            labels.append(label)
    if not paths:
        raise RuntimeError(
            f"No labelled .wav files found under {root_dir}.\n"
            "Expected structure: machineX/Normal/*.wav  and  machineX/Abnormal/*.wav"
        )
    return paths, labels


class MachineDataset(Dataset):
    """
    Dataset for the Machine-Fault dataset.

    Requires split_indices_clean.json to already exist in SPLIT_DIR.
    Run train_phase1.py once to create it.
    """

    def __init__(
        self,
        root_dir: str,
        preprocessor,
        feature_fn: Callable,
        split: str,
        augment: bool = False,
    ):
        self.preprocessor = preprocessor
        self.feature_fn   = feature_fn
        self.split        = split
        self.augment      = augment

        self.paths, self.labels = scan_wav_files(root_dir)

        all_splits   = load_clean_split(SPLIT_DIR)
        self.indices = all_splits[split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path     = self.paths[real_idx]
        label    = self.labels[real_idx]

        mode     = "train" if (self.split == "train" and self.augment) else "inference"
        waveform = self.preprocessor.preprocess(path, mode=mode)
        features = self.feature_fn(waveform)

        label_tensor = torch.tensor(label, dtype=torch.long)
        if isinstance(features, tuple):
            return tuple(torch.tensor(f, dtype=torch.float32) for f in features), label_tensor
        return torch.tensor(features, dtype=torch.float32), label_tensor
