"""
PyTorch Dataset — wraps the preprocessor and any feature function.

__getitem__ flow:
  path  →  preprocessor.preprocess()  →  waveform (44000,)
        →  feature_fn(waveform)        →  np.ndarray  or  tuple of np.ndarray
        →  torch.tensor(...)           →  returned to DataLoader

feature_fn can return:
    • a single np.ndarray   (Phase 1: mel-spec,  Phase 3: already merged inside model)
    • a tuple of np.ndarray (Phase 2: (mel, mfcc),  Phase 3 dataset: (mel, mfcc, stat))
"""

import json
import pathlib
from typing import Callable, Optional
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class MachineDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        preprocessor,           
        feature_fn: Callable,   # fn(waveform: np.ndarray) -> np.ndarray | tuple
        split: str,             # "train" | "val" | "test"
        augment: bool = False,
    ):
        self.root_dir    = pathlib.Path(root_dir)
        self.preprocessor = preprocessor   # do not call it again with () it is already an instance of AudioPreprocessor
        self.feature_fn  = feature_fn
        self.split       = split
        self.augment     = augment
        self.paths, self.labels = self._load_paths_and_labels()
        self.indices  = self._create_or_load_split()  

    def _load_paths_and_labels(self):
        """
        Walk the directory tree and build two parallel lists:
        paths[i]  → pathlib.Path to a .wav file
        labels[i] → integer 0-5

        Directory structure expected:
        <root_dir>/
            Machine 1 / machine_data / Normal   / *.wav   → label 0
        """
        # Actual dataset structure:
        #   machine-fault-dataset/
        #     machine1/Normal/*.wav   → label 0
        #     machine1/Abnormal/*.wav → label 1
        #     machine2/...            → label 2, 3
        #     machine3/...            → label 4, 5
        mapping = {
            ("machine1", "Normal"):   0,
            ("machine1", "Abnormal"): 1,
            ("machine2", "Normal"):   2,
            ("machine2", "Abnormal"): 3,
            ("machine3", "Normal"):   4,
            ("machine3", "Abnormal"): 5,
        }

        paths, labels = [], []
        for wav_file in self.root_dir.rglob("*.wav"):
            # wav_file = …/machine-fault-dataset/machine1/Normal/001.wav
            state   = wav_file.parent.name        # "Normal" or "Abnormal"
            machine = wav_file.parent.parent.name # "machine1", "machine2", "machine3"
            label   = mapping.get((machine, state))
            if label is not None:
                paths.append(wav_file)
                labels.append(label)

        if not paths:
            raise RuntimeError(
                f"No .wav files found under {self.root_dir}. "
                "Check that the path points to the 'Students' folder."
            )
        return paths, labels

    def _create_or_load_split(self):
        """
        70 / 15 / 15 stratified split.
        Saves indices to split_indices.json next to root_dir so every phase
        uses the SAME split.  Falls back to cwd if root_dir is read-only
        (e.g. /kaggle/input).
        """
        try:
            split_file = self.root_dir.parent / "split_indices.json"
            split_file.touch(exist_ok=True)   # test writeability
        except (OSError, PermissionError):
            split_file = pathlib.Path("split_indices.json")  # cwd fallback

        if split_file.exists():
            with open(split_file) as f:
                all_splits = json.load(f)
        else:
            indices = list(range(len(self.paths)))

            train_idx, temp_idx, _, temp_labels = train_test_split(
                indices, self.labels,
                test_size=0.30, stratify=self.labels, random_state=42,
            )
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=0.50, stratify=temp_labels, random_state=42,
            )

            all_splits = {"train": train_idx, "val": val_idx, "test": test_idx}
            with open(split_file, "w") as f:
                json.dump(all_splits, f)

        return all_splits[self.split]

    def __len__(self):                         # FIX: was missing entirely
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path     = self.paths[real_idx]
        label    = self.labels[real_idx]
        mode = "train" if (self.split == "train" and self.augment) else "inference"
        waveform = self.preprocessor.preprocess(path, mode=mode)  # np.ndarray (44000,)
        features = self.feature_fn(waveform)                       # np.ndarray or tuple
        # here we wrap label(plain int) in tensor so DataLoader can stack a batch
        label_tensor = torch.tensor(label, dtype=torch.long)
        if isinstance(features, tuple):
            # In update 2 and 3: feature_fn returns (mel, mfcc) or (mel, mfcc, stat)
            return tuple(torch.tensor(f, dtype=torch.float32) for f in features), label_tensor
        return torch.tensor(features, dtype=torch.float32), label_tensor
