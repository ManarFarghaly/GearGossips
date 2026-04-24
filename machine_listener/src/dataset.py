# Phase 1: PyTorch Dataset wrapper
import json
import pathlib 
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split

class MachineDataset(Dataset):
    
    def __init__(self, root_dir,preprocessor,PreprocessConfig,featurefn,split,augment=False):
        self.root_dir = pathlib.Path(root_dir) # student data dir
        self.preprocessor = preprocessor()
        self.featurefn = featurefn
        self.split = split
        self.paths, self.labels = self._load_paths_and_labels()
        self.indices = self._create_or_load_split()
    
    def _load_paths_and_labels(self):
        """
        - This function traverses the directory structure to find all .wav files, 
        extracts their labels based on their parent directories, 
        and returns two lists: one for file paths and another for corresponding labels.
        - Why:
        This is just an INDEX (a catalog) of where the data is and what it is, so we can load it later in __getitem__ without having to traverse the directory structure again.
        ("file1.wav", 0)
        ("file2.wav", 1)
        """
        paths = []
        labels = []
        mapping = {
            ("Machine 1", "Normal"): 0,
            ("Machine 1", "Abnormal"): 1,
            ("Machine 2", "Normal"): 2,
            ("Machine 2", "Abnormal"): 3,
            ("Machine 3", "Normal"): 4,
            ("Machine 3", "Abnormal"): 5,
        }

        for file in self.root_dir.rglob("*.wav"):
            machine = file.parent.parent.parent.name  # Machine 1
            state = file.parent.name                  # Normal / Abnormal

            label = mapping.get((machine, state))
            if label is not None:
                paths.append(file)
                labels.append(label)
        return paths, labels
    
    def create_or_load_split(self):
        split_file = self.root_dir / "split_indices.json"
        if split_file.exists():
            with open(split_file, "r") as f:
                split_indices = json.load(f)
        else:
            indices = list(range(len(self.paths)))
            train_indices , temp_indices , train_labels, temp_labels = train_test_split(indices,self.labels, test_size=0.3 , stratify=self.labels, random_state=42)
            val_indices, test_indices = train_test_split(temp_indices,test_size=0.5 , stratify=temp_labels, random_state=42) 

            split_indices = {
                "train" : train_indices,
                "val" : val_indices,
                "test" : test_indices
            }
            with open(split_file,"w") as f:
                json.dump(split_indices, f)
        return split_indices[self.split]
    
    def __getitem__(self,idx):
        read_idx = self.indices[idx]
        path = self.paths[read_idx]
        label = self.labels[read_idx]
        self.preprocessor.preprocess(path, mode="train" if self.split == "train" else "inference")
        feature = self.featurefn(path) 
        return torch.tensor(feature), label