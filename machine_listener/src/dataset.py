# Phase 1: PyTorch Dataset wrapper
import torch
import pathlib 
from torch.utils.data import Dataset
from preprocess import AudioPreprocessor,PreprocessConfig
from features import calculate_mel_spectrogram

class MachineDataset(Dataset):
    def __init__(self, root_dir,preprocessor,PreprocessConfig,featurefn,split,augment=False):
        self.root_dir = pathlib.Path(root_dir) # student data dir
        self.preprocessor = preprocessor(PreprocessConfig)
        self.featurefn = featurefn
        self.split = split
    
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
                "Machine 1 Normal": 0, 
                "Machine 1 Abnormal": 1,
                "Machine 2 Normal": 2, 
                "Machine 2 Abnormal": 3, 
                "Machine 3 Normal": 4, 
                "Machine 3 Abnormal": 5
                }
        for path in self.root_dir:
            if path and path.suffix == ".wav":
                machine_state = path.parent.name.split()[-1]  # "Normal" or "Abnormal"
                machine_number = path.parent.parent.name  # "Machine 1", "Machine 2", or "Machine 3"
                label = mapping.get(f"{machine_number} {machine_state}", -1)  # Default to -1 if not found
                paths.append(path)
                labels.append(label)
        return paths, labels
    
    def __getitem__(self, idx):
        file_path = self.data_dir / f"{idx}.pt"
        data = torch.load(file_path)
        return data