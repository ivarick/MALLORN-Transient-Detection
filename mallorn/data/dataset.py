"""
Data loading and dataset definitions for PyTorch models.
"""
import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset

from mallorn.config import cfg

def load_all_splits(mode):
    """Load all data splits."""
    split_lcs = {}
    for i in range(1, 21):
        split = f"Split_{i:02d}"
        path = os.path.join(cfg.BASE_DIR, split, f"{mode}_full_lightcurves.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "Time" in df.columns and "Time (MJD)" not in df.columns:
                df.rename(columns={"Time": "Time (MJD)"}, inplace=True)
            split_lcs[split] = df
    return split_lcs

def format_split_name(split_val):
    """Normalize split name format."""
    if isinstance(split_val, str):
        match = re.search(r'(\d+)', split_val)
        if match:
            return f"Split_{int(match.group(1)):02d}"
    return f"Split_{int(split_val):02d}"

class NNDataset(Dataset):
    def __init__(self, X, y=None, swap_noise_rate=0.0):
        """
        # Swap Noise regularization: randomly swap feature values between samples
        # to encourage the model to learn robust, positional invariance.
        to prevent overfitting by randomly swapping feature values.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.swap_noise_rate = swap_noise_rate
  
    def __len__(self):
        return len(self.X)
  
    def __getitem__(self, idx):
        x = self.X[idx].clone()
        
        # Apply Swap Noise during training only
        if self.swap_noise_rate > 0 and self.y is not None:
            mask = torch.rand(x.shape) < self.swap_noise_rate
            random_idx = torch.randint(0, len(self.X), (1,)).item()
            x[mask] = self.X[random_idx][mask]

        if self.y is not None:
            return x, self.y[idx]
        return x
