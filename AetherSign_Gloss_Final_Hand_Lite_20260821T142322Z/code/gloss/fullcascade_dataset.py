from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .fullcascade_features import fullcascade_to_model_input


class FullCascadeDataset(Dataset):
    def __init__(self, csv_path: str, target_length: int = 64, augment: bool = False) -> None:
        self.frame = pd.read_csv(csv_path)
        self.target_length = target_length
        self.augment = augment
        required = {"path", "label"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        raw = np.load(str(row["path"])).astype(np.float32)
        x = fullcascade_to_model_input(raw, self.target_length)
        if self.augment:
            x = augment_model_input(x)
        y = int(row["label"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def augment_model_input(x: np.ndarray) -> np.ndarray:
    """Small train-time perturbations that keep the board input contract unchanged."""
    x = np.asarray(x, dtype=np.float32).copy()
    valid = x[2:3] > 0.5
    if np.random.rand() < 0.6:
        noise = np.random.normal(loc=0.0, scale=0.015, size=x[0:2].shape).astype(np.float32)
        x[0:2] = x[0:2] + noise * valid
    if np.random.rand() < 0.35:
        total_frames = x.shape[2]
        drop_count = max(1, int(round(total_frames * 0.06)))
        drop_idx = np.random.choice(total_frames, size=drop_count, replace=False)
        x[:, :, drop_idx] = 0.0
    return x.astype(np.float32, copy=False)
