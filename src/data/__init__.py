"""Data package."""

from src.data.dataset import VideoDataset
from src.data.splits import build_train_val_datasets

__all__ = ["VideoDataset", "build_train_val_datasets"]
