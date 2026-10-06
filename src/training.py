"""
Training Utilities Module for Legal AI Relevance Models.
Provides standard helpers for seed initialization, device configuration, and early stopping checks.
"""

import random
import torch
import numpy as np

def set_seeds(seed: int = 42):
    """Sets random seeds for Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """Detects and returns active PyTorch compute device (CUDA or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

