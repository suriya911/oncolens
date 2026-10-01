"""Reproducible seeding, including distinct per-rank seeds for distributed training."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def rank_seed(seed: int, rank: int = 0) -> int:
    """Each rank gets its own seed so data augmentation differs across GPUs."""
    return seed + rank


def seed_everything(seed: int, rank: int = 0, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch (CPU + CUDA). Returns the effective seed."""
    effective = rank_seed(seed, rank)
    os.environ["PYTHONHASHSEED"] = str(effective)
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    torch.cuda.manual_seed_all(effective)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return effective


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding: derive from the torch seed PyTorch gives each worker."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
