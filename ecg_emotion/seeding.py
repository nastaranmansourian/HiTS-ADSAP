"""
One seeding function, used everywhere.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy, and torch (CPU + all CUDA devices)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Pass as `worker_init_fn` to DataLoader so each worker gets its own
    deterministic numpy/python RNG state, derived from torch's per-worker
    seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Build a fresh, independently-seeded generator. Always create a new
    one per DataLoader -- sharing a single generator across multiple
    DataLoaders means the second one silently inherits whatever RNG state
    the first left behind."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
