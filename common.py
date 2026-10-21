"""Shared detector interface and GPU helpers.

Every model is a `Detector`: `score(texts) -> np.ndarray` of P(malicious) in [0, 1] (default decision
threshold 0.5), so the evaluation code never needs model-specific logic.
"""
import gc
import random
import time

import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class Detector:
    name = "detector"

    def __init__(self):
        self.meta = {}  # n_params, load_time_s, peak_vram_gb, ...

    def score(self, texts):  # -> np.ndarray
        raise NotImplementedError

    def close(self):
        free_gpu()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def timed_load(fn):
    """Run fn() (model loading); return (result, seconds)."""
    t = time.perf_counter()
    out = fn()
    cuda_sync()
    return out, time.perf_counter() - t
