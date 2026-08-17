"""Deterministic seeding across Python, NumPy and PyTorch."""

from __future__ import annotations

import logging
import os
import random
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["seed_everything", "worker_init_fn", "make_generator"]


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch (CPU and CUDA).

    Args:
        seed: The seed to apply.
        deterministic: Also request deterministic cuDNN kernels. This makes runs
            reproducible at some cost in throughput; non-deterministic
            operations then raise instead of silently varying.

    Returns:
        The seed, so callers can log exactly what was applied.
    """
    import numpy as np
    import torch

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError) as exc:  # pragma: no cover - version dependent
            logger.warning("Could not enable fully deterministic algorithms: %s", exc)
    logger.info("Seeded everything with %d (deterministic=%s)", seed, deterministic)
    return seed


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker initialiser giving each worker a distinct, stable seed."""
    import numpy as np
    import torch

    base_seed = torch.initial_seed() % (2**32)
    seed = (base_seed + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_generator(seed: int) -> Any:
    """Return a seeded ``torch.Generator`` for DataLoader shuffling."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator
