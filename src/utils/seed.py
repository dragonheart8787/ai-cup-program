"""Global seeding."""
from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed python, numpy and (if installed) torch.

    ``PYTHONHASHSEED`` only affects subprocesses started afterwards.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        logger.info("torch not installed; seeded python and numpy only")
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Seeded python, numpy and torch with %d", seed)
