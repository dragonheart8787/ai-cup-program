"""Location of the scratch tree (outside the repo and the read-only data)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

ENV_WORK_ROOT = "AICUP_WORK_ROOT"


def work_root(cfg: Mapping[str, Any], env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    raw = env.get(ENV_WORK_ROOT) or cfg.get("paths", {}).get("work_root")
    if not raw:
        raise ValueError(f"work root is not set: define {ENV_WORK_ROOT} or paths.work_root")
    root = Path(raw)
    root.mkdir(parents=True, exist_ok=True)
    return root


def dataset_dir(cfg: Mapping[str, Any], modality: str, fold: int) -> Path:
    return work_root(cfg) / modality / f"fold{fold}"
