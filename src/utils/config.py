"""YAML config loading and a stable config hash."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict. An empty file yields ``{}``."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top level of {path} must be a mapping, got {type(cfg).__name__}")
    logger.info("Loaded config %s", path)
    return cfg


def config_hash(cfg: dict[str, Any], length: int = 12) -> str:
    """SHA-256 of the canonical JSON form of ``cfg`` (key order does not matter)."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]
