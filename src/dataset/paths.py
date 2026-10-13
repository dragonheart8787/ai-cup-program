"""Resolve dataset locations from config / environment. Data is only ever read."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

ENV_DATA_ROOT = "AICUP_DATA_ROOT"
MODALITIES = ("EO", "IR")
SPLITS = ("train", "test")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(p: str | Path) -> Path:
    """Repo-relative paths (outputs, data/metadata.csv ...) resolve against the repo root."""
    p = Path(p)
    return p if p.is_absolute() else repo_root() / p


def _join(root: Path, rel: str | Path) -> Path:
    rel = Path(rel)
    return rel if rel.is_absolute() else root / rel


@dataclass(frozen=True)
class DatasetPaths:
    data_root: Path
    train_root: Path
    test_root: Path
    make_submission_py: Path
    sample_submission_json: Path

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], env: Mapping[str, str] | None = None) -> "DatasetPaths":
        env = os.environ if env is None else env
        ds = cfg.get("dataset")
        if not isinstance(ds, Mapping):
            raise KeyError("config has no 'dataset' section")
        raw = env.get(ENV_DATA_ROOT) or ds.get("data_root")
        if not raw:
            raise ValueError(f"data root is not set: define {ENV_DATA_ROOT} or dataset.data_root in the config")
        root = Path(raw)
        if not root.is_dir():
            raise FileNotFoundError(f"data root does not exist: {root}")
        paths = cls(
            data_root=root,
            train_root=_join(root, ds["train_root"]),
            test_root=_join(root, ds["test_root"]),
            make_submission_py=_join(root, ds["make_submission_py"]),
            sample_submission_json=_join(root, ds["sample_submission_json"]),
        )
        for name in ("train_root", "test_root", "make_submission_py", "sample_submission_json"):
            if not getattr(paths, name).exists():
                raise FileNotFoundError(f"dataset.{name} does not exist: {getattr(paths, name)}")
        logger.info("data_root=%s", root)
        return paths

    def split_root(self, split: str) -> Path:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        return self.train_root if split == "train" else self.test_root

    def image_dir(self, split: str, modality: str) -> Path:
        _check_modality(modality)
        return self.split_root(split) / modality / "image"

    def label_dir(self, modality: str) -> Path:
        _check_modality(modality)
        return self.train_root / modality / "label"

    def list_stems(self, split: str, modality: str) -> list[str]:
        d = self.image_dir(split, modality)
        if not d.is_dir():
            raise FileNotFoundError(d)
        return sorted(p.stem for p in d.glob("*.jpg"))


def _check_modality(modality: str) -> None:
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}; expected one of {MODALITIES}")
