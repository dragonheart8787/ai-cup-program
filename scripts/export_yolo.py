"""Export one fold of one modality to an Ultralytics segmentation dataset.

Usage: AICUP_DATA_ROOT=<dir> AICUP_WORK_ROOT=<dir> python scripts/export_yolo.py --modality EO --fold 0
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.dataset.workroot import dataset_dir  # noqa: E402
from src.dataset.yolo_export import build_items, export_items  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--modality", choices=("EO", "IR"), required=True)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    setup_logging("INFO")
    cfg = load_config(resolve_repo_path(args.config))
    paths = DatasetPaths.from_config(cfg)
    folds = pd.read_csv(resolve_repo_path(cfg["folds"]["folds_csv"]))
    out = dataset_dir(cfg, args.modality, args.fold)
    items = build_items(paths, folds, args.modality, args.fold)
    counts = export_items(items, args.modality, out)
    print(counts, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
