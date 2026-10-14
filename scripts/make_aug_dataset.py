"""Build the offline-augmented fold dataset (outside the repo; source data read-only).

Usage: AICUP_DATA_ROOT=.. AICUP_WORK_ROOT=.. python scripts/make_aug_dataset.py --modality EO --fold 0
Output: <work>/<MOD>_aug/fold<k>/{images,labels}/{train,val}, data.yaml, manifest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.dataset.augment_export import build_eo_aug, build_ir_aug  # noqa: E402
from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.dataset.workroot import dataset_dir  # noqa: E402
from src.dataset.yolo_export import build_items  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", choices=("EO", "IR"), required=True)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    setup_logging("INFO")
    base = load_config(resolve_repo_path("configs/default.yaml"))
    cfg = load_config(resolve_repo_path(f"configs/yolo_{args.modality.lower()}.yaml"))
    paths = DatasetPaths.from_config(base)
    folds = pd.read_csv(resolve_repo_path(base["folds"]["folds_csv"]))
    items = build_items(paths, folds, args.modality, args.fold)
    out = dataset_dir(base, args.modality, args.fold, "aug")
    if out.exists():
        shutil.rmtree(out)  # work tree only (never the source data)
    seed = int(cfg["seed"])
    res = build_eo_aug(items, out, seed) if args.modality == "EO" else build_ir_aug(items, out, seed)
    (out / "manifest.json").write_text(json.dumps({"seed": seed, "fold": args.fold, **res}, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
