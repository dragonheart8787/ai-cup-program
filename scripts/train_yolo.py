"""Config-driven YOLO11-seg training for one modality / fold.

Usage:
  AICUP_DATA_ROOT=.. AICUP_WORK_ROOT=.. python scripts/train_yolo.py --config configs/yolo_eo.yaml --smoke
  python scripts/train_yolo.py --config configs/yolo_eo.yaml [--batch N]   # --batch only to react to OOM
Requires the exported dataset (scripts/export_yolo.py). Needs requirements-train.txt.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.workroot import dataset_dir, work_root  # noqa: E402
from src.dataset.paths import repo_root, resolve_repo_path  # noqa: E402
from src.utils.config import config_hash, load_config  # noqa: E402
from src.utils.experiments import append_experiment, file_sha256, get_git_commit  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

logger = logging.getLogger("train")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--default-config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true", help="run smoke_epochs epochs and tag the run '_smoke'")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run from <work>/runs/<tag>/weights/last.pt")
    ap.add_argument("--batch", type=int, help="override batch (OOM handling only; logged in notes)")
    args = ap.parse_args()
    setup_logging("INFO")
    base = load_config(resolve_repo_path(args.default_config))
    cfg = load_config(resolve_repo_path(args.config))
    if args.batch:
        cfg["batch"] = args.batch
    import torch
    from ultralytics import YOLO

    mod, fold = cfg["modality"], int(cfg["fold"])
    variant = cfg.get("variant")
    data = dataset_dir(base, mod, fold, variant) / "data.yaml"
    if not data.is_file():
        raise FileNotFoundError(f"{data} missing; run scripts/export_yolo.py --modality {mod} --fold {fold}")
    seed_everything(int(cfg["seed"]))
    tag = f"{mod}_fold{fold}" + (f"_{variant}" if variant else "") + ("_smoke" if args.smoke else "")
    epochs = int(cfg["smoke_epochs"] if args.smoke else cfg["epochs"])
    runs = work_root(base) / "runs"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    if args.resume:
        last = runs / tag / "weights" / "last.pt"
        if not last.is_file():
            raise FileNotFoundError(last)
        YOLO(str(last)).train(resume=True)
    else:
        model = YOLO(cfg["model"])
        model.train(
            data=str(data), epochs=epochs, imgsz=int(cfg["imgsz"]), batch=int(cfg["batch"]), seed=int(cfg["seed"]),
            deterministic=bool(cfg["deterministic"]), val=bool(cfg.get("val", True)), amp=bool(cfg["amp"]),
            workers=int(cfg["workers"]), patience=int(cfg["patience"]), max_det=int(cfg["max_det"]), device=cfg["device"],
            plots=bool(cfg["plots"]), project=str(runs), name=tag, exist_ok=True, pretrained=True,
        )
    elapsed = time.time() - t0
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_resv = torch.cuda.max_memory_reserved() / 2**30
    best = runs / tag / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    ck_dir = repo_root() / "checkpoints"
    ck_dir.mkdir(exist_ok=True)
    ck = ck_dir / f"{tag}.pt"
    shutil.copy2(best, ck)
    note = (f"train {tag}: epochs={epochs} batch={cfg['batch']} workers={cfg['workers']} elapsed_s={elapsed:.0f} "
            f"peak_alloc_GiB={peak_alloc:.2f} peak_reserved_GiB={peak_resv:.2f} model={cfg['model']} variant={variant} val={cfg.get('val', True)}")
    append_experiment(resolve_repo_path(base["paths"]["experiments_csv"]), {
        "git_commit": get_git_commit(repo_root()), "config_hash": config_hash(cfg),
        "dataset_version": f"folds.csv:{file_sha256(resolve_repo_path(base['folds']['folds_csv']))[:12]}",
        "fold": fold, "seed": cfg["seed"], "model": "yolo11s-seg", "checkpoint_sha256": file_sha256(ck),
        "input_modality": mod, "resolution": cfg["imgsz"], "notes": note,
    })
    print(note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
