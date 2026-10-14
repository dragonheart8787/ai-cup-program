"""Measure peak GPU memory / time of the Ultralytics validation stage with an existing checkpoint (no training).

Usage: python scripts/measure_val_cost.py --config configs/yolo_eo.yaml --weights checkpoints/EO_fold0.pt --batch 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.paths import resolve_repo_path  # noqa: E402
from src.dataset.workroot import dataset_dir  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--data", help="override data.yaml")
    args = ap.parse_args()
    import torch
    from ultralytics import YOLO

    base = load_config(resolve_repo_path("configs/default.yaml"))
    cfg = load_config(resolve_repo_path(args.config))
    data = args.data or str(dataset_dir(base, cfg["modality"], int(cfg["fold"])) / "data.yaml")
    model = YOLO(args.weights)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model.val(data=data, imgsz=int(cfg["imgsz"]), batch=args.batch, conf=args.conf, max_det=int(cfg["max_det"]), half=True,
              device=cfg["device"], plots=False, verbose=False, workers=int(cfg["workers"]))
    print("RESULT " + json.dumps({"batch": args.batch, "conf": args.conf, "imgsz": cfg["imgsz"], "max_det": cfg["max_det"],
                                  "peak_alloc_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                  "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                                  "seconds": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
