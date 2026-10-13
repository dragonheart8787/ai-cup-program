"""Measure peak GPU memory and speed of a short YOLO-seg training run (fresh pretrained weights; nothing is saved).

Usage: python scripts/measure_train_cost.py --config configs/yolo_eo.yaml --imgsz 1280 --batch 4 --steps 20
Run each combination in its own process so peaks are independent.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.workroot import dataset_dir, work_root  # noqa: E402
from src.dataset.paths import resolve_repo_path  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    import torch
    from ultralytics import YOLO

    base = load_config(resolve_repo_path("configs/default.yaml"))
    cfg = load_config(resolve_repo_path(args.config))
    data = dataset_dir(base, cfg["modality"], int(cfg["fold"])) / "data.yaml"
    n_train = 691
    fraction = min(1.0, args.steps * args.batch / n_train)
    times: list[float] = []

    def on_batch_end(trainer) -> None:
        torch.cuda.synchronize()
        times.append(time.perf_counter())

    model = YOLO(cfg["model"])
    model.add_callback("on_train_batch_end", on_batch_end)
    torch.cuda.reset_peak_memory_stats()
    model.train(data=str(data), epochs=1, imgsz=args.imgsz, batch=args.batch, fraction=fraction, seed=int(cfg["seed"]),
                amp=True, workers=int(cfg["workers"]), device=cfg["device"], plots=False, val=False, save=False,
                project=str(work_root(base) / "runs"), name=f"cost_{args.imgsz}_{args.batch}", exist_ok=True, verbose=False)
    steady = [b - a for a, b in zip(times[3:-1], times[4:])]  # skip warm-up steps
    print("RESULT " + json.dumps({
        "imgsz": args.imgsz, "batch": args.batch, "steps": len(times),
        "peak_alloc_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "s_per_step": round(sum(steady) / len(steady), 3) if steady else None,
        "est_epoch_min": round(sum(steady) / len(steady) * n_train / args.batch / 60, 1) if steady else None,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
