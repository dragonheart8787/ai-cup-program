"""Training watchdog: restart an Ultralytics training run from last.pt when it stalls.

Why: on the 12 GB Windows GPU some epochs fall into GPU-memory spill (tens of seconds per step, host RAM grows). A fresh
process resuming from last.pt is fast again for a few epochs. This supervisor starts

    python scripts/train_yolo.py --config <cfg> --resume

and, if no new row is appended to <work>/runs/<tag>/results.csv for --stall-minutes, kills the whole process tree and
starts it again (up to --max-attempts). It finishes when train_yolo.py exits with code 0 (which also copies best.pt to
checkpoints/). If there is no last.pt yet the first attempt starts the run from scratch (no --resume).

Usage:
  AICUP_DATA_ROOT=.. AICUP_WORK_ROOT=.. python scripts/watchdog_resume.py --config configs/yolo_eo.yaml --fold 1 --stall-minutes 5
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.paths import repo_root, resolve_repo_path  # noqa: E402
from src.dataset.workroot import work_root  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger("watchdog")
ALLOC_CONF = "garbage_collection_threshold:0.7,max_split_size_mb:256"  # allocator fragmentation settings, not hyperparameters


def n_epochs(results_csv: Path) -> int:
    if not results_csv.is_file():
        return 0
    return max(0, len(results_csv.read_text(encoding="utf-8").splitlines()) - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="training config, e.g. configs/yolo_eo_aug.yaml")
    ap.add_argument("--fold", type=int, help="override the fold in the config")
    ap.add_argument("--stall-minutes", type=float, default=5.0)
    ap.add_argument("--initial-grace-minutes", type=float, default=25.0,
                    help="stall limit for the first epoch of every attempt (dataset cache, resume start-up, first epoch incl. validation)")
    ap.add_argument("--max-attempts", type=int, default=12)
    ap.add_argument("--poll-seconds", type=float, default=20.0)
    args = ap.parse_args()
    setup_logging("INFO")
    base = load_config(resolve_repo_path("configs/default.yaml"))
    cfg = load_config(resolve_repo_path(args.config))
    if args.fold is not None:
        cfg["fold"] = args.fold
    variant = cfg.get("variant")
    tag = f"{cfg['modality']}_fold{int(cfg['fold'])}" + (f"_{variant}" if variant else "")
    runs = work_root(base) / "runs" / tag
    results = runs / "results.csv"
    last = runs / "weights" / "last.pt"
    fold_args = ["--fold", str(args.fold)] if args.fold is not None else []
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF=ALLOC_CONF)
    log_dir = work_root(base) / "watchdog_logs"
    log_dir.mkdir(exist_ok=True)
    stall = args.stall_minutes * 60
    grace = max(stall, args.initial_grace_minutes * 60)
    for attempt in range(1, args.max_attempts + 1):
        out = (log_dir / f"{tag}_attempt{attempt}.log").open("w", encoding="utf-8")
        resume = ["--resume"] if last.is_file() else []  # first start of a new run has no last.pt yet
        proc = subprocess.Popen([sys.executable, str(repo_root() / "scripts" / "train_yolo.py"), "--config", args.config, *fold_args, *resume],
                                cwd=str(repo_root()), env=env, stdout=out, stderr=subprocess.STDOUT)
        start_n = last_n = n_epochs(results)
        last_t = time.time()
        logger.info("attempt %d started at %d finished epochs", attempt, last_n)
        killed = False
        while proc.poll() is None:
            time.sleep(args.poll_seconds)
            n = n_epochs(results)
            if n != last_n:
                last_n, last_t = n, time.time()
                logger.info("epochs finished: %d", n)
            elif time.time() - last_t > (grace if n == start_n else stall):  # first epoch of an attempt gets the grace time
                logger.warning("no new epoch for %.0f s at %d epochs; killing process tree", grace if n == start_n else stall, n)
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
                killed = True
                break
        out.close()
        if not killed and proc.returncode == 0:
            logger.info("training finished after %d attempt(s)", attempt)
            return 0
        if not killed:
            # crash (e.g. CUDA "unknown error"): retry from last.pt, but only if there is something to resume from
            logger.error("train_yolo.py exited with code %s (not a stall); see %s", proc.returncode, out.name)
            if not last.is_file():
                return int(proc.returncode or 1)
        time.sleep(20)
    logger.error("giving up after %d attempts", args.max_attempts)
    return 1


if __name__ == "__main__":
    sys.exit(main())
