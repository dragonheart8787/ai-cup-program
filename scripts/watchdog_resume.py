"""Training watchdog: restart an Ultralytics training run from last.pt when it stalls.

Why: on the 12 GB Windows GPU some epochs fall into GPU-memory spill (tens of seconds per step, host RAM grows). A fresh
process resuming from last.pt is fast again for a few epochs. This supervisor starts

    python scripts/train_yolo.py --config <cfg> --resume

and, if no new row is appended to <work>/runs/<tag>/results.csv for --stall-minutes, kills the whole process tree and
starts it again (up to --max-attempts). It finishes when train_yolo.py exits with code 0 (which also copies best.pt to
checkpoints/). The very first start needs an existing last.pt (start the run once with train_yolo.py first).

Usage:
  AICUP_DATA_ROOT=.. AICUP_WORK_ROOT=.. python scripts/watchdog_resume.py --config configs/yolo_eo_aug.yaml --stall-minutes 5
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
    ap.add_argument("--stall-minutes", type=float, default=5.0)
    ap.add_argument("--max-attempts", type=int, default=12)
    ap.add_argument("--poll-seconds", type=float, default=20.0)
    args = ap.parse_args()
    setup_logging("INFO")
    base = load_config(resolve_repo_path("configs/default.yaml"))
    cfg = load_config(resolve_repo_path(args.config))
    variant = cfg.get("variant")
    tag = f"{cfg['modality']}_fold{int(cfg['fold'])}" + (f"_{variant}" if variant else "")
    runs = work_root(base) / "runs" / tag
    results = runs / "results.csv"
    last = runs / "weights" / "last.pt"
    if not last.is_file():
        raise FileNotFoundError(f"{last} missing: start the run once with scripts/train_yolo.py before using the watchdog")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF=ALLOC_CONF)
    log_dir = work_root(base) / "watchdog_logs"
    log_dir.mkdir(exist_ok=True)
    stall = args.stall_minutes * 60
    for attempt in range(1, args.max_attempts + 1):
        out = (log_dir / f"{tag}_attempt{attempt}.log").open("w", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, str(repo_root() / "scripts" / "train_yolo.py"), "--config", args.config, "--resume"],
                                cwd=str(repo_root()), env=env, stdout=out, stderr=subprocess.STDOUT)
        last_n, last_t = n_epochs(results), time.time()
        logger.info("attempt %d started at %d finished epochs", attempt, last_n)
        killed = False
        while proc.poll() is None:
            time.sleep(args.poll_seconds)
            n = n_epochs(results)
            if n != last_n:
                last_n, last_t = n, time.time()
                logger.info("epochs finished: %d", n)
            elif time.time() - last_t > stall:
                logger.warning("no new epoch for %.0f s at %d epochs; killing process tree", stall, n)
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
                killed = True
                break
        out.close()
        if not killed and proc.returncode == 0:
            logger.info("training finished after %d attempt(s)", attempt)
            return 0
        if not killed:
            logger.error("train_yolo.py exited with code %s (not a stall); see %s", proc.returncode, out.name)
            return int(proc.returncode or 1)
        time.sleep(10)
    logger.error("giving up after %d attempts", args.max_attempts)
    return 1


if __name__ == "__main__":
    sys.exit(main())
