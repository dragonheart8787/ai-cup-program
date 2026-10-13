"""Append-only experiment log (experiments/experiments.csv)."""
from __future__ import annotations

import csv
import hashlib
import logging
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

COLUMNS: tuple[str, ...] = (
    "exp_id",
    "timestamp",
    "git_commit",
    "config_hash",
    "dataset_version",
    "fold",
    "seed",
    "model",
    "checkpoint_sha256",
    "input_modality",
    "resolution",
    "postprocess",
    "localization_score",
    "text_score",
    "combined_score",
    "notes",
)


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit(repo_dir: str | Path = ".") -> str:
    """HEAD commit hash, or ``UNKNOWN`` if git is missing or there are no commits."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"
    return out.stdout.strip() or "UNKNOWN"


def new_exp_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def append_experiment(csv_path: str | Path, record: Mapping[str, Any]) -> dict[str, str]:
    """Append one row. Missing columns are written empty; unknown keys raise.

    ``exp_id`` and ``timestamp`` are filled automatically when absent.
    Returns the row as written.
    """
    unknown = set(record) - set(COLUMNS)
    if unknown:
        raise ValueError(f"Unknown experiment columns: {sorted(unknown)}")

    row = {c: "" if record.get(c) is None else str(record[c]) for c in COLUMNS}
    if not row["exp_id"]:
        row["exp_id"] = new_exp_id()
    if not row["timestamp"]:
        row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info("Logged experiment %s to %s", row["exp_id"], path)
    return row
