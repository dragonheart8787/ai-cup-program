"""Convert colorized test IR images back to gray (inferno inverse) and compare
the brightness distribution with the training IR gray images. Read-only; no model.

Usage: AICUP_DATA_ROOT=<dir> python scripts/ir_color_report.py [--list path/to/test_ir_pseudocolor.csv]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.dataset.ir_color import colorized_to_gray  # noqa: E402
from src.dataset.paths import resolve_repo_path  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger("ir_color_report")
QS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--list", help="optional csv with a 'filename' column to cross-check against metadata.csv")
    args = ap.parse_args()
    setup_logging("INFO")
    cfg = load_config(resolve_repo_path(args.config))
    meta = pd.read_csv(resolve_repo_path(cfg["audit"]["metadata_csv"]), dtype={"pair_id": str})
    ir = meta[meta.modality == "IR"]
    col = ir[(ir.split == "test") & (ir.is_colorized.astype("boolean").fillna(False))]
    if args.list:
        listed = set(pd.read_csv(args.list).filename)
        found = {Path(p).name for p in col.image_path}
        if listed != found:
            raise RuntimeError(f"list mismatch: only in list {sorted(listed - found)[:3]}, only in metadata {sorted(found - listed)[:3]}")
        logger.info("list matches metadata (%d files)", len(found))
    logger.info("converting %d colorized test IR images", len(col))
    conv = []
    for p in col.image_path:
        with Image.open(p) as im:
            rgb = np.asarray(im.convert("RGB"))
        conv.append(colorized_to_gray(rgb))
    c_mean = np.array([g.mean() for g in conv])
    c_std = np.array([g.std() for g in conv])
    train_gray = ir[ir.split == "train"]
    test_gray = ir[(ir.split == "test") & ~(ir.is_colorized.astype("boolean").fillna(False))]

    def row(name: str, v: np.ndarray) -> str:
        q = np.quantile(v, QS)
        return f"| {name} | {len(v)} | " + " | ".join(f"{x:.1f}" for x in q) + " |"

    lines = ["| set (per-image mean gray) | n | min | q25 | median | q75 | max |", "|---|---|---|---|---|---|---|",
             row("train IR gray", train_gray.mean_brightness.to_numpy()),
             row("test IR gray", test_gray.mean_brightness.to_numpy()),
             row("test IR colorized -> inferno index", c_mean),
             "", "| set (per-image std) | n | min | q25 | median | q75 | max |", "|---|---|---|---|---|---|---|",
             row("train IR gray", train_gray.std_brightness.to_numpy()),
             row("test IR colorized -> inferno index", c_std)]
    # pooled pixel-level comparison (subsample)
    rng = np.random.default_rng(int(cfg["seed"]))
    pix_c = np.concatenate([rng.choice(g.ravel(), 2000, replace=False) for g in conv])
    pix_t = []
    for p in train_gray.image_path.sample(200, random_state=int(cfg["seed"])):
        with Image.open(p) as im:
            pix_t.append(rng.choice(np.asarray(im.convert("L")).ravel(), 2000, replace=False))
    pix_t = np.concatenate(pix_t)
    lines += ["", "| pooled pixels | q10 | q50 | q90 | q99 | share<=10 |", "|---|---|---|---|---|---|"]
    for name, v in (("train IR gray (200 imgs)", pix_t), ("colorized -> inferno index", pix_c)):
        q = np.quantile(v, [.1, .5, .9, .99])
        lines.append(f"| {name} | " + " | ".join(f"{x:.0f}" for x in q) + f" | {(v <= 10).mean():.2f} |")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
