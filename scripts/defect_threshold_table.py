"""Defect-threshold table from cached predictions (no inference): test object / image counts and val F1, FP/img, FN/img.

Usage: python scripts/defect_threshold_table.py --modality EO --val-cache <val.json> --test-cache <test.json>
PROXY metric only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.metrics.gt import load_gt  # noqa: E402
from src.metrics.proxy import evaluate  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.utils.config import load_config  # noqa: E402

THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.75)
TRAIN = {"EO": (258, 352), "IR": (428, 444)}  # (images with a defect, defect objects) of 1046 train images
N_TRAIN, N_TEST = 1046, 696


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", choices=("EO", "IR"), required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--test-cache", required=True)
    ap.add_argument("--method", default="approx4")
    args = ap.parse_args()
    mod = args.modality
    paths = DatasetPaths.from_config(load_config(resolve_repo_path("configs/default.yaml")))
    val = json.loads(Path(args.val_cache).read_text(encoding="utf-8"))["images"]
    test = json.loads(Path(args.test_cache).read_text(encoding="utf-8"))["images"]
    gts = load_gt(paths, mod, sorted(val))
    gts = {k: [b for b in v if b.label != "solar_panel"] for k, v in gts.items()}
    vb = {k: [b for b in raw_to_boxes(v, args.method, 0.05) if b.label != "solar_panel"] for k, v in val.items()}
    tb = {k: [b for b in raw_to_boxes(v, args.method, 0.05) if b.label != "solar_panel"] for k, v in test.items()}
    print(f"## {mod} defect thresholds (method {args.method}; PROXY, not official)\n")
    print("| conf | test objects | test images with >=1 | val defect F1@.5 | val FP/img | val FN/img | val TP/FP/FN |")
    print("|---|---|---|---|---|---|---|")
    for t in THRESHOLDS:
        kept = {k: [b for b in v if b.conf >= t] for k, v in tb.items()}
        n_obj = sum(len(v) for v in kept.values())
        n_img = sum(1 for v in kept.values() if v)
        e = evaluate({k: [b for b in v if b.conf >= t] for k, v in vb.items()}, gts, 0.5)
        print(f"| {t:.2f} | {n_obj} | {n_img} | {e['all_f1']:.3f} | {e['all_fp_per_img']:.3f} | {e['all_fn_per_img']:.3f} | {e['all_tp']}/{e['all_fp']}/{e['all_fn']} |")
    imgs, objs = TRAIN[mod]
    print(f"\nTrain reference ({N_TRAIN} images): {imgs} images with a defect ({imgs / N_TRAIN:.3f}), {objs} defect objects.")
    print(f"Scaled to {N_TEST} test images (comparison only, NOT a claim about the test set): "
          f"{imgs / N_TRAIN * N_TEST:.0f} images with a defect, {objs / N_TRAIN * N_TEST:.0f} objects.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
