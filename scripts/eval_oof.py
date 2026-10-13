"""Evaluate cached predictions with the PROXY metric (not the official score).

  1. compare polygon methods (minrect / approx / approx4) at a fixed confidence
  2. sweep the confidence threshold with one method

Usage: python scripts/eval_oof.py --cache <work>/cache/EO_fold0_val_base.json [--method minrect] [--compare-conf 0.25]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.metrics.gt import load_gt  # noqa: E402
from src.metrics.proxy import Box, evaluate, polygon_iou  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.postprocess.quad import METHODS  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger("eval")
THRESHOLDS = [round(x, 2) for x in np.arange(0.05, 0.801, 0.05)]


def mean_matched_iou(preds: dict[str, list[Box]], gts: dict[str, list[Box]], assoc_iou: float = 0.3) -> dict[str, float]:
    """Associate each prediction with its best same-class GT (by IoU of the method polygon) and average that IoU for matches."""
    acc: dict[str, list[float]] = {"solar_panel": [], "defect": []}
    for img, g in gts.items():
        for p in preds.get(img, []):
            best = 0.0
            for t in g:
                if t.label == p.label:
                    best = max(best, polygon_iou(p.points, t.points))
            if best >= assoc_iou:
                acc["solar_panel" if p.label == "solar_panel" else "defect"].append(best)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--compare-conf", type=float, default=0.25)
    ap.add_argument("--out", help="write the results as json")
    ap.add_argument("--default-config", default="configs/default.yaml")
    args = ap.parse_args()
    setup_logging("INFO")
    cache = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    meta, images = cache["meta"], cache["images"]
    mod = meta["modality"]
    paths = DatasetPaths.from_config(load_config(resolve_repo_path(args.default_config)))
    gts = load_gt(paths, mod, sorted(images))
    res: dict = {"meta": meta, "compare": {}, "sweep": {}}

    print(f"## {mod} fold{meta['fold']} PROXY (not official), cache {Path(args.cache).name}\n")
    print(f"### method comparison at conf >= {args.compare_conf}\n")
    print("| method | dropped | F1@.5 all | F1@.5 panel | F1@.5 defect | F1@.75 all | F1@.75 panel | F1@.75 defect | mIoU panel | mIoU defect |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in METHODS:
        drop: Counter = Counter()
        preds = {k: raw_to_boxes(v, m, args.compare_conf, drop) for k, v in images.items()}
        e5, e75 = evaluate(preds, gts, 0.5), evaluate(preds, gts, 0.75)
        mi = mean_matched_iou(preds, gts)
        res["compare"][m] = {"drop": dict(drop), "iou50": e5, "iou75": e75, "mean_iou": mi}
        print(f"| {m} | {sum(drop.values())} | {e5['all_f1']:.3f} | {e5['panel_f1']:.3f} | {e5['defect_f1']:.3f} | "
              f"{e75['all_f1']:.3f} | {e75['panel_f1']:.3f} | {e75['defect_f1']:.3f} | {mi['solar_panel']:.3f} | {mi['defect']:.3f} |")
    method = args.method or max(METHODS, key=lambda m: res["compare"][m]["iou50"]["all_f1"] + res["compare"][m]["iou75"]["all_f1"])
    res["method"] = method
    print(f"\nmethod used for the sweep: **{method}**\n")
    print("| conf | F1@.5 all | F1@.5 panel | F1@.5 defect | F1@.75 all | FP/img (all@.5) | FN/img (all@.5) | FP/img defect@.5 | FN/img defect@.5 |")
    print("|---|---|---|---|---|---|---|---|---|")
    best = (-1.0, 0.0)
    for t in THRESHOLDS:
        preds = {k: raw_to_boxes(v, method, t) for k, v in images.items()}
        e5, e75 = evaluate(preds, gts, 0.5), evaluate(preds, gts, 0.75)
        res["sweep"][str(t)] = {"iou50": e5, "iou75": e75}
        if e5["all_f1"] > best[0]:
            best = (e5["all_f1"], t)
        print(f"| {t:.2f} | {e5['all_f1']:.3f} | {e5['panel_f1']:.3f} | {e5['defect_f1']:.3f} | {e75['all_f1']:.3f} | "
              f"{e5['all_fp_per_img']:.2f} | {e5['all_fn_per_img']:.2f} | {e5['defect_fp_per_img']:.2f} | {e5['defect_fn_per_img']:.2f} |")
    res["best_conf_all_f1_iou50"] = {"conf": best[1], "f1": best[0]}
    bd = max(THRESHOLDS, key=lambda t: res["sweep"][str(t)]["iou50"]["defect_f1"])
    res["best_conf_defect_f1_iou50"] = {"conf": bd, "f1": res["sweep"][str(bd)]["iou50"]["defect_f1"]}
    print(f"\nbest conf for all-objects F1@.5: {best[1]:.2f} (F1 {best[0]:.3f}); best for defect F1@.5: {bd:.2f} (F1 {res['best_conf_defect_f1_iou50']['f1']:.3f})")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
