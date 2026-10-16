"""Defect F1 / FP per image / FN per image at fixed confidence thresholds, pooled OOF and per fold (cached predictions, no inference).

Usage: python scripts/threshold_table_oof.py --modality IR --thresholds 0.30 0.50 0.65 0.75 0.80
PROXY metric (F1@IoU .5, approx4 polygons).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.dataset.workroot import work_root  # noqa: E402
from src.metrics.gt import load_gt  # noqa: E402
from src.metrics.proxy import evaluate  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", choices=("EO", "IR"), required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", required=True)
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()
    mod = args.modality
    base = load_config(resolve_repo_path("configs/default.yaml"))
    paths = DatasetPaths.from_config(base)
    cache = work_root(base) / "cache"
    boxes, gts = {}, {}
    for k in (0, 1, 2):
        imgs = json.loads((cache / f"{mod}_fold{k}_val_{args.tag}.json").read_text(encoding="utf-8"))["images"]
        boxes[k] = {s: [b for b in raw_to_boxes(v, "approx4", 0.05) if b.label != "solar_panel"] for s, v in imgs.items()}
        gts[k] = {s: [b for b in g if b.label != "solar_panel"] for s, g in load_gt(paths, mod, sorted(imgs)).items()}
    boxes["pooled"] = {s: b for k in (0, 1, 2) for s, b in boxes[k].items()}
    gts["pooled"] = {s: g for k in (0, 1, 2) for s, g in gts[k].items()}
    print(f"## {mod} defect metrics by threshold (PROXY, F1@.5)\n")
    print("| conf | set | defect F1 | FP/img | FN/img | TP/FP/FN |\n|---|---|---|---|---|---|")
    for t in args.thresholds:
        for k in ("pooled", 0, 1, 2):
            e = evaluate({s: [b for b in v if b.conf >= t] for s, v in boxes[k].items()}, gts[k], 0.5)
            name = "pooled OOF" if k == "pooled" else f"fold {k}"
            print(f"| {t:.2f} | {name} | {e['all_f1']:.3f} | {e['all_fp_per_img']:.3f} | {e['all_fn_per_img']:.3f} | {e['all_tp']}/{e['all_fp']}/{e['all_fn']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
