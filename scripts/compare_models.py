"""PROXY comparison of cached predictions across models / shift scenarios (NOT the official score).

For every run the panel and defect confidence thresholds are swept separately (F1@IoU .5):
  * "own-best"  = best threshold on that run's own scenario (oracle, for reference)
  * "tuned@orig" = thresholds chosen on the same model's ORIGINAL-val run, then applied to the shifted runs
Run spec: label,scenario,cache   with scenario in orig | closeup | gain | irsim

Usage: python scripts/compare_models.py --modality EO --run old,orig,<c1> --run old,closeup,<c2> ... --out table.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.dataset.closeup import closeup_plan  # noqa: E402
from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.metrics.gt import load_gt  # noqa: E402
from src.metrics.proxy import evaluate  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.utils.config import load_config  # noqa: E402

THR = [round(float(x), 2) for x in np.arange(0.05, 0.951, 0.05)]
DEFECT = {"EO": "crack", "IR": "hotspot"}


def f1_at(boxes_by_img, gts, label_is_panel: bool, t: float, iou: float = 0.5) -> float:
    sel = (lambda b: b.label == "solar_panel") if label_is_panel else (lambda b: b.label != "solar_panel")
    preds = {k: [b for b in v if sel(b) and b.conf >= t] for k, v in boxes_by_img.items()}
    g = {k: [b for b in v if sel(b)] for k, v in gts.items()}
    return evaluate(preds, g, iou)["all_f1"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", choices=("EO", "IR"), required=True)
    ap.add_argument("--run", action="append", required=True, help="label,scenario,cache")
    ap.add_argument("--out")
    args = ap.parse_args()
    mod = args.modality
    paths = DatasetPaths.from_config(load_config(resolve_repo_path("configs/default.yaml")))
    seed = int(load_config(resolve_repo_path(f"configs/yolo_{mod.lower()}.yaml"))["seed"])
    runs = []
    for spec in args.run:
        label, scen, cache = spec.split(",", 2)
        runs.append((label, scen, cache))
    res: dict = {}
    for label, scen, cache in runs:
        imgs = json.loads(Path(cache).read_text(encoding="utf-8"))["images"]
        stems = sorted(imgs)
        plan = closeup_plan(stems, seed) if scen == "closeup" else None
        gts = load_gt(paths, mod, stems, closeup=plan)
        boxes = {k: raw_to_boxes(v, "approx4", 0.05) for k, v in imgs.items()}
        r = {"panel": {t: f1_at(boxes, gts, True, t) for t in THR}, "defect": {t: f1_at(boxes, gts, False, t) for t in THR}}
        r["n_gt_defects"] = sum(1 for v in gts.values() for b in v if b.label != "solar_panel")
        res[(label, scen)] = r
        print("done", label, scen, flush=True)
    out = {}
    print(f"\n## {mod} PROXY comparison (F1@IoU0.5; not official)\n")
    print("| model | scenario | GT defects | panel own-best (conf, F1) | defect own-best (conf, F1) | panel F1 @orig-tuned conf | defect F1 @orig-tuned conf |")
    print("|---|---|---|---|---|---|---|")
    for (label, scen), r in res.items():
        o = res[(label, "orig")]
        tp = max(THR, key=lambda t: o["panel"][t]); td = max(THR, key=lambda t: o["defect"][t])
        bp = max(THR, key=lambda t: r["panel"][t]); bd = max(THR, key=lambda t: r["defect"][t])
        print(f"| {label} | {scen} | {r['n_gt_defects']} | {bp:.2f}, {r['panel'][bp]:.3f} | {bd:.2f}, {r['defect'][bd]:.3f} | "
              f"{r['panel'][tp]:.3f} (conf {tp:.2f}) | {r['defect'][td]:.3f} (conf {td:.2f}) |")
        out[f"{label}/{scen}"] = {"panel_best": [bp, r["panel"][bp]], "defect_best": [bd, r["defect"][bd]],
                                  "orig_tuned_conf": {"panel": tp, "defect": td},
                                  "panel_at_orig_tuned": r["panel"][tp], "defect_at_orig_tuned": r["defect"][td]}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
