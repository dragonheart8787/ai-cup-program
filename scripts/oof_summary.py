"""Out-of-fold summary over folds 0-2 from cached val predictions (PROXY metric, F1@IoU .5 unless stated).

For every fold and the pooled OOF set: panel F1, defect F1, FP/img, FN/img at the given thresholds; max-min spread over folds;
defect-confidence sweep 0.05-0.95 on the pooled OOF with the best point; (IR) distribution of defect boxes per image vs labels.

Usage: python scripts/oof_summary.py --modality IR --panel-conf 0.85 --defect-conf 0.75 --tag base
Caches expected: <work>/cache/<MOD>_fold<k>_val_<tag>.json for k in 0,1,2 (scripts/predict_oof.py --fold k --tag <tag>).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.dataset.workroot import work_root  # noqa: E402
from src.metrics.gt import load_gt  # noqa: E402
from src.metrics.proxy import evaluate  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.utils.config import load_config  # noqa: E402

THR = [round(float(x), 2) for x in np.arange(0.05, 0.951, 0.05)]


def at(boxes, gts, panel_t, defect_t, iou=0.5):
    preds = {k: [b for b in v if (b.label == "solar_panel" and b.conf >= panel_t) or (b.label != "solar_panel" and b.conf >= defect_t)]
             for k, v in boxes.items()}
    return evaluate(preds, gts, iou)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", choices=("EO", "IR"), required=True)
    ap.add_argument("--panel-conf", type=float, required=True)
    ap.add_argument("--defect-conf", type=float, required=True)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out")
    args = ap.parse_args()
    mod = args.modality
    base = load_config(resolve_repo_path("configs/default.yaml"))
    paths = DatasetPaths.from_config(base)
    cache_dir = work_root(base) / "cache"
    boxes_f, gts_f = {}, {}
    for k in (0, 1, 2):
        imgs = json.loads((cache_dir / f"{mod}_fold{k}_val_{args.tag}.json").read_text(encoding="utf-8"))["images"]
        boxes_f[k] = {s: raw_to_boxes(v, "approx4", 0.05) for s, v in imgs.items()}
        gts_f[k] = load_gt(paths, mod, sorted(imgs))
    pooled_b = {s: b for k in boxes_f for s, b in boxes_f[k].items()}
    pooled_g = {s: g for k in gts_f for s, g in gts_f[k].items()}
    if len(pooled_b) != 1046 or set(pooled_b) != set(pooled_g):
        raise RuntimeError(f"expected 1046 distinct OOF images, got {len(pooled_b)}")
    res: dict = {"modality": mod, "panel_conf": args.panel_conf, "defect_conf": args.defect_conf}
    print(f"## {mod} OOF summary (PROXY, F1@.5; thresholds panel {args.panel_conf}, defect {args.defect_conf})\n")
    print("| set | images | panel F1 | defect F1 | panel FP/img | panel FN/img | defect FP/img | defect FN/img |\n|---|---|---|---|---|---|---|---|")
    rows = {}
    for name, b, g in [(f"fold {k}", boxes_f[k], gts_f[k]) for k in (0, 1, 2)] + [("pooled OOF", pooled_b, pooled_g)]:
        e = at(b, g, args.panel_conf, args.defect_conf)
        rows[name] = e
        print(f"| {name} | {len(g)} | {e['panel_f1']:.3f} | {e['defect_f1']:.3f} | {e['panel_fp_per_img']:.3f} | {e['panel_fn_per_img']:.3f} | "
              f"{e['defect_fp_per_img']:.3f} | {e['defect_fn_per_img']:.3f} |")
    folds = [rows[f"fold {k}"] for k in (0, 1, 2)]
    print("\n| metric | min over folds | max over folds | max - min |\n|---|---|---|---|")
    for key in ("panel_f1", "defect_f1"):
        v = [f[key] for f in folds]
        print(f"| {key} | {min(v):.3f} | {max(v):.3f} | {max(v) - min(v):.3f} |")
    res["rows"] = {k: {x: float(y) for x, y in v.items()} for k, v in rows.items()}
    print("\n### defect-confidence sweep on pooled OOF (panel threshold fixed)\n")
    print("| conf | defect F1 | defect FP/img | defect FN/img | TP/FP/FN |\n|---|---|---|---|---|")
    best = (-1.0, 0.0)
    sweep = {}
    for t in THR:
        e = at(pooled_b, pooled_g, args.panel_conf, t)
        sweep[t] = e["defect_f1"]
        if e["defect_f1"] > best[0]:
            best = (e["defect_f1"], t)
        print(f"| {t:.2f} | {e['defect_f1']:.3f} | {e['defect_fp_per_img']:.3f} | {e['defect_fn_per_img']:.3f} | {e['defect_tp']}/{e['defect_fp']}/{e['defect_fn']} |")
    print(f"\nbest defect conf on pooled OOF: {best[1]:.2f} (F1 {best[0]:.3f})")
    res["sweep"] = {str(k): v for k, v in sweep.items()}
    res["best_defect"] = {"conf": best[1], "f1": best[0]}
    for t in sorted({args.defect_conf, best[1]}):
        pred_n = Counter(min(sum(1 for b in v if b.label != "solar_panel" and b.conf >= t), 3) for v in pooled_b.values())
        gt_n = Counter(min(sum(1 for b in v if b.label != "solar_panel"), 3) for v in pooled_g.values())
        print(f"\n### defect boxes per image, pooled OOF, conf >= {t:.2f} vs labels\n")
        print("| boxes per image | predicted images | label images |\n|---|---|---|")
        for n in (0, 1, 2, 3):
            print(f"| {'3+' if n == 3 else n} | {pred_n.get(n, 0)} | {gt_n.get(n, 0)} |")
        res[f"box_count_conf{t:.2f}"] = {"pred": dict(pred_n), "gt": dict(gt_n)}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
