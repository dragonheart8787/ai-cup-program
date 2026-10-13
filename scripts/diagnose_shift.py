"""Val vs test distribution diagnostics from cached predictions (no inference).

  a. detected solar_panel count per image (conf >= 0.25) and share of images with 0 panels
  b. confidence quantiles of panels / defects
  c. val split in brightness thirds -> proxy F1 (panel, defect) per third; test share per third
  (d. brightness-shift robustness: see --eval-fixed for any cache)

Usage: python scripts/diagnose_shift.py --modality EO --val-cache <val.json> --test-cache <test.json>
       python scripts/diagnose_shift.py --modality EO --eval-fixed <val_cache.json>
PROXY metric only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.metrics.gt import load_gt  # noqa: E402
from src.metrics.proxy import evaluate  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DEFECT = {"EO": "crack", "IR": "hotspot"}


def q(a, ps=(10, 25, 50, 75, 90)) -> str:
    a = np.asarray(a, dtype=float)
    return " / ".join(f"{np.percentile(a, p):.2f}" for p in ps) if len(a) else "n/a"


def fixed_eval(images: dict, gts: dict, mod: str, sub: dict, keys: list[str]) -> dict[str, float]:
    s = sub[mod]
    preds = {}
    for k in keys:
        bs = raw_to_boxes(images[k], s["method"], min(s["conf"], s["panel_conf"]))
        preds[k] = [b for b in bs if (b.label == "solar_panel" and b.conf >= s["panel_conf"]) or (b.label != "solar_panel" and b.conf >= s["conf"])]
    g = {k: gts[k] for k in keys}
    e = evaluate(preds, g, 0.5)
    return {"panel_f1": e["panel_f1"], "defect_f1": e["defect_f1"], "defect_tp": e["defect_tp"], "defect_fp": e["defect_fp"],
            "defect_fn": e["defect_fn"], "panel_fp_img": e["panel_fp_per_img"], "panel_fn_img": e["panel_fn_per_img"], "n": len(keys)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", choices=("EO", "IR"), required=True)
    ap.add_argument("--val-cache")
    ap.add_argument("--test-cache")
    ap.add_argument("--eval-fixed", help="val cache to evaluate at the submission thresholds")
    args = ap.parse_args()
    mod = args.modality
    base = load_config(resolve_repo_path("configs/default.yaml"))
    sub = load_config(resolve_repo_path("configs/submission.yaml"))
    paths = DatasetPaths.from_config(base)
    meta = pd.read_csv(resolve_repo_path(base["audit"]["metadata_csv"]), dtype={"pair_id": str})
    if args.eval_fixed:
        imgs = json.loads(Path(args.eval_fixed).read_text(encoding="utf-8"))["images"]
        gts = load_gt(paths, mod, sorted(imgs))
        print(json.dumps(fixed_eval(imgs, gts, mod, sub, sorted(imgs))))
        return 0
    val = json.loads(Path(args.val_cache).read_text(encoding="utf-8"))["images"]
    test = json.loads(Path(args.test_cache).read_text(encoding="utf-8"))["images"]
    d = DEFECT[mod]
    print(f"## {mod}\n\n### a. solar_panel detections per image (conf >= 0.25)\n")
    print("| set | images | mean | median | p10 | p90 | images with 0 panels |\n|---|---|---|---|---|---|---|")
    counts = {}
    for name, imgs in (("val", val), ("test", test)):
        c = np.array([sum(p["label"] == "solar_panel" and p["conf"] >= 0.25 for p in e["preds"]) for e in imgs.values()])
        counts[name] = c
        print(f"| {name} | {len(c)} | {c.mean():.1f} | {np.median(c):.0f} | {np.percentile(c, 10):.0f} | {np.percentile(c, 90):.0f} | {(c == 0).mean():.3f} |")
    print(f"\n### b. confidence quantiles (p10 / p25 / p50 / p75 / p90), conf >= 0.05 cache\n")
    print("| set | label | n | quantiles |\n|---|---|---|---|")
    for name, imgs in (("val", val), ("test", test)):
        for lab in ("solar_panel", d):
            cf = [p["conf"] for e in imgs.values() for p in e["preds"] if p["label"] == lab]
            print(f"| {name} | {lab} | {len(cf)} | {q(cf)} |")
    # c
    m = meta[meta.modality == mod].set_index("pair_id")
    vb = m.loc[sorted(val), "mean_brightness"]
    edges = np.quantile(vb, [1 / 3, 2 / 3])
    thirds = np.digitize(vb.to_numpy(), edges)
    tb = m.loc[[Path(k).stem for k in sorted(test)], "mean_brightness"].to_numpy()
    tt = np.digitize(tb, edges)
    gts = load_gt(paths, mod, sorted(val))
    print(f"\n### c. val split in brightness thirds (edges {edges[0]:.1f}, {edges[1]:.1f}) - proxy F1 at the submission thresholds\n")
    print("| third | brightness range (val) | val images | panel F1@.5 | defect F1@.5 | defect TP/FP/FN | test share in third |\n|---|---|---|---|---|---|---|")
    keys = sorted(val)
    for t in range(3):
        ks = [k for k, th in zip(keys, thirds) if th == t]
        r = fixed_eval(val, gts, mod, sub, ks)
        rng = f"{vb.to_numpy()[thirds == t].min():.0f}-{vb.to_numpy()[thirds == t].max():.0f}"
        print(f"| {t + 1} | {rng} | {len(ks)} | {r['panel_f1']:.3f} | {r['defect_f1']:.3f} | {int(r['defect_tp'])}/{int(r['defect_fp'])}/{int(r['defect_fn'])} | {(tt == t).mean():.3f} |")
    print(f"\nval brightness median {np.median(vb):.1f}; test brightness median {np.median(tb):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
