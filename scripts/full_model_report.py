"""Reports for a full-data (final) IR model. Numbers only; nothing here selects weights or thresholds.

(a) sanity: panel / defect recall of the model on its OWN training images (expected close to 1.0; NOT a generalisation metric)
(b) test statistics of the full-data model next to the three fold models (same table format as scripts/ir_color_diagnostics.py)
(c) agreement of the IR hotspot boxes of two submission files (IoU > 0.5, one-to-one), both directions

Usage: python scripts/full_model_report.py --full-train-cache <IR_fold0_train_all_full.json> --full-test-cache <IR_fold0_test_full.json> \
           [--base-sub v2.json --new-sub v6.json]
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
from src.dataset.workroot import work_root  # noqa: E402
from src.metrics.gt import load_gt  # noqa: E402
from src.metrics.proxy import Box, evaluate, match_image  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.utils.config import load_config  # noqa: E402

THR = (0.75, 0.50, 0.30)
PANEL_CONF, DEFECT_CONF = 0.85, 0.75


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))["images"]


def hot(entry: dict, t: float) -> int:
    return sum(1 for p in entry["preds"] if p["label"] == "hotspot" and p["conf"] >= t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-train-cache", required=True)
    ap.add_argument("--full-test-cache", required=True)
    ap.add_argument("--base-sub")
    ap.add_argument("--new-sub")
    args = ap.parse_args()
    base = load_config(resolve_repo_path("configs/default.yaml"))
    paths = DatasetPaths.from_config(base)
    cd = work_root(base) / "cache"
    # (a)
    imgs = load(Path(args.full_train_cache))
    gts = load_gt(paths, "IR", sorted(imgs))
    preds = {s: [b for b in raw_to_boxes(e, "approx4", 0.05) if (b.label == "solar_panel" and b.conf >= PANEL_CONF) or (b.label != "solar_panel" and b.conf >= DEFECT_CONF)] for s, e in imgs.items()}
    r = evaluate(preds, gts, 0.5)
    print("## (a) sanity check on the 1046 TRAINING images (the model was trained on them; not a generalisation metric)\n")
    print("| label | GT objects | recall @IoU .5 | precision @IoU .5 | F1 |\n|---|---|---|---|---|")
    for g, name in (("panel", "solar_panel"), ("defect", "hotspot")):
        print(f"| {name} | {r[g + '_tp'] + r[g + '_fn']} | {r[g + '_recall']:.3f} | {r[g + '_precision']:.3f} | {r[g + '_f1']:.3f} |")
    # (b)
    meta = pd.read_csv(resolve_repo_path(base["audit"]["metadata_csv"]), dtype={"pair_id": str})
    t = meta[(meta.split == "test") & (meta.modality == "IR")].copy()
    t["key"] = t.pair_id.map(lambda s: f"test/IR/image/{s}.jpg")
    flag = t.is_colorized.astype("boolean").fillna(False).astype(bool)
    groups = {"colorized": t.key[flag].tolist(), "gray": t.key[~flag].tolist()}
    models = {"IR_fold0": "IR_fold0_test_run1.json", "IR_fold1": "IR_fold1_test_base.json", "IR_fold2": "IR_fold2_test_base.json"}
    print("\n## (b) test IR statistics (colorized test images mapped to gray by the inferno LUT before inference)\n")
    print("| model | group | images | panels/img (conf>=0.85) median / mean | >=1 box @0.75 / @0.50 / @0.30 | 3+ boxes @0.75 / @0.50 / @0.30 |\n|---|---|---|---|---|---|")
    rows = [(n, load(cd / f)) for n, f in models.items()] + [("IR_full", load(Path(args.full_test_cache)))]
    for name, im in rows:
        for gname, keys in groups.items():
            pn = [sum(1 for p in im[k]["preds"] if p["label"] == "solar_panel" and p["conf"] >= PANEL_CONF) for k in keys]
            s1 = [float(np.mean([hot(im[k], th) >= 1 for k in keys])) for th in THR]
            s3 = [float(np.mean([hot(im[k], th) >= 3 for k in keys])) for th in THR]
            print(f"| {name} | {gname} | {len(keys)} | {np.median(pn):.0f} / {np.mean(pn):.1f} | " + " / ".join(f"{x:.3f}" for x in s1) + " | " + " / ".join(f"{x:.3f}" for x in s3) + " |")
    # (c)
    if args.base_sub and args.new_sub:
        a = json.loads(Path(args.base_sub).read_text(encoding="utf-8"))
        b = json.loads(Path(args.new_sub).read_text(encoding="utf-8"))
        tp = fp = fn = 0
        for k in a:
            if "/IR/" not in k:
                continue
            old = [Box("hotspot", np.asarray(o["points"], float)) for o in a[k] if o["label"] == "hotspot"]
            new = [Box("hotspot", np.asarray(o["points"], float)) for o in b[k] if o["label"] == "hotspot"]
            c = match_image(new, old, 0.5)
            tp += c.tp; fp += c.fp; fn += c.fn
        print("\n## (c) IR hotspot agreement between the two submissions (IoU > 0.5, one-to-one)\n")
        print(f"- base boxes {tp + fn}, new boxes {tp + fp}, matched pairs {tp}")
        print(f"- share of base boxes found in new: {tp / (tp + fn):.3f}; share of new boxes found in base: {tp / (tp + fp):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
