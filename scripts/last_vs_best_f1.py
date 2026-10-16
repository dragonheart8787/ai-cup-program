"""last.pt vs best.pt on the held-out fold, in proxy F1 units (needs caches tag 'base' = best.pt and tag 'last' = last.pt).

PRE-REGISTERED PASS CONDITION (written before looking at the results):
  * judged folds: only runs that trained (almost) the full 30 epochs: IR fold 1, IR fold 2, EO fold 0, EO fold 1, EO fold 2 (28 epochs).
    IR fold 0 stopped early at epoch 20 (learning rate not decayed) -> listed but NOT judged.
  * per modality, for panel F1 and for defect F1 separately: mean over judged folds of (last - best) >= -0.010 AND no single judged
    fold with (last - best) < -0.020.
  * all modalities pass -> continue with full-data training; any modality fails -> stop and report.
Operating thresholds as in the OOF tables: EO panel 0.80 / defect 0.65; IR panel 0.85 / defect 0.75 (F1@IoU 0.5, approx4).
Selection bias: best.pt was picked on the SAME held-out fold (Ultralytics fitness per epoch); last.pt has no such bias.

Usage: python scripts/last_vs_best_f1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.dataset.workroot import work_root  # noqa: E402
from src.metrics.gt import load_gt  # noqa: E402
from src.metrics.proxy import evaluate  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.utils.config import load_config  # noqa: E402

TH = {"EO": (0.80, 0.65), "IR": (0.85, 0.75)}  # (panel, defect)
JUDGED = {"IR": (1, 2), "EO": (0, 1, 2)}


def f1s(imgs: dict, gts: dict, mod: str) -> tuple[float, float]:
    pt, dt = TH[mod]
    preds = {s: [b for b in raw_to_boxes(e, "approx4", min(pt, dt)) if (b.label == "solar_panel" and b.conf >= pt) or (b.label != "solar_panel" and b.conf >= dt)]
             for s, e in imgs.items()}
    r = evaluate(preds, gts, 0.5)
    return r["panel_f1"], r["defect_f1"]


def main() -> int:
    print(__doc__.split("Usage:")[0])
    base = load_config(resolve_repo_path("configs/default.yaml"))
    paths = DatasetPaths.from_config(base)
    cd, runs = work_root(base) / "cache", work_root(base) / "runs"
    table, pooled = [], {}
    for mod in ("IR", "EO"):
        pooled[mod] = {"base": {}, "last": {}}
        for k in range(3):
            d = pd.read_csv(runs / f"{mod}_fold{k}" / "results.csv")
            d.columns = [c.strip() for c in d.columns]
            fit = 0.1 * d["metrics/mAP50(B)"] + 0.9 * d["metrics/mAP50-95(B)"] + 0.1 * d["metrics/mAP50(M)"] + 0.9 * d["metrics/mAP50-95(M)"]
            row = {"mod": mod, "fold": k, "epochs": len(d), "best_epoch": int(fit.idxmax()) + 1}
            first = None
            for tag in ("base", "last"):
                imgs = json.loads((cd / f"{mod}_fold{k}_val_{tag}.json").read_text(encoding="utf-8"))["images"]
                pooled[mod][tag].update(imgs)
                if first is None:
                    first = load_gt(paths, mod, sorted(imgs))
                row[tag] = f1s(imgs, first, mod)
            table.append(row)
    print("| run | epochs run | best epoch | judged | panel F1 best | panel F1 last | panel diff | defect F1 best | defect F1 last | defect diff |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in table:
        j = r["fold"] in JUDGED[r["mod"]]
        pb, pl, db, dl = r["base"][0], r["last"][0], r["base"][1], r["last"][1]
        print(f"| {r['mod']} fold {r['fold']} | {r['epochs']} | {r['best_epoch']} | {'yes' if j else 'no (early stop at 20)'} | {pb:.3f} | {pl:.3f} | {pl - pb:+.3f} | {db:.3f} | {dl:.3f} | {dl - db:+.3f} |")
    print("\n| modality | metric | mean diff (judged folds) | worst single fold | condition mean >= -0.010 | condition worst >= -0.020 | pass |\n|---|---|---|---|---|---|---|")
    allpass = True
    for mod in ("IR", "EO"):
        for mi, name in ((0, "panel F1"), (1, "defect F1")):
            diffs = [r["last"][mi] - r["base"][mi] for r in table if r["mod"] == mod and r["fold"] in JUDGED[mod]]
            mean, worst = float(np.mean(diffs)), float(np.min(diffs))
            ok = mean >= -0.010 and worst >= -0.020
            allpass &= ok
            print(f"| {mod} | {name} | {mean:+.4f} | {worst:+.4f} | {mean >= -0.010} | {worst >= -0.020} | {ok} |")
    print(f"\nOVERALL: {'PASS - continue with full-data training' if allpass else 'FAIL - stop and report'}")
    print("\n### pooled OOF (1046 images), operating thresholds\n")
    print("| modality | weights | panel F1 | defect F1 |\n|---|---|---|---|")
    for mod in ("IR", "EO"):
        gts = {}
        for k in range(3):
            gts.update(load_gt(paths, mod, sorted(json.loads((cd / f"{mod}_fold{k}_val_base.json").read_text(encoding='utf-8'))["images"])))
        for tag, label in (("base", "best.pt (selected on the same fold; biased)"), ("last", "last.pt (no selection bias)")):
            p, dd = f1s(pooled[mod][tag], gts, mod)
            print(f"| {mod} | {label} | {p:.3f} | {dd:.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
