"""EO crack-confidence diagnostics from low-threshold caches (val and test, produced with predict_oof.py --conf 0.001).

* crack box confidence quantiles
* crack boxes / images with a crack at conf >= 0.001, 0.01, 0.05, 0.1, 0.3, 0.65
* the same split into close-up images (mean brightness > 100) and the rest (test only has this split meaningfully)
* per-image MAX crack confidence distribution, val vs test
Diagnosis only (PROXY context); nothing here edits predictions.

Usage: python scripts/diagnose_eo_crack.py --val-cache <EO val c0001 cache> --test-cache <EO test c0001 cache>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.dataset.paths import resolve_repo_path  # noqa: E402
from src.utils.config import load_config  # noqa: E402

THRESHOLDS = (0.001, 0.01, 0.05, 0.1, 0.3, 0.65)
QS = (0.5, 0.75, 0.9, 0.95, 0.99)


def crack_confs(entry: dict) -> list[float]:
    return [p["conf"] for p in entry["preds"] if p["label"] == "crack"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--test-cache", required=True)
    ap.add_argument("--closeup-brightness", type=float, default=100.0)
    args = ap.parse_args()
    base = load_config(resolve_repo_path("configs/default.yaml"))
    meta = pd.read_csv(resolve_repo_path(base["audit"]["metadata_csv"]), dtype={"pair_id": str})
    eo = meta[meta.modality == "EO"].set_index(["split", "pair_id"])
    sets = {}
    for name, split, path in (("val", "train", args.val_cache), ("test", "test", args.test_cache)):
        imgs = json.loads(Path(path).read_text(encoding="utf-8"))["images"]
        bright = {k: float(eo.loc[(split, Path(k).stem), "mean_brightness"]) for k in imgs}
        sets[name] = (imgs, bright)
    print("## EO crack confidence diagnostics (caches with conf >= 0.001)\n")
    print("### box confidence quantiles\n")
    print("| set | group | images | crack boxes | q50 | q75 | q90 | q95 | q99 |\n|---|---|---|---|---|---|---|---|---|")
    groups = {}
    for name, (imgs, bright) in sets.items():
        for gname, sel in (("all", lambda b: True), ("close-up (mean>100)", lambda b: b > args.closeup_brightness),
                           ("other (mean<=100)", lambda b: b <= args.closeup_brightness)):
            keys = [k for k in imgs if sel(bright[k])]
            groups[(name, gname)] = keys
            cf = [c for k in keys for c in crack_confs(imgs[k])]
            q = " | ".join(f"{np.quantile(cf, x):.3f}" for x in QS) if cf else " | ".join("n/a" for _ in QS)
            print(f"| {name} | {gname} | {len(keys)} | {len(cf)} | {q} |")
    print("\n### crack boxes / images with >= 1 crack by threshold\n")
    head = " | ".join(f"{t:g}" for t in THRESHOLDS)
    print(f"| set | group | images | metric | {head} |\n|---|---|---|---|" + "---|" * len(THRESHOLDS))
    for (name, gname), keys in groups.items():
        imgs = sets[name][0]
        nb = [sum(1 for k in keys for c in crack_confs(imgs[k]) if c >= t) for t in THRESHOLDS]
        ni = [sum(1 for k in keys if any(c >= t for c in crack_confs(imgs[k]))) for t in THRESHOLDS]
        print(f"| {name} | {gname} | {len(keys)} | boxes | " + " | ".join(map(str, nb)) + " |")
        print(f"| {name} | {gname} | {len(keys)} | images | " + " | ".join(map(str, ni)) + " |")
    print("\n### per-image maximum crack confidence (images with no crack box at conf >= 0.001 count as 0)\n")
    print("| set | group | images | share max<0.05 | share max<0.3 | share max>=0.65 | q25 | q50 | q75 | q90 |\n|---|---|---|---|---|---|---|---|---|---|")
    for (name, gname), keys in groups.items():
        imgs = sets[name][0]
        mx = np.array([max(crack_confs(imgs[k]), default=0.0) for k in keys])
        if not len(mx):
            continue
        print(f"| {name} | {gname} | {len(mx)} | {(mx < .05).mean():.3f} | {(mx < .3).mean():.3f} | {(mx >= .65).mean():.3f} | "
              + " | ".join(f"{np.quantile(mx, q):.3f}" for q in (.25, .5, .75, .9)) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
