"""EO brightness sensitivity: is the low test crack count a brightness-shift effect or a genuinely low crack share?

Pre-registered decision rules (written before looking at the results; thresholds 0.65 unless stated):
  control      : OOF at per-image mean gray 50 (training level) must change the image-level crack detection rate by <= 10% (relative);
                 otherwise the transformation itself is suspect and no verdict is drawn.
  SHIFT        : OOF rate at mean gray 76 drops by > 50% (relative)  AND  test crack images after darkening to 50 are > 2x the original
                 (median ratio over the three fold models).
  FEW CRACKS   : OOF rate at 76 drops by < 20%  AND  test crack images after darkening are < 1.3x the original (median ratio).
  UNCERTAIN    : everything else.
Transform: per-image gamma (limits 0.4-2.5) so that the image mean gray becomes M (own pixels only); parameters stored in the caches.
All numbers are PROXY / diagnostic.

Usage: python scripts/eo_brightness_sensitivity.py
Needs caches (scripts/predict_oof.py): EO_fold{k}_val_{base,m76,m50}, EO_fold0_test_run1, EO_fold{1,2}_test_base, EO_fold{k}_test_m50.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.dataset.workroot import work_root  # noqa: E402
from src.metrics.gt import load_gt  # noqa: E402
from src.metrics.proxy import evaluate  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.utils.config import load_config  # noqa: E402

CONF = 0.65


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["images"]


def max_crack(entry: dict) -> float:
    return max((p["conf"] for p in entry["preds"] if p["label"] == "crack"), default=0.0)


def main() -> int:
    base = load_config(resolve_repo_path("configs/default.yaml"))
    paths = DatasetPaths.from_config(base)
    cd = work_root(base) / "cache"
    print(__doc__.split("Usage:")[0])
    # ---- (a) OOF --------------------------------------------------------------
    print("## (a) OOF, 1046 EO images, each scored by the model that did not train on it\n")
    print("| setting | images with crack (max conf >= 0.65) | detection rate | rel. change vs original | crack boxes >= 0.65 | crack F1@.5 (conf 0.65) | mean gamma | clamped share |")
    print("|---|---|---|---|---|---|---|---|")
    gts = {}
    for k in (0, 1, 2):
        keys = sorted(load(cd / f"EO_fold{k}_val_base.json"))
        gts.update({s: [b for b in g if b.label == "crack"] for s, g in load_gt(paths, "EO", keys).items()})
    rates, rows = {}, {}
    for tag, label in (("base", "original"), ("m76", "mean gray -> 76 (test level)"), ("m50", "mean gray -> 50 (control)")):
        imgs = {}
        for k in (0, 1, 2):
            imgs.update(load(cd / f"EO_fold{k}_val_{tag}.json"))
        n_img = sum(1 for e in imgs.values() if max_crack(e) >= CONF)
        n_box = sum(1 for e in imgs.values() for p in e["preds"] if p["label"] == "crack" and p["conf"] >= CONF)
        preds = {s: [b for b in raw_to_boxes(e, "approx4", CONF) if b.label == "crack"] for s, e in imgs.items()}
        f1 = evaluate(preds, gts, 0.5)["all_f1"]
        tr = [e.get("transform", {}) for e in imgs.values() if "gamma" in e.get("transform", {})]
        rates[tag] = n_img / len(imgs)
        rel = (rates[tag] - rates["base"]) / rates["base"] if "base" in rates else 0.0
        gm = f"{np.mean([t['gamma'] for t in tr]):.2f}" if tr else "-"
        cl = f"{np.mean([t['clamped'] for t in tr]):.3f}" if tr else "-"
        rows[tag] = rel
        print(f"| {label} | {n_img} / {len(imgs)} | {rates[tag]:.3f} | {rel:+.1%} | {n_box} | {f1:.3f} | {gm} | {cl} |")
    print(f"\nlabels: {sum(1 for g in gts.values() if g)} / {len(gts)} images with a crack ({sum(1 for g in gts.values() if g) / len(gts):.3f})")
    ctrl_ok = abs(rows["m50"]) <= 0.10
    # ---- (b) test -------------------------------------------------------------
    print("\n## (b) test set, 696 EO images, three fold models\n")
    orig_name = {0: "EO_fold0_test_run1.json", 1: "EO_fold1_test_base.json", 2: "EO_fold2_test_base.json"}
    print("| model | setting | crack boxes >=0.65 | >=0.30 | >=0.10 | crack images >=0.65 | >=0.30 | >=0.10 |\n|---|---|---|---|---|---|---|---|")
    sets065, ratios = {}, []
    for k in (0, 1, 2):
        counts = {}
        for setting, fn in (("original", orig_name[k]), ("mean gray -> 50", f"EO_fold{k}_test_m50.json")):
            imgs = load(cd / fn)
            nb = [sum(1 for e in imgs.values() for p in e["preds"] if p["label"] == "crack" and p["conf"] >= t) for t in (0.65, 0.30, 0.10)]
            ni = [sum(1 for e in imgs.values() if max_crack(e) >= t) for t in (0.65, 0.30, 0.10)]
            counts[setting] = ni
            if setting == "original":
                sets065[k] = {s for s, e in imgs.items() if max_crack(e) >= CONF}
            print(f"| EO_fold{k} | {setting} | {nb[0]} | {nb[1]} | {nb[2]} | {ni[0]} | {ni[1]} | {ni[2]} |")
        ratios.append(counts["mean gray -> 50"][0] / counts["original"][0] if counts["original"][0] else float("inf"))
    inter = set.intersection(*sets065.values())
    union = set.union(*sets065.values())
    print(f"\ncrack images >= 0.65 on the ORIGINAL images: intersection of the three models = {len(inter)}, union = {len(union)}")
    med = float(np.median(ratios))
    print(f"darkened / original crack images (>=0.65), per model: {[round(r, 2) for r in ratios]}; median {med:.2f}")
    # ---- verdict ---------------------------------------------------------------
    drop76 = -rows["m76"]
    print("\n## verdict against the pre-registered rules\n")
    print(f"- control (OOF mean gray 50): relative change {rows['m50']:+.1%} -> {'OK (<=10%)' if ctrl_ok else 'FAILED (>10%): transformation suspect, no verdict'}")
    print(f"- OOF drop at mean gray 76: {drop76:.1%}; test median ratio after darkening: {med:.2f}")
    if not ctrl_ok:
        verdict = "NO VERDICT (control failed)"
    elif drop76 > 0.50 and med > 2:
        verdict = "SHIFT is the main cause"
    elif drop76 < 0.20 and med < 1.3:
        verdict = "FEW CRACKS in the test set"
    else:
        verdict = "UNCERTAIN"
    print(f"- verdict: **{verdict}**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
