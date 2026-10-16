"""IR pseudo-color diagnostics on cached predictions (+ one CPU measurement of the inferno LUT fit).

Pre-registered rule (written before looking at the raw-input results): the pseudo-color shift is ESTABLISHED if, at >= 2 of the
thresholds 0.75 / 0.50 / 0.30 (average over the 3 fold models), (i) the share of colorized test images with >= 1 hotspot box is more
than 30% lower (relative) than the share for gray test images when the colorized RGB is fed to the model as is, AND (ii) after mapping
the colorized images back to gray with the inferno LUT the two shares differ by less than 15% (relative). Otherwise: NOT ESTABLISHED.
Caveat built into the design: colorized and gray test images may come from different capture sessions, so a difference in box share can
also be a genuine difference in defect rate; the rule only checks the shift-and-recovery pattern.

Usage: python scripts/ir_color_diagnostics.py
Caches (scripts/predict_oof.py): IR_fold0_test_run1 / IR_fold{1,2}_test_base (converted = submission pipeline), IR_fold{k}_test_raw,
IR_fold{k}_val_{base,lutrt,infraw}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from src.dataset.ir_color import inferno_lut  # noqa: E402
from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.dataset.workroot import work_root  # noqa: E402
from src.metrics.gt import load_gt  # noqa: E402
from src.metrics.proxy import evaluate  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.utils.config import load_config  # noqa: E402

THR = (0.75, 0.50, 0.30)
PANEL_CONF, DEFECT_CONF = 0.85, 0.75


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["images"]


def hot_boxes(entry: dict, t: float) -> int:
    return sum(1 for p in entry["preds"] if p["label"] == "hotspot" and p["conf"] >= t)


def shares(imgs: dict, keys: list[str], t: float) -> tuple[float, float]:
    n = [hot_boxes(imgs[k], t) for k in keys]
    return float(np.mean([x >= 1 for x in n])), float(np.mean([x >= 3 for x in n]))


def main() -> int:
    base = load_config(resolve_repo_path("configs/default.yaml"))
    paths = DatasetPaths.from_config(base)
    cd = work_root(base) / "cache"
    meta = pd.read_csv(resolve_repo_path(base["audit"]["metadata_csv"]), dtype={"pair_id": str})
    ir_test = meta[(meta.split == "test") & (meta.modality == "IR")].copy()
    ir_test["key"] = ir_test.pair_id.map(lambda s: f"test/IR/image/{s}.jpg")
    flag = ir_test.is_colorized.astype("boolean").fillna(False).astype(bool)
    col_keys, gray_keys = ir_test.key[flag].tolist(), ir_test.key[~flag].tolist()
    thr = base["ir_color"]["colorized_threshold"]
    print("## (a) how the groups are defined\n")
    print(f"- rule: an IR image is 'colorized' if mean|R-G| + mean|G-B| of its RGB pixels > {thr} (src/dataset/ir_color.py:is_colorized). "
          f"Measured gap: gray images 0.000, colorized images >= 28.8, so the rule has no borderline cases.")
    print(f"- test IR: {len(col_keys)} colorized, {len(gray_keys)} gray; train IR: 0 colorized of 1046.")
    print("- the augmented dataset (v3 models IR_fold0_aug) DID include colorization augmentation: 35% of the training images were replaced by "
          "gray -> random contrast stretch -> inferno -> JPEG q85-95 -> inverse inferno. The models used here (IR_fold0/1/2) were trained WITHOUT it.")
    print("- the submission pipeline already maps colorized test images back to gray with the nearest inferno LUT entry ('converted'); "
          "'raw' below means the colorized RGB was fed to the model unchanged.\n")
    # ---- (b) --------------------------------------------------------------------
    print("## (b) test IR by group and model (conf thresholds for 'images with >=1 box' / '3+ boxes')\n")
    names = {"converted": {0: "IR_fold0_test_run1.json", 1: "IR_fold1_test_base.json", 2: "IR_fold2_test_base.json"},
             "raw": {k: f"IR_fold{k}_test_raw.json" for k in (0, 1, 2)}}
    print("| input | model | group | images | panels/img (conf>=0.85) median / mean | hotspot conf q25 / q50 / q75 / q90 | >=1 box @0.75 / @0.50 / @0.30 | 3+ boxes @0.75 / @0.50 / @0.30 |")
    print("|---|---|---|---|---|---|---|---|")
    table: dict = {}
    for mode in ("converted", "raw"):
        for k in (0, 1, 2):
            imgs = load(cd / names[mode][k])
            for gname, keys in (("colorized", col_keys), ("gray", gray_keys)):
                pn = [sum(1 for p in imgs[s]["preds"] if p["label"] == "solar_panel" and p["conf"] >= PANEL_CONF) for s in keys]
                cf = [p["conf"] for s in keys for p in imgs[s]["preds"] if p["label"] == "hotspot"]
                sh = [shares(imgs, keys, t) for t in THR]
                table[(mode, k, gname)] = [x[0] for x in sh]
                q = " / ".join(f"{np.quantile(cf, x):.2f}" for x in (.25, .5, .75, .9)) if cf else "n/a"
                print(f"| {mode} | fold{k} | {gname} | {len(keys)} | {np.median(pn):.0f} / {np.mean(pn):.1f} | {q} | "
                      + " / ".join(f"{x[0]:.3f}" for x in sh) + " | " + " / ".join(f"{x[1]:.3f}" for x in sh) + " |")
    oof = {}
    for k in (0, 1, 2):
        oof.update(load(cd / f"IR_fold{k}_val_base.json"))
    okeys = sorted(oof)
    sh = [shares(oof, okeys, t) for t in THR]
    pn = [sum(1 for p in oof[s]["preds"] if p["label"] == "solar_panel" and p["conf"] >= PANEL_CONF) for s in okeys]
    print(f"| OOF (gray) | all folds | gray | {len(okeys)} | {np.median(pn):.0f} / {np.mean(pn):.1f} | - | "
          + " / ".join(f"{x[0]:.3f}" for x in sh) + " | " + " / ".join(f"{x[1]:.3f}" for x in sh) + " |")
    # ---- (c) -------------------------------------------------------------------
    print("\n## (c) inferno LUT mapping\n")
    tree = cKDTree(inferno_lut().astype(np.float64))
    errs = []
    for k in col_keys:
        stem = Path(k).stem
        with Image.open(paths.image_dir("test", "IR") / f"{stem}.jpg") as im:
            px = np.asarray(im.convert("RGB")).reshape(-1, 3).astype(np.float64)
        d, _ = tree.query(px[:: 7], workers=-1)
        errs.append(float(d.mean()))
    print(f"- fit of the 200 colorized test images to the inferno LUT (mean RGB distance of a pixel to its nearest LUT entry, per image): "
          f"median {np.median(errs):.1f}, p90 {np.quantile(errs, .9):.1f}, max {np.max(errs):.1f} (RGB range 0-255; JPEG noise alone is a few units; "
          f"a large value would mean the palette is not exactly inferno)")
    lut_err, exact = [], []
    for k in (0, 1, 2):
        for e in load(cd / f"IR_fold{k}_val_lutrt.json").values():
            t = e["transform"]
            lut_err.append(t["lut_roundtrip_abs_err_mean"]); exact.append(t["lut_roundtrip_exact_share"])
    print(f"- OOF gray -> inferno -> inverse LUT (no stretch, no JPEG): mean abs gray error {np.mean(lut_err):.3f} levels, "
          f"{np.mean(exact):.1%} of the pixels come back exactly")
    gts, boxes = {}, {}
    print("\n| OOF input (pooled 1046) | panel F1@.5 (conf 0.85) | defect F1@.5 (conf 0.75) | panel FP/img | defect FP/img | defect FN/img |\n|---|---|---|---|---|---|")
    for tag, label in (("base", "original gray"), ("lutrt", "gray -> inferno -> inverse LUT"), ("infraw", "(d) gray -> inferno RGB fed raw")):
        imgs = {}
        for k in (0, 1, 2):
            imgs.update(load(cd / f"IR_fold{k}_val_{tag}.json"))
        if not gts:
            gts = load_gt(paths, "IR", sorted(imgs))
        preds = {s: [b for b in raw_to_boxes(e, "approx4", 0.05) if (b.label == "solar_panel" and b.conf >= PANEL_CONF) or (b.label != "solar_panel" and b.conf >= DEFECT_CONF)] for s, e in imgs.items()}
        r = evaluate(preds, gts, 0.5)
        print(f"| {label} | {r['panel_f1']:.3f} | {r['defect_f1']:.3f} | {r['panel_fp_per_img']:.3f} | {r['defect_fp_per_img']:.3f} | {r['defect_fn_per_img']:.3f} |")
    print("\n- effect of the inverse mapping on the 200 colorized TEST images (average over the 3 models; share of images with >=1 hotspot box):\n")
    print("| threshold | raw colorized input | converted to gray | gray test images (converted pipeline) |\n|---|---|---|---|")
    for i, t in enumerate(THR):
        raw = np.mean([table[("raw", k, "colorized")][i] for k in (0, 1, 2)])
        con = np.mean([table[("converted", k, "colorized")][i] for k in (0, 1, 2)])
        gry = np.mean([table[("converted", k, "gray")][i] for k in (0, 1, 2)])
        print(f"| {t:.2f} | {raw:.3f} | {con:.3f} | {gry:.3f} |")
    # ---- (e) -------------------------------------------------------------------
    hits = 0
    print("\n## (e) verdict against the pre-registered rule\n")
    print("| threshold | raw colorized vs gray (relative) | converted vs gray (relative) | (i) raw < -30% | (ii) converted within 15% | both |\n|---|---|---|---|---|---|")
    for i, t in enumerate(THR):
        gry = np.mean([table[("raw", k, "gray")][i] for k in (0, 1, 2)])  # gray images are unaffected by the mode
        raw = np.mean([table[("raw", k, "colorized")][i] for k in (0, 1, 2)])
        con = np.mean([table[("converted", k, "colorized")][i] for k in (0, 1, 2)])
        rr, rc = (raw - gry) / gry, (con - gry) / gry
        ok = rr < -0.30 and abs(rc) < 0.15
        hits += ok
        print(f"| {t:.2f} | {rr:+.1%} | {rc:+.1%} | {rr < -0.30} | {abs(rc) < 0.15} | {ok} |")
    print(f"\n- verdict: **{'SHIFT ESTABLISHED' if hits >= 2 else 'NOT ESTABLISHED'}** ({hits} of 3 thresholds satisfy both conditions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
