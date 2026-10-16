"""Calibrate the TIoU definition against the one public-leaderboard observation (v2): reads cached OOF predictions only.

Observed (v2): TIoU_solar_panel 0.88761, TIoU_crack 0.28506, TIoU_hotspot 0.03647.
Candidate definitions for one class (all with the v2 operating thresholds; polygons are the approx4 polygons of the cache):
  (a) F1@IoU0.5
  (b) mean over IoU thresholds 0.50:0.05:0.95 of F1
  (c) IoU-weighted F1 = 2*sum(IoU of matched pairs) / (N_pred + N_gt), greedy one-to-one matching with IoU >= 0.5; and a variant that matches at IoU > 0
  (d) every definition aggregated two ways: pooled over all images, or computed per image and averaged
EO and IR are computed separately and also pooled with the v2 test object counts (EO 34,155, IR 25,174 panels) as weights.
Nothing here tunes a model, threshold or prediction: it only picks the definition that reproduces the observed number best.

Usage: python scripts/tiou_calibration.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.dataset.workroot import work_root  # noqa: E402
from src.metrics.gt import load_gt  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.utils.config import load_config  # noqa: E402

OBS = {"solar_panel": 0.88761, "crack": 0.28506, "hotspot": 0.03647}
W = {"EO": 34155, "IR": 25174}  # v2 panel object counts
OPS = {"EO": {"solar_panel": 0.80, "crack": 0.65}, "IR": {"solar_panel": 0.85, "hotspot": 0.75}}
THS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
EPS = 1e-9


def poly(p: np.ndarray) -> Polygon:
    g = Polygon(p)
    return g if g.is_valid else g.buffer(0)


def pair_ious(preds, gts):
    """per prediction (in confidence order): list of (gt index, IoU) for overlapping GT polygons."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i].conf)
    pp = [poly(preds[i].points) for i in order]
    gp = [poly(g.points) for g in gts]
    gb = np.array([g.bounds for g in gp]).reshape(-1, 4)
    out = []
    for p in pp:
        if not len(gp) or p.is_empty:
            out.append([])
            continue
        x0, y0, x1, y1 = p.bounds
        near = np.nonzero((gb[:, 0] < x1) & (gb[:, 2] > x0) & (gb[:, 1] < y1) & (gb[:, 3] > y0))[0]
        row = []
        for j in near:
            if gp[j].is_empty:
                continue
            inter = p.intersection(gp[j]).area
            u = p.area + gp[j].area - inter
            if inter > 0 and u > 0:
                row.append((int(j), inter / u))
        out.append(row)
    return out


def greedy(pairs, thr):
    """same greedy rule as the proxy metric: each prediction (by confidence) takes its best unused GT if IoU >= thr."""
    used, ious = set(), []
    for row in pairs:
        best, bj = 0.0, -1
        for j, v in row:
            if j not in used and v > best:
                best, bj = v, j
        if bj >= 0 and best >= thr:
            used.add(bj)
            ious.append(best)
    return ious


def collect(mod: str, label: str, conf: float, paths, cd):
    imgs = {}
    for k in range(3):
        imgs.update(json.loads((cd / f"{mod}_fold{k}_val_base.json").read_text(encoding="utf-8"))["images"])
    gts = load_gt(paths, mod, sorted(imgs))
    rows = []
    for s in sorted(imgs):
        preds = [b for b in raw_to_boxes(imgs[s], "approx4", conf) if b.label == label and b.conf >= conf]
        g = [b for b in gts[s] if b.label == label]
        rows.append((len(preds), len(g), pair_ious(preds, g)))
    return rows


def definitions(rows) -> dict[str, float]:
    """each definition under both aggregations: pooled over images, and per image then averaged."""
    out = {}
    cache = {t: [(np_, ng, greedy(pr, t)) for np_, ng, pr in rows] for t in THS + [EPS]}

    def f1(tp, np_, ng):
        return 2 * tp / (np_ + ng) if np_ + ng else None

    # F1 at one threshold
    def f1_at(t, mode):
        data = cache[t]
        if mode == "pooled":
            return f1(sum(len(m) for _, _, m in data), sum(n for n, _, _ in data), sum(g for _, g, _ in data))
        v = [f1(len(m), n, g) for n, g, m in data]
        v = [x for x in v if x is not None]
        return float(np.mean(v))

    def iouw(t, mode):
        data = cache[t]
        if mode == "pooled":
            return 2 * sum(sum(m) for _, _, m in data) / (sum(n for n, _, _ in data) + sum(g for _, g, _ in data))
        v = [2 * sum(m) / (n + g) for n, g, m in data if n + g]
        return float(np.mean(v))

    for mode in ("pooled", "per-image"):
        out[f"(a) F1@IoU0.5 [{mode}]"] = f1_at(0.5, mode)
        out[f"(b) mean F1 over IoU 0.50:0.05:0.95 [{mode}]"] = float(np.mean([f1_at(t, mode) for t in THS]))
        out[f"(c) IoU-weighted F1, match IoU>=0.5 [{mode}]"] = iouw(0.5, mode)
        out[f"(c') IoU-weighted F1, match IoU>0 [{mode}]"] = iouw(EPS, mode)
    return out


def main() -> int:
    base = load_config(resolve_repo_path("configs/default.yaml"))
    paths = DatasetPaths.from_config(base)
    cd = work_root(base) / "cache"
    res: dict = {}
    print("## solar_panel: candidate TIoU definitions on the pooled OOF (1046 images per modality, v2 operating thresholds)\n")
    for mod in ("EO", "IR"):
        res[(mod, "solar_panel")] = definitions(collect(mod, "solar_panel", OPS[mod]["solar_panel"], paths, cd))
    names = list(res[("EO", "solar_panel")])
    print("| definition | EO | IR | EO+IR weighted (34,155 / 25,174) | |diff to 0.88761| |\n|---|---|---|---|---|")
    ranked = []
    for n in names:
        e, i = res[("EO", "solar_panel")][n], res[("IR", "solar_panel")][n]
        pooled = (W["EO"] * e + W["IR"] * i) / (W["EO"] + W["IR"])
        ranked.append((abs(pooled - OBS["solar_panel"]), n, e, i, pooled))
    for d, n, e, i, p in sorted(ranked):
        print(f"| {n} | {e:.4f} | {i:.4f} | {p:.4f} | {d:.4f} |")
    best = sorted(ranked)[0][1]
    print(f"\nclosest definition to 0.88761: **{best}**\n")
    print("## crack / hotspot with the closest definition (OOF) next to the leaderboard values\n")
    print("| class | modality | operating conf | GT objects (OOF) | OOF value | leaderboard (test, v2) |\n|---|---|---|---|---|---|")
    for mod, label in (("EO", "crack"), ("IR", "hotspot")):
        rows = collect(mod, label, OPS[mod][label], paths, cd)
        v = definitions(rows)[best]
        # also show every definition for transparency
        print(f"| {label} | {mod} | {OPS[mod][label]} | {sum(g for _, g, _ in rows)} | {v:.4f} | {OBS[label]} |")
        res[(mod, label)] = definitions(rows)
    print("\nall definitions for the defect classes:\n")
    print("| definition | EO crack | IR hotspot |\n|---|---|---|")
    for n in names:
        print(f"| {n} | {res[('EO', 'crack')][n]:.4f} | {res[('IR', 'hotspot')][n]:.4f} |")
    # ---- geometry (Step 2) -----------------------------------------------------------------
    print("\n## panel geometry: IoU of greedy-matched approx4 panels vs GT (match IoU > 0, v2 operating thresholds)\n")
    print("| modality | GT panels | predicted | matched (IoU>0) | matched (IoU>=0.5) | mean IoU of matched(>0) | p10 | p50 | p90 | mean IoU of matched(>=0.5) | count F1@.5 | IoU-weighted F1 (>=0.5) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mod in ("EO", "IR"):
        rows = collect(mod, "solar_panel", OPS[mod]["solar_panel"], paths, cd)
        m0 = np.concatenate([np.array(greedy(pr, EPS)) for _, _, pr in rows])
        m5 = np.concatenate([np.array(greedy(pr, 0.5)) for _, _, pr in rows])
        npred, ngt = sum(n for n, _, _ in rows), sum(g for _, g, _ in rows)
        f1c = 2 * len(m5) / (npred + ngt)
        iouw = 2 * m5.sum() / (npred + ngt)
        print(f"| {mod} | {ngt} | {npred} | {len(m0)} | {len(m5)} | {m0.mean():.3f} | {np.quantile(m0, .1):.3f} | {np.quantile(m0, .5):.3f} | {np.quantile(m0, .9):.3f} | {m5.mean():.3f} | {f1c:.3f} | {iouw:.3f} |")
        print(f"\n  {mod}: loss decomposition of the IoU-weighted F1: detection factor (count F1) {f1c:.3f} -> loss {1 - f1c:.3f}; "
              f"fit factor (mean IoU of matched) {m5.mean():.3f} -> loss {1 - m5.mean():.3f}; product {f1c * m5.mean():.3f}; "
              f"misses (FN) {ngt - len(m5)} = {(ngt - len(m5)) / ngt:.3f} of GT; extra boxes (FP) {npred - len(m5)} = {(npred - len(m5)) / npred:.3f} of predictions\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
