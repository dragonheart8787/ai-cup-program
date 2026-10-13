"""PROXY metric -- NOT the official competition score.

Used only to compare our own versions until the official evaluation program is
available (NEED OFFICIAL FILE). One-to-one greedy matching by descending
confidence at a fixed polygon IoU; no confidence field exists in submissions, so
every kept box counts in the denominator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from shapely.geometry import Polygon

PANEL = "solar_panel"


@dataclass(frozen=True)
class Box:
    label: str
    points: np.ndarray  # (N, 2)
    conf: float = 1.0


def _poly(points: np.ndarray) -> Polygon:
    p = Polygon(np.asarray(points, dtype=np.float64))
    return p if p.is_valid else p.buffer(0)


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = _poly(a), _poly(b)
    if pa.is_empty or pb.is_empty:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, o: "Counts") -> None:
        self.tp += o.tp
        self.fp += o.fp
        self.fn += o.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match_image(preds: Sequence[Box], gts: Sequence[Box], iou_thr: float) -> Counts:
    """Greedy one-to-one matching; a prediction only matches a GT of the same label."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i].conf)
    used: set[int] = set()
    c = Counts()
    pp = [_poly(p.points) for p in preds]
    gp = [_poly(g.points) for g in gts]
    gb = np.array([g.bounds for g in gp]).reshape(-1, 4)
    for i in order:
        best, best_j = 0.0, -1
        x0, y0, x1, y1 = pp[i].bounds
        near = np.nonzero((gb[:, 0] < x1) & (gb[:, 2] > x0) & (gb[:, 1] < y1) & (gb[:, 3] > y0))[0] if len(gp) else []
        for j in near:
            if j in used or gts[j].label != preds[i].label or pp[i].is_empty or gp[j].is_empty:
                continue
            inter = pp[i].intersection(gp[j]).area
            union = pp[i].area + gp[j].area - inter
            v = inter / union if union > 0 else 0.0
            if v > best:
                best, best_j = v, int(j)
        if best_j >= 0 and best >= iou_thr:
            used.add(best_j)
            c.tp += 1
        else:
            c.fp += 1
    c.fn = len(gts) - len(used)
    return c


def evaluate(preds_by_image: dict[str, list[Box]], gts_by_image: dict[str, list[Box]], iou_thr: float) -> dict[str, float]:
    """Counts per group: panel / defect / all. Keys with prefix ``{group}_``."""
    groups = {"panel": lambda b: b.label == PANEL, "defect": lambda b: b.label != PANEL, "all": lambda b: True}
    tot = {g: Counts() for g in groups}
    n_img = len(gts_by_image)
    if set(preds_by_image) - set(gts_by_image):
        raise KeyError("predictions for images that have no ground truth entry")
    for img, gts in gts_by_image.items():
        preds = preds_by_image.get(img, [])
        for g, sel in groups.items():
            tot[g].add(match_image([p for p in preds if sel(p)], [x for x in gts if sel(x)], iou_thr))
    out: dict[str, float] = {"iou": iou_thr, "n_images": n_img}
    for g, c in tot.items():
        out.update({f"{g}_f1": c.f1, f"{g}_precision": c.precision, f"{g}_recall": c.recall,
                    f"{g}_fp_per_img": c.fp / n_img, f"{g}_fn_per_img": c.fn / n_img,
                    f"{g}_tp": c.tp, f"{g}_fp": c.fp, f"{g}_fn": c.fn})
    return out
