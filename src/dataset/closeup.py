"""Close-up crop + brightness gain helpers (val proxy sets and offline training augmentation).

The same crop box is used for the image (integer pixel box, resized back to full size) and for the
polygons (shapely clip, then the same affine map), so image and labels stay consistent.
Objects whose clipped area is below ``min_keep_frac`` of the original area are dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box as sbox
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class CropBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


def _stem_int(stem: str) -> int:
    m = re.search(r"(\d+)$", stem)
    if not m:
        raise ValueError(f"cannot derive a number from stem {stem!r}")
    return int(m.group(1))


def box_from_ratio(width: int, height: int, area_ratio: float, cx: float = 0.5, cy: float = 0.5) -> CropBox:
    """Box with the image aspect ratio and ``area_ratio`` of its area; (cx, cy) in [0, 1] places it inside the image."""
    if not 0 < area_ratio <= 1:
        raise ValueError("area_ratio must be in (0, 1]")
    s = np.sqrt(area_ratio)
    cw, ch = max(2, int(round(width * s))), max(2, int(round(height * s)))
    x0 = int(round((width - cw) * cx))
    y0 = int(round((height - ch) * cy))
    return CropBox(x0, y0, x0 + cw, y0 + ch)


def closeup_plan(stems: list[str], seed: int, frac: float = 0.3, width: int = 1920, height: int = 1080,
                 ratio_range: tuple[float, float] = (0.15, 0.35)) -> dict[str, CropBox]:
    """Deterministic: ``frac`` of the stems get a CENTER crop with area ratio ~ U(ratio_range)."""
    rng = np.random.default_rng(seed)
    n = int(round(frac * len(stems)))
    chosen = sorted(rng.choice(len(stems), size=n, replace=False).tolist())
    plan = {}
    for i in chosen:
        r = np.random.default_rng([seed, _stem_int(stems[i])]).uniform(*ratio_range)
        plan[stems[i]] = box_from_ratio(width, height, float(r))
    return plan


def crop_resize(img: np.ndarray, cb: CropBox) -> np.ndarray:
    h, w = img.shape[:2]
    if not (0 <= cb.x0 < cb.x1 <= w and 0 <= cb.y0 < cb.y1 <= h):
        raise ValueError(f"crop box {cb} outside image {w}x{h}")
    return cv2.resize(img[cb.y0:cb.y1, cb.x0:cb.x1], (w, h), interpolation=cv2.INTER_LINEAR)


def _largest_polygon(g: BaseGeometry) -> Polygon | None:
    if g.is_empty:
        return None
    if isinstance(g, Polygon):
        return g
    parts = [p for p in getattr(g, "geoms", []) if isinstance(p, (Polygon, MultiPolygon))]
    flat: list[Polygon] = []
    for p in parts:
        flat += list(p.geoms) if isinstance(p, MultiPolygon) else [p]
    return max(flat, key=lambda p: p.area) if flat else None


def transform_polygon(points: np.ndarray, cb: CropBox, width: int, height: int, min_keep_frac: float = 0.3) -> np.ndarray | None:
    """Clip to the crop box, map to full-size coordinates; None if the kept area is < min_keep_frac of the original."""
    poly = Polygon(np.asarray(points, dtype=np.float64))
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    clipped = _largest_polygon(poly.intersection(sbox(cb.x0, cb.y0, cb.x1, cb.y1)))
    if clipped is None or clipped.area < min_keep_frac * poly.area:
        return None
    xy = np.asarray(clipped.exterior.coords)[:-1]
    out = np.empty_like(xy)
    out[:, 0] = (xy[:, 0] - cb.x0) * (width / cb.w)
    out[:, 1] = (xy[:, 1] - cb.y0) * (height / cb.h)
    out[:, 0] = np.clip(out[:, 0], 0, width)
    out[:, 1] = np.clip(out[:, 1], 0, height)
    return out if len(out) >= 3 else None


def apply_gain(img: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(img.astype(np.float64) * gain, 0, 255).round().astype(np.uint8)
