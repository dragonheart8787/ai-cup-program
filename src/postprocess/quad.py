"""Turn a raw instance contour into a (preferably 4-point) polygon.

Training annotations are almost all 4-point quadrilaterals (EO 29647/29934,
IR 26492/26701 non-5+ point shapes), so the submission polygons are quads.
Methods (compared on validation data, see scripts/eval_oof.py):

* ``minrect``  - cv2.minAreaRect of the contour (always 4 points, rotated rectangle)
* ``approx``   - cv2.approxPolyDP with a fixed epsilon ratio (variable vertex count)
* ``approx4``  - approxPolyDP on the convex hull with the smallest epsilon that yields
                 exactly 4 vertices; if none exists the result is ``None`` (the caller counts it)
"""
from __future__ import annotations

import cv2
import numpy as np

from src.postprocess.validate import validate_polygon

METHODS = ("minrect", "approx", "approx4")


def _as_contour(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError(f"contour must be (N>=3, 2), got {pts.shape}")
    return pts.reshape(-1, 1, 2)


def minrect(points: np.ndarray) -> np.ndarray:
    box = cv2.boxPoints(cv2.minAreaRect(_as_contour(points)))
    return box.astype(np.float64)


def approx(points: np.ndarray, epsilon_ratio: float = 0.02, max_vertices: int = 100) -> np.ndarray | None:
    cnt = _as_contour(points)
    per = cv2.arcLength(cnt, True)
    eps = max(epsilon_ratio * per, 1e-6)
    a = cv2.approxPolyDP(cnt, eps, True)
    while len(a) > max_vertices:
        eps *= 2
        a = cv2.approxPolyDP(cnt, eps, True)
    return a.reshape(-1, 2).astype(np.float64) if len(a) >= 3 else None


def approx4(points: np.ndarray) -> np.ndarray | None:
    hull = cv2.convexHull(_as_contour(points))
    per = cv2.arcLength(hull, True)
    lo, hi = 0.0, 0.5 * per
    best = None
    for _ in range(40):  # bisection on epsilon: vertex count is non-increasing in epsilon
        mid = (lo + hi) / 2
        a = cv2.approxPolyDP(hull, max(mid, 1e-6), True)
        if len(a) == 4:
            best = a
            hi = mid  # smallest epsilon that still gives 4
        elif len(a) > 4:
            lo = mid
        else:
            hi = mid
    return None if best is None else best.reshape(-1, 2).astype(np.float64)


def to_polygon(points: np.ndarray, method: str) -> np.ndarray | None:
    if method == "minrect":
        return minrect(points)
    if method == "approx":
        return approx(points)
    if method == "approx4":
        return approx4(points)
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def clip_to_image(poly: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.asarray(poly, dtype=np.float64).copy()
    out[:, 0] = np.clip(out[:, 0], 0, width)
    out[:, 1] = np.clip(out[:, 1], 0, height)
    return out


def finalize(poly: np.ndarray | None, width: int, height: int, max_vertices: int = 100) -> tuple[np.ndarray | None, list[str]]:
    """Clip to the image, validate; returns (polygon or None, issues). Invalid polygons are dropped by the caller."""
    if poly is None:
        return None, ["no_polygon"]
    clipped = clip_to_image(poly, width, height)
    res = validate_polygon(clipped, width, height, max_vertices=max_vertices)
    return (clipped, []) if res.valid else (None, res.issues)
