"""Polygon validity checks (geometry only; no assumption about the official format)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LinearRing

BAD_SHAPE = "bad_shape"
NON_FINITE = "non_finite"
TOO_FEW_VERTICES = "too_few_vertices"
TOO_MANY_VERTICES = "too_many_vertices"
ZERO_AREA = "zero_area"
OUT_OF_BOUNDS = "out_of_bounds"
SELF_INTERSECTING = "self_intersecting"


@dataclass
class ValidationResult:
    issues: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues


def _shoelace_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def validate_polygon(
    polygon: np.ndarray,
    width: int | None = None,
    height: int | None = None,
    max_vertices: int | None = None,
    area_eps: float = 1e-9,
) -> ValidationResult:
    """Return every problem found with ``polygon``.

    Bounds are checked inclusively against ``[0, width] x [0, height]`` when both
    are given; whether the official format uses a different convention is unknown.
    """
    res = ValidationResult()
    try:
        pts = np.asarray(polygon, dtype=np.float64)
    except (TypeError, ValueError):
        res.issues.append(BAD_SHAPE)
        return res
    if pts.ndim != 2 or pts.shape[1] != 2:
        res.issues.append(BAD_SHAPE)
        return res
    if not np.isfinite(pts).all():
        res.issues.append(NON_FINITE)
        return res
    if len(pts) < 3:
        res.issues.append(TOO_FEW_VERTICES)
        return res
    if max_vertices is not None and len(pts) > max_vertices:
        res.issues.append(TOO_MANY_VERTICES)

    if width is not None and height is not None:
        if (pts[:, 0] < 0).any() or (pts[:, 0] > width).any() or (pts[:, 1] < 0).any() or (pts[:, 1] > height).any():
            res.issues.append(OUT_OF_BOUNDS)

    # Degenerate (all vertices coincident or collinear) is checked first: the shoelace
    # area of a self-intersecting polygon can cancel to 0 (e.g. a symmetric bowtie),
    # so it cannot be used alone to decide "zero area".
    centered = pts - pts.mean(axis=0)
    sv = np.linalg.svd(centered, compute_uv=False)
    if sv[0] <= area_eps or sv[1] <= area_eps * (1.0 + sv[0]):
        res.issues.append(ZERO_AREA)
    elif not LinearRing(pts).is_simple:
        res.issues.append(SELF_INTERSECTING)
    elif _shoelace_area(pts) <= area_eps:
        res.issues.append(ZERO_AREA)
    return res
