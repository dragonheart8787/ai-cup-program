"""Polygon <-> mask conversion.

Conventions (these are OUR internal conventions, not the official format):
a polygon is an ``(N, 2)`` array of ``(x, y)`` pixel coordinates; a mask is a
``(H, W)`` ``uint8`` array with values 0/1. Converting from the official
format into this one is a separate step that has to wait for the official spec.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def polygon_to_mask(polygon: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterise one polygon. Coordinates are rounded to the nearest pixel and clipped by OpenCV."""
    pts = np.asarray(polygon, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"polygon must have shape (N, 2), got {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("polygon contains NaN or inf")
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask


def mask_to_polygons(
    mask: np.ndarray,
    epsilon_ratio: float = 0.01,
    min_area: float = 1.0,
    max_vertices: int | None = None,
) -> list[np.ndarray]:
    """Outer contours of a binary mask as simplified polygons.

    ``approxPolyDP`` epsilon is ``epsilon_ratio * perimeter``. Holes are ignored.
    Contours with fewer than 3 vertices or area below ``min_area`` are dropped.
    If ``max_vertices`` is set, epsilon is doubled until the polygon fits.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {m.shape}")
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[np.ndarray] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        eps = max(epsilon_ratio * perimeter, 1e-6)
        approx = cv2.approxPolyDP(cnt, eps, True)
        while max_vertices is not None and len(approx) > max_vertices:
            eps *= 2
            approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        polygons.append(approx.reshape(-1, 2).astype(np.float64))
    logger.debug("mask_to_polygons: %d polygon(s)", len(polygons))
    return polygons
