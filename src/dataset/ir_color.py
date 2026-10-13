"""Detect colorized (pseudo-color) IR images and map them back to uint8 gray.

The 200 colorized test IR images use a palette close to matplotlib's inferno
(mean nearest-LUT RGB distance ~15). The mapping returns the inferno LUT index
(0..255); whether that index matches the gray level of the training IR images
is NOT established -- compare distributions before relying on it.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def channel_difference(rgb: np.ndarray) -> float:
    """mean|R-G| + mean|G-B| (0.0 for a true gray image)."""
    _check_rgb(rgb)
    a = rgb.astype(np.int16)
    return float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean())


def is_colorized(rgb: np.ndarray, threshold: float = 1.0) -> bool:
    return channel_difference(rgb) > threshold


@lru_cache(maxsize=1)
def inferno_lut() -> np.ndarray:
    """(256, 3) uint8 inferno colors."""
    from matplotlib import colormaps

    return np.round(colormaps["inferno"](np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(np.uint8)


@lru_cache(maxsize=1)
def _tree() -> cKDTree:
    return cKDTree(inferno_lut().astype(np.float64))


def colorized_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Nearest inferno entry per pixel -> uint8 index image of shape (H, W)."""
    _check_rgb(rgb)
    h, w, _ = rgb.shape
    _, idx = _tree().query(rgb.reshape(-1, 3).astype(np.float64), workers=-1)
    return idx.reshape(h, w).astype(np.uint8)


def apply_inferno(gray: np.ndarray) -> np.ndarray:
    """Forward map used by the tests: uint8 gray (H, W) -> inferno RGB (H, W, 3)."""
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("gray must be a 2-D uint8 array")
    return inferno_lut()[gray]


def _check_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}")
