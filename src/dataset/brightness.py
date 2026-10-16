"""Per-image, deterministic brightness transform (gamma) used by the distribution-shift sensitivity tests.

Each image is transformed using ONLY its own pixels: gamma is solved so that the image's mean gray level becomes
``target`` (bisection, gamma limited to [0.4, 2.5]). No dataset or test-set statistics are involved, so the same function can
be applied to any single image at inference time. The achieved mean and the clamping are returned for logging.

Why the MEAN gray level and not the median pixel: most EO images are mostly black background (the pixel median is 0 for 55-69% of
the images), and gamma can never move a 0 median. The mean gray level is the quantity behind the dataset-level brightness numbers
(train median of the per-image mean ~50, test ~76) quoted in the reports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

GAMMA_MIN, GAMMA_MAX = 0.4, 2.5


@dataclass(frozen=True)
class GammaResult:
    gamma: float
    mean_before: float
    mean_after: float
    clamped: bool


def gray_of(rgb: np.ndarray) -> np.ndarray:
    """uint8 (H, W) luminance, identical to PIL's convert('L') used for the metadata brightness."""
    if rgb.ndim == 2:
        return rgb
    return np.asarray(Image.fromarray(rgb).convert("L"))


def _mean_after(hist: np.ndarray, gamma: float) -> float:
    v = np.arange(256, dtype=np.float64)
    return float((hist * (255.0 * (v / 255.0) ** gamma)).sum() / hist.sum())


def solve_gamma(gray: np.ndarray, target: float) -> GammaResult:
    if gray.dtype != np.uint8:
        raise ValueError("gray must be uint8")
    if not 0 < target < 255:
        raise ValueError("target must be in (0, 255)")
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    before = float(gray.mean())
    hi_mean, lo_mean = _mean_after(hist, GAMMA_MAX), _mean_after(hist, GAMMA_MIN)  # mean decreases with gamma
    if target >= lo_mean:
        g, clamped = GAMMA_MIN, target > lo_mean
    elif target <= hi_mean:
        g, clamped = GAMMA_MAX, target < hi_mean
    else:
        lo, hi = GAMMA_MIN, GAMMA_MAX
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _mean_after(hist, mid) > target:
                lo = mid
            else:
                hi = mid
        g, clamped = 0.5 * (lo + hi), False
    return GammaResult(g, before, _mean_after(hist, g), clamped)


def apply_gamma(rgb: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.round(255.0 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
    return lut[rgb]


def match_mean(rgb: np.ndarray, target: float) -> tuple[np.ndarray, GammaResult]:
    """Gamma-transform ``rgb`` so that its mean gray becomes ``target`` (within the gamma limits)."""
    res = solve_gamma(gray_of(rgb), target)
    return apply_gamma(rgb, res.gamma), res
