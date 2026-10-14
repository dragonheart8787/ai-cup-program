"""Simulate the colorized-IR pipeline on a gray IR image (robustness experiment).

gray -> per-image min-max stretch -> inferno -> JPEG(q=90) -> inverse inferno -> gray
"""
from __future__ import annotations

import cv2
import numpy as np

from src.dataset.ir_color import apply_inferno, colorized_to_gray


def simulate_colorized_roundtrip(gray: np.ndarray, jpeg_quality: int = 90, lo_pct: float = 0.0, hi_pct: float = 100.0) -> np.ndarray:
    if gray.ndim == 3:
        if not (gray[..., 0] == gray[..., 1]).all() or not (gray[..., 1] == gray[..., 2]).all():
            raise ValueError("expected a gray (3-channel identical) image")
        gray = gray[..., 0]
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("gray must be uint8 (H, W) or (H, W, 3)")
    lo, hi = float(np.percentile(gray, lo_pct)), float(np.percentile(gray, hi_pct))  # 0/100 = plain min-max
    if hi <= lo:
        raise ValueError("constant image cannot be contrast-stretched")
    stretched = np.round(np.clip((gray.astype(np.float64) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    rgb = apply_inferno(stretched)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    dec = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return colorized_to_gray(dec)
