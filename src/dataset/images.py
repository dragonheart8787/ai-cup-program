"""Image-level measurements (read-only)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from src.dataset.ir_color import is_colorized


@dataclass(frozen=True)
class ImageRecord:
    width: int
    height: int
    mean_brightness: float
    std_brightness: float
    phash: str
    is_colorized: bool | None  # None for EO (not applicable)


def read_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def measure_image(path: Path, modality: str, colorized_threshold: float) -> ImageRecord:
    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValueError(f"{path}: expected RGB JPEG, got mode {im.mode}")
        rgb = im.copy()
    gray = np.asarray(rgb.convert("L"), dtype=np.float64)
    colorized = is_colorized(np.asarray(rgb), colorized_threshold) if modality == "IR" else None
    return ImageRecord(
        width=rgb.width, height=rgb.height,
        mean_brightness=float(gray.mean()), std_brightness=float(gray.std()),
        phash=str(imagehash.phash(rgb)), is_colorized=colorized,
    )


def thumbnail(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Small grayscale thumbnail (float32, flattened) used for near-duplicate checks."""
    with Image.open(path) as im:
        im.draft("L", (size[0] * 4, size[1] * 4))
        t = im.convert("L").resize(size, Image.BILINEAR)
    return np.asarray(t, dtype=np.float32).ravel()
