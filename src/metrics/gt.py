"""Ground-truth boxes for the proxy metric."""
from __future__ import annotations

import numpy as np

from src.dataset.closeup import CropBox, transform_polygon
from src.dataset.labelme import load_labelme
from src.dataset.paths import DatasetPaths
from src.metrics.proxy import Box


def load_gt(paths: DatasetPaths, modality: str, stems: list[str], closeup: dict[str, CropBox] | None = None) -> dict[str, list[Box]]:
    """GT boxes; with ``closeup`` the polygons of the listed stems are cropped/zoomed like the image (small leftovers dropped)."""
    out: dict[str, list[Box]] = {}
    for s in stems:
        doc = load_labelme(paths.label_dir(modality) / f"{s}.json", modality)
        boxes = []
        for sh in doc.shapes:
            pts = np.asarray(sh.points, dtype=np.float64)
            if closeup and s in closeup:
                pts = transform_polygon(pts, closeup[s], doc.width, doc.height)
                if pts is None:
                    continue
            boxes.append(Box(sh.label, pts))
        out[s] = boxes
    return out
