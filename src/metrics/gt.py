"""Ground-truth boxes for the proxy metric."""
from __future__ import annotations

import numpy as np

from src.dataset.labelme import load_labelme
from src.dataset.paths import DatasetPaths
from src.metrics.proxy import Box


def load_gt(paths: DatasetPaths, modality: str, stems: list[str]) -> dict[str, list[Box]]:
    out: dict[str, list[Box]] = {}
    for s in stems:
        doc = load_labelme(paths.label_dir(modality) / f"{s}.json", modality)
        out[s] = [Box(sh.label, np.asarray(sh.points, dtype=np.float64)) for sh in doc.shapes]
    return out
