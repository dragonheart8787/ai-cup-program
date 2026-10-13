"""Cached raw predictions -> final polygons (shared by evaluation and the submission writer)."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np

from src.metrics.proxy import Box
from src.postprocess.quad import finalize, to_polygon


def raw_to_boxes(entry: Mapping[str, Any], method: str, conf_min: float, drop: Counter | None = None) -> list[Box]:
    """Boxes of one cached image entry with conf >= conf_min. Invalid polygons are dropped and counted in ``drop``."""
    drop = drop if drop is not None else Counter()
    out: list[Box] = []
    for p in entry["preds"]:
        if p["conf"] < conf_min:
            continue
        pts = np.asarray(p["poly"], dtype=np.float64)
        if len(pts) < 3:
            drop["contour<3"] += 1
            continue
        poly, issues = finalize(to_polygon(pts, method), entry["w"], entry["h"])
        if poly is None:
            for i in issues:
                drop[i] += 1
            continue
        out.append(Box(p["label"], poly, p["conf"]))
    return out
