"""Submission writer (format from the official sample_submission.json / make_submission.py).

* keys: ``test/EO/image/test_001.jpg`` and ``test/IR/image/test_001.jpg``; EO and IR are independent
* value: list of ``{"label", "points", "reason"}``; no confidence field
* defective panels are output ONLY as crack / hotspot (the training labels never carry a
  solar_panel box on a defective panel); solar_panel predictions are therefore not written
* ``reason`` is a PLACEHOLDER (see src/submission/reasons.py)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

from src.metrics.proxy import Box, polygon_iou
from src.postprocess.validate import validate_polygon

logger = logging.getLogger(__name__)

DEFECTS = {"EO": "crack", "IR": "hotspot"}


def build_entries(boxes: Sequence[Box], modality: str, reason: str, width: int, height: int) -> list[dict]:
    if not reason.strip():
        raise ValueError("reason must not be empty")
    entries = []
    for b in boxes:
        if b.label == "solar_panel":
            continue
        if b.label != DEFECTS[modality]:
            raise ValueError(f"{modality} submission cannot contain label {b.label!r}")
        res = validate_polygon(b.points, width, height, max_vertices=100)
        if not res.valid:
            raise ValueError(f"invalid polygon in submission: {res.issues}")
        entries.append({"label": b.label, "points": [[round(float(x), 2), round(float(y), 2)] for x, y in b.points], "reason": reason})
    return entries


def write_submission(path: str | Path, by_key: Mapping[str, list[dict]], expected_keys: Sequence[str]) -> None:
    """Write every expected key (empty list if no detection). Output is deterministic (sorted keys, fixed separators)."""
    unknown = set(by_key) - set(expected_keys)
    if unknown:
        raise KeyError(f"{len(unknown)} keys are not in the expected list, e.g. {sorted(unknown)[:3]}")
    data = {k: by_key.get(k, []) for k in sorted(expected_keys)}
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    logger.info("wrote %s (%d keys, %d objects)", path, len(data), sum(len(v) for v in data.values()))


PANEL_DEFECT_IOU = 0.5


def build_entries_with_panels(
    defect_boxes: Sequence[Box], panel_boxes: Sequence[Box], modality: str, reason: str, width: int, height: int,
) -> list[dict]:
    """v2: defects (with placeholder reason) plus solar_panel boxes (no reason).

    A panel that overlaps a kept defect with IoU > 0.5 is dropped: in the training labels the two classes are
    mutually exclusive (a defective panel is labelled only with its defect class).
    """
    entries = build_entries(defect_boxes, modality, reason, width, height)
    kept_defects = [b for b in defect_boxes if b.label == DEFECTS[modality]]
    for b in panel_boxes:
        if b.label != "solar_panel":
            raise ValueError(f"panel_boxes contains label {b.label!r}")
        if any(polygon_iou(b.points, d.points) > PANEL_DEFECT_IOU for d in kept_defects):
            continue
        res = validate_polygon(b.points, width, height, max_vertices=100)
        if not res.valid:
            raise ValueError(f"invalid polygon in submission: {res.issues}")
        entries.append({"label": "solar_panel", "points": [[round(float(x), 2), round(float(y), 2)] for x, y in b.points]})
    return entries
