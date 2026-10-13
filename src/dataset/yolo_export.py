"""labelme -> Ultralytics segmentation dataset (one dataset per modality).

Ultralytics maps ``.../images/...`` to ``.../labels/...`` while the competition
folders are called ``image`` / ``label``, so a conventional tree is built outside
the repo (and outside the read-only source data):

    <work_root>/<MOD>/fold<k>/images/{train,val}/<stem>.jpg   (hardlink, else copy)
                          /labels/{train,val}/<stem>.txt
                          /data.yaml

Source files are never modified. Only the requested fold's split is written.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.dataset.labelme import IMAGE_SIZE, LabelmeDoc, load_labelme
from src.dataset.paths import DatasetPaths

logger = logging.getLogger(__name__)

CLASS_NAMES: dict[str, list[str]] = {"EO": ["solar_panel", "crack"], "IR": ["solar_panel", "hotspot"]}


def class_index(modality: str, label: str) -> int:
    try:
        return CLASS_NAMES[modality].index(label)
    except ValueError as e:
        raise ValueError(f"label {label!r} is not a {modality} class {CLASS_NAMES[modality]}") from e


def polygon_to_yolo_line(cls_id: int, points: tuple[tuple[float, float], ...] | np.ndarray, width: int, height: int) -> str:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError(f"need >= 3 (x, y) points, got shape {pts.shape}")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    norm = np.clip(pts / np.array([width, height], dtype=np.float64), 0.0, 1.0)  # drops ~1e-13 overshoot only
    return f"{cls_id} " + " ".join(f"{v:.8f}" for v in norm.ravel())


def yolo_line_to_polygon(line: str, width: int, height: int) -> tuple[int, np.ndarray]:
    parts = line.split()
    if len(parts) < 7 or (len(parts) - 1) % 2:
        raise ValueError(f"malformed segmentation line: {line[:60]!r}")
    vals = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
    return int(parts[0]), vals * np.array([width, height], dtype=np.float64)


def doc_to_lines(doc: LabelmeDoc) -> list[str]:
    return [polygon_to_yolo_line(class_index(doc.modality, s.label), s.points, doc.width, doc.height) for s in doc.shapes]


@dataclass(frozen=True)
class Item:
    stem: str
    image: Path
    label: Path
    split: str  # "train" | "val"


def build_items(paths: DatasetPaths, folds: pd.DataFrame, modality: str, fold: int) -> list[Item]:
    """Items of one modality; samples in ``fold`` are val, all other train samples are train."""
    need = {"sample_id", "fold"}
    if not need <= set(folds.columns):
        raise ValueError(f"folds table needs columns {need}")
    suffix = f"_{modality}"
    rows = folds[folds.sample_id.str.endswith(suffix)]
    if rows.empty:
        raise ValueError(f"no {modality} rows in folds table")
    if fold not in set(rows.fold):
        raise ValueError(f"fold {fold} not present (have {sorted(set(rows.fold))})")
    items = []
    for sid, f in zip(rows.sample_id, rows.fold):
        stem = sid[: -len(suffix)]
        items.append(Item(stem, paths.image_dir("train", modality) / f"{stem}.jpg",
                          paths.label_dir(modality) / f"{stem}.json", "val" if f == fold else "train"))
    return items


def _place_image(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError as e:
        logger.warning("hardlink failed for %s (%s); copying instead", src.name, e)
        shutil.copy2(src, dst)
        return "copy"


def export_items(items: list[Item], modality: str, out_dir: Path) -> dict[str, int]:
    """Write images + labels + data.yaml; returns counts."""
    from PIL import Image

    counts = {"train": 0, "val": 0, "shapes": 0, "hardlink": 0, "copy": 0}
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    for it in items:
        with Image.open(it.image) as im:
            size = im.size
        doc = load_labelme(it.label, modality, actual_size=size)
        if size != IMAGE_SIZE[modality]:
            raise ValueError(f"{it.image}: unexpected size {size}")
        lines = doc_to_lines(doc)
        (out_dir / "labels" / it.split / f"{it.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts[_place_image(it.image, out_dir / "images" / it.split / f"{it.stem}.jpg")] += 1
        counts[it.split] += 1
        counts["shapes"] += len(lines)
    data = {"path": str(out_dir), "train": "images/train", "val": "images/val",
            "names": dict(enumerate(CLASS_NAMES[modality]))}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger.info("exported %s: %s", out_dir, counts)
    return counts
