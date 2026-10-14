"""Offline augmented YOLO datasets (outside the repo, source data untouched). Only the TRAIN split is augmented;
the val split is the original fold-0 val (hardlinks).

EO : original train + copies (~40% of the train count): half close-up crops (random location, area ratio U(0.10, 0.50),
     50% of them also get a brightness gain U(1.3, 2.0)), half gain-only copies. Polygons are clipped with shapely;
     fragments below 30% of the original area are dropped; a crop must keep at least one object.
IR : ~35% of the train images are replaced by a simulated colorization round trip (random contrast stretch,
     inferno, JPEG quality 85-95, inverse inferno); labels unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from src.dataset.closeup import apply_gain, box_from_ratio, crop_resize, transform_polygon
from src.dataset.ir_sim import simulate_colorized_roundtrip
from src.dataset.labelme import load_labelme
from src.dataset.yolo_export import CLASS_NAMES, Item, class_index, export_items, polygon_to_yolo_line

logger = logging.getLogger(__name__)


def _write_jpg(path: Path, rgb: np.ndarray, quality: int = 95) -> None:
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"failed to write {path}")


def _write_data_yaml(out_dir: Path, modality: str) -> None:
    data = {"path": str(out_dir), "train": "images/train", "val": "images/val", "names": dict(enumerate(CLASS_NAMES[modality]))}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def build_eo_aug(items: list[Item], out_dir: Path, seed: int, copy_frac: float = 0.40) -> dict[str, int]:
    train = [i for i in items if i.split == "train"]
    val = [i for i in items if i.split == "val"]
    counts = export_items(train + val, "EO", out_dir)  # originals (train + val), hardlinks
    rng = np.random.default_rng(seed)
    n_copy = int(round(copy_frac * len(train)))
    n_crop, n_gain = n_copy // 2, n_copy - n_copy // 2
    order = rng.permutation(len(train))
    crop_src, gain_src = order[:n_crop], order[n_crop:n_crop + n_gain]
    stats = {"crop": 0, "crop_gain": 0, "gain": 0, "crop_skipped": 0, "shapes_dropped": 0}
    img_dir, lab_dir = out_dir / "images" / "train", out_dir / "labels" / "train"
    for idx in crop_src:
        it = train[int(idx)]
        doc = load_labelme(it.label, "EO")
        img = _read_rgb(it.image)
        done = False
        for _ in range(30):
            cb = box_from_ratio(doc.width, doc.height, float(rng.uniform(0.10, 0.50)), float(rng.uniform(0, 1)), float(rng.uniform(0, 1)))
            kept, dropped = [], 0
            for s in doc.shapes:
                pts = transform_polygon(np.asarray(s.points), cb, doc.width, doc.height)
                if pts is None:
                    dropped += 1
                else:
                    kept.append((class_index("EO", s.label), pts))
            if kept:
                done = True
                break
        if not done:
            stats["crop_skipped"] += 1
            continue
        out = crop_resize(img, cb)
        kind = "crop"
        if rng.random() < 0.5:
            out = apply_gain(out, float(rng.uniform(1.3, 2.0)))
            kind = "crop_gain"
        name = f"{it.stem}_aug{'cg' if kind == 'crop_gain' else 'c'}"
        _write_jpg(img_dir / f"{name}.jpg", out)
        (lab_dir / f"{name}.txt").write_text("\n".join(polygon_to_yolo_line(c, p, doc.width, doc.height) for c, p in kept) + "\n", encoding="utf-8")
        stats[kind] += 1
        stats["shapes_dropped"] += dropped
    for idx in gain_src:
        it = train[int(idx)]
        doc = load_labelme(it.label, "EO")
        out = apply_gain(_read_rgb(it.image), float(rng.uniform(1.3, 2.0)))
        name = f"{it.stem}_augg"
        _write_jpg(img_dir / f"{name}.jpg", out)
        (lab_dir / f"{name}.txt").write_text((out_dir / "labels" / "train" / f"{it.stem}.txt").read_text(encoding="utf-8"), encoding="utf-8")
        stats["gain"] += 1
    _write_data_yaml(out_dir, "EO")
    result = {**counts, **{f"aug_{k}": v for k, v in stats.items()}}
    logger.info("EO aug dataset: %s", result)
    return result


def build_ir_aug(items: list[Item], out_dir: Path, seed: int, frac: float = 0.35) -> dict[str, int]:
    train = [i for i in items if i.split == "train"]
    val = [i for i in items if i.split == "val"]
    counts = export_items(train + val, "IR", out_dir)
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(train), size=int(round(frac * len(train))), replace=False).tolist())
    img_dir = out_dir / "images" / "train"
    n = 0
    for i in chosen:
        it = train[i]
        rgb = _read_rgb(it.image)
        gray = rgb[..., 0]
        lo, hi = float(rng.uniform(0, 5)), float(rng.uniform(95, 100))
        q = int(rng.integers(85, 96))
        back = simulate_colorized_roundtrip(gray, jpeg_quality=q, lo_pct=lo, hi_pct=hi)
        (img_dir / f"{it.stem}.jpg").unlink()  # replace the hardlinked original in the work tree (source untouched)
        _write_jpg(img_dir / f"{it.stem}.jpg", np.repeat(back[..., None], 3, axis=2), quality=95)
        n += 1
    result = {**counts, "aug_ir_roundtrip": n}
    logger.info("IR aug dataset: %s", result)
    return result
