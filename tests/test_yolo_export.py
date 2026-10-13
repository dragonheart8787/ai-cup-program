import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.dataset.labelme import IMAGE_SIZE
from src.dataset.paths import DatasetPaths
from src.dataset.yolo_export import (build_items, class_index, export_items, polygon_to_yolo_line,
                                     yolo_line_to_polygon)


def test_roundtrip_error_is_tiny():
    rng = np.random.default_rng(0)
    for w, h in ((1920, 1080), (640, 512)):
        for _ in range(200):
            n = int(rng.integers(3, 9))
            pts = np.c_[rng.uniform(0, w, n), rng.uniform(0, h, n)]
            cid, back = yolo_line_to_polygon(polygon_to_yolo_line(1, pts, w, h), w, h)
            assert cid == 1 and back.shape == pts.shape
            assert np.abs(back - pts).max() < 1e-4  # 8 decimals of a normalised coordinate


def test_clip_and_validation():
    line = polygon_to_yolo_line(0, [(-1e-13, 0), (100, 0), (100, 1080.0000000000005)], 1920, 1080)
    vals = np.array(line.split()[1:], float)
    assert vals.min() >= 0.0 and vals.max() <= 1.0
    with pytest.raises(ValueError):
        polygon_to_yolo_line(0, [(0, 0), (1, 1)], 10, 10)
    with pytest.raises(ValueError):
        yolo_line_to_polygon("0 0.1 0.1 0.2 0.2", 10, 10)
    with pytest.raises(ValueError):
        class_index("EO", "hotspot")


def _tree(tmp_path):
    root = tmp_path / "src"
    w, h = IMAGE_SIZE["IR"]
    for m, d in (("IR", "hotspot"),):
        (root / "tr" / m / "image").mkdir(parents=True)
        (root / "tr" / m / "label").mkdir(parents=True)
        for i in (1, 2):
            Image.new("RGB", (w, h), (10, 10, 10)).save(root / "tr" / m / "image" / f"train_000{i}.jpg")
            shapes = [{"label": "solar_panel", "points": [[10, 10], [100, 10], [100, 100], [10, 100]], "shape_type": "polygon",
                       "flags": {}, "reason": None},
                      {"label": d, "points": [[200, 200], [300, 200], [300, 300]], "shape_type": "polygon", "flags": {}, "reason": "過熱"}]
            (root / "tr" / m / "label" / f"train_000{i}.json").write_text(json.dumps(
                {"version": "7.1.1", "flags": {}, "shapes": shapes, "imagePath": "../image/x.jpg", "imageData": None,
                 "imageHeight": h, "imageWidth": w}), encoding="utf-8")
    (root / "te").mkdir()
    (root / "ms.py").write_text("")
    (root / "ss.json").write_text("{}")
    cfg = {"dataset": {"data_root": str(root), "train_root": "tr", "test_root": "te",
                       "make_submission_py": "ms.py", "sample_submission_json": "ss.json"}}
    return DatasetPaths.from_config(cfg, env={}), root


def test_export_fold_split_and_source_untouched(tmp_path):
    paths, root = _tree(tmp_path)
    folds = pd.DataFrame({"sample_id": ["train_0001_IR", "train_0002_IR"], "group_id": ["a", "b"], "fold": [0, 1]})
    items = build_items(paths, folds, "IR", 0)
    assert [(i.stem, i.split) for i in items] == [("train_0001", "val"), ("train_0002", "train")]
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    out = tmp_path / "work" / "IR" / "fold0"
    counts = export_items(items, "IR", out)
    assert counts["val"] == 1 and counts["train"] == 1 and counts["shapes"] == 4
    assert (out / "labels" / "val" / "train_0001.txt").read_text().count("\n") == 2
    assert (out / "images" / "train" / "train_0002.jpg").exists() and (out / "data.yaml").exists()
    assert before == {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        build_items(paths, folds, "IR", 5)
