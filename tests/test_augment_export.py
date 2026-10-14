import json

import numpy as np
from PIL import Image

from src.dataset.augment_export import build_eo_aug, build_ir_aug
from src.dataset.ir_sim import simulate_colorized_roundtrip
from src.dataset.labelme import IMAGE_SIZE
from src.dataset.yolo_export import Item


def make_items(tmp_path, modality, n=10):
    w, h = IMAGE_SIZE[modality]
    d = tmp_path / "src" / modality
    (d / "image").mkdir(parents=True)
    (d / "label").mkdir()
    rng = np.random.default_rng(0)
    items = []
    for i in range(n):
        gray = rng.integers(20, 200, (h, w), dtype=np.uint8)
        Image.fromarray(np.stack([gray] * 3, -1)).save(d / "image" / f"train_{i:04d}.jpg", quality=95)
        shapes = [{"label": "solar_panel", "points": [[w * .1, h * .1], [w * .45, h * .1], [w * .45, h * .9], [w * .1, h * .9]], "shape_type": "polygon", "flags": {}, "reason": None},
                  {"label": "solar_panel", "points": [[w * .55, h * .1], [w * .9, h * .1], [w * .9, h * .9], [w * .55, h * .9]], "shape_type": "polygon", "flags": {}, "reason": None}]
        (d / "label" / f"train_{i:04d}.json").write_text(json.dumps(
            {"version": "5.8.3", "flags": {}, "shapes": shapes, "imagePath": "x", "imageData": None, "imageHeight": h, "imageWidth": w}), encoding="utf-8")
        items.append(Item(f"train_{i:04d}", d / "image" / f"train_{i:04d}.jpg", d / "label" / f"train_{i:04d}.json", "val" if i < 2 else "train"))
    return items


def test_eo_aug_counts_and_source_untouched(tmp_path):
    items = make_items(tmp_path, "EO", 10)
    before = {p: p.stat().st_mtime_ns for p in (tmp_path / "src").rglob("*") if p.is_file()}
    out = tmp_path / "work" / "EO_aug"
    res = build_eo_aug(items, out, seed=1, copy_frac=0.5)
    n_train = 8
    imgs = sorted((out / "images" / "train").glob("*.jpg"))
    assert len(imgs) == n_train + round(0.5 * n_train) - res["aug_crop_skipped"]
    assert len(list((out / "images" / "val").glob("*.jpg"))) == 2
    assert not any("_aug" in p.name for p in (out / "images" / "val").glob("*"))
    assert all((out / "labels" / "train" / f"{p.stem}.txt").read_text().strip() for p in imgs)
    assert before == {p: p.stat().st_mtime_ns for p in (tmp_path / "src").rglob("*") if p.is_file()}


def test_ir_aug_replaces_share_and_keeps_labels(tmp_path):
    items = make_items(tmp_path, "IR", 20)
    out = tmp_path / "work" / "IR_aug"
    res = build_ir_aug(items, out, seed=1, frac=0.35)
    assert res["aug_ir_roundtrip"] == round(0.35 * 18)
    assert len(list((out / "images" / "train").glob("*.jpg"))) == 18
    orig = (tmp_path / "src" / "IR" / "image" / "train_0005.jpg")
    assert orig.exists()  # source image still there after the work-tree replacement


def test_roundtrip_stretch_is_gray_and_changes_pixels():
    g = np.tile(np.arange(40, 200, dtype=np.uint8), (16, 1))
    back = simulate_colorized_roundtrip(g, jpeg_quality=90, lo_pct=2, hi_pct=98)
    assert back.shape == g.shape and back.dtype == np.uint8 and back.max() - back.min() > 200
