import numpy as np
import pytest

from src.dataset.closeup import (CropBox, apply_gain, box_from_ratio, closeup_plan, crop_resize, transform_polygon)

W, H = 1920, 1080


def test_plan_deterministic_and_ratio():
    stems = [f"train_{i:04d}" for i in range(1, 201)]
    a = closeup_plan(stems, seed=42)
    b = closeup_plan(stems, seed=42)
    assert a == b and len(a) == 60
    for cb in a.values():
        r = cb.w * cb.h / (W * H)
        assert 0.14 <= r <= 0.36
    assert closeup_plan(stems, seed=1) != a


def test_polygon_fully_inside_maps_exactly():
    cb = CropBox(480, 270, 1440, 810)  # 0.25 area, 2x zoom
    poly = np.array([[500, 300], [900, 300], [900, 700], [500, 700]], float)
    out = transform_polygon(poly, cb, W, H)
    expect = np.array([[40, 60], [840, 60], [840, 860], [40, 860]], float)
    assert np.allclose(out[np.lexsort((out[:, 1], out[:, 0]))], expect[np.lexsort((expect[:, 1], expect[:, 0]))], atol=1e-6)


def test_partial_polygon_kept_or_dropped_by_area():
    cb = CropBox(0, 0, 960, 540)
    straddle = np.array([[900, 100], [1100, 100], [1100, 300], [900, 300]], float)  # 30% inside = 60/200
    assert transform_polygon(straddle, cb, W, H, min_keep_frac=0.3) is not None
    assert transform_polygon(straddle, cb, W, H, min_keep_frac=0.31) is None
    outside = np.array([[1200, 600], [1300, 600], [1300, 700]], float)
    assert transform_polygon(outside, cb, W, H) is None


def test_image_and_label_consistent():
    img = np.zeros((H, W, 3), np.uint8)
    img[300:700, 500:900] = 255
    poly = np.array([[500, 300], [900, 300], [900, 700], [500, 700]], float)
    cb = box_from_ratio(W, H, 0.25, 0.5, 0.5)
    out = crop_resize(img, cb)
    p = transform_polygon(poly, cb, W, H)
    assert out.shape == img.shape
    mid = p.mean(0).astype(int)
    assert out[mid[1], mid[0]].min() == 255
    assert out[5, 5].max() == 0 or p[:, 0].min() <= 5


def test_gain_clips_and_box_validation():
    assert apply_gain(np.array([100, 200], np.uint8), 2.0).tolist() == [200, 255]
    with pytest.raises(ValueError):
        crop_resize(np.zeros((10, 10, 3), np.uint8), CropBox(0, 0, 20, 5))
    with pytest.raises(ValueError):
        box_from_ratio(W, H, 0)
