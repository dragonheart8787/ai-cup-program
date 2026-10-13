import numpy as np
import pytest

from src.postprocess.polygon import mask_to_polygons, polygon_to_mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0


def test_rectangle_roundtrip():
    poly = np.array([[10, 20], [60, 20], [60, 50], [10, 50]], dtype=float)
    mask = polygon_to_mask(poly, 100, 120)
    assert mask.shape == (100, 120) and mask.dtype == np.uint8
    assert mask[35, 30] == 1 and mask[5, 5] == 0
    out = mask_to_polygons(mask)
    assert len(out) == 1
    assert len(out[0]) == 4
    assert iou(mask, polygon_to_mask(out[0], 100, 120)) > 0.95


def test_circle_roundtrip_simplifies():
    yy, xx = np.mgrid[:200, :200]
    mask = ((xx - 100) ** 2 + (yy - 100) ** 2 <= 60**2).astype(np.uint8)
    out = mask_to_polygons(mask, epsilon_ratio=0.005)
    assert len(out) == 1
    assert 8 <= len(out[0]) < 100
    assert iou(mask, polygon_to_mask(out[0], 200, 200)) > 0.97


def test_multiple_blobs_and_min_area():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[5:25, 5:25] = 1
    mask[50:70, 40:70] = 1
    mask[40, 10] = 1  # single pixel, area 0
    assert len(mask_to_polygons(mask, min_area=1.0)) == 2


def test_max_vertices_respected():
    yy, xx = np.mgrid[:200, :200]
    mask = ((xx - 100) ** 2 + (yy - 100) ** 2 <= 80**2).astype(np.uint8)
    out = mask_to_polygons(mask, epsilon_ratio=0.001, max_vertices=12)
    assert len(out) == 1 and len(out[0]) <= 12


def test_empty_mask():
    assert mask_to_polygons(np.zeros((10, 10), dtype=np.uint8)) == []


def test_polygon_to_mask_rejects_bad_input():
    with pytest.raises(ValueError):
        polygon_to_mask(np.array([[0, 0], [np.nan, 1], [2, 2]]), 10, 10)
    with pytest.raises(ValueError):
        polygon_to_mask(np.zeros((3, 3)), 10, 10)


def test_polygon_partly_outside_is_clipped():
    poly = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], dtype=float)
    mask = polygon_to_mask(poly, 50, 50)
    assert mask[0, 0] == 1 and mask[20, 20] == 0
