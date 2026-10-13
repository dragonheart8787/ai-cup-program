import numpy as np

from src.postprocess import validate as v
from src.postprocess.validate import validate_polygon

SQUARE = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=float)


def test_valid_square():
    r = validate_polygon(SQUARE, width=100, height=100, max_vertices=10)
    assert r.valid and r.issues == []


def test_self_intersecting_bowtie():
    bow = np.array([[0, 0], [10, 10], [10, 0], [0, 10]], dtype=float)
    assert v.SELF_INTERSECTING in validate_polygon(bow).issues


def test_zero_area_collinear_and_duplicate():
    line = np.array([[0, 0], [5, 5], [10, 10]], dtype=float)
    assert v.ZERO_AREA in validate_polygon(line).issues
    dup = np.array([[3, 3], [3, 3], [3, 3]], dtype=float)
    assert v.ZERO_AREA in validate_polygon(dup).issues


def test_out_of_bounds():
    poly = SQUARE + 70
    assert v.OUT_OF_BOUNDS in validate_polygon(poly, width=100, height=100).issues
    assert v.OUT_OF_BOUNDS not in validate_polygon(poly).issues  # no image size -> not checked
    neg = SQUARE - 20
    assert v.OUT_OF_BOUNDS in validate_polygon(neg, width=100, height=100).issues


def test_bounds_inclusive_at_edge():
    edge = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    assert validate_polygon(edge, width=100, height=100).valid


def test_too_many_vertices():
    t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    circle = np.stack([50 + 30 * np.cos(t), 50 + 30 * np.sin(t)], axis=1)
    assert v.TOO_MANY_VERTICES in validate_polygon(circle, max_vertices=200).issues
    assert validate_polygon(circle, max_vertices=300).valid


def test_nan_inf():
    for bad in (np.nan, np.inf, -np.inf):
        p = SQUARE.copy()
        p[1, 0] = bad
        assert validate_polygon(p).issues == [v.NON_FINITE]


def test_bad_shape_and_few_vertices():
    assert validate_polygon(np.zeros((4, 3))).issues == [v.BAD_SHAPE]
    assert validate_polygon(np.zeros(5)).issues == [v.BAD_SHAPE]
    assert validate_polygon(SQUARE[:2]).issues == [v.TOO_FEW_VERTICES]
    assert validate_polygon([["a", "b"]]).issues == [v.BAD_SHAPE]


def test_multiple_issues_reported():
    p = np.array([[0, 0], [200, 200], [200, 0], [0, 200]], dtype=float)
    assert set(validate_polygon(p, width=100, height=100).issues) == {v.OUT_OF_BOUNDS, v.SELF_INTERSECTING}
