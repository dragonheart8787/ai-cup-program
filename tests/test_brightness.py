import numpy as np
import pytest

from src.dataset.brightness import GAMMA_MAX, GAMMA_MIN, apply_gamma, gray_of, match_mean, solve_gamma


def test_target_mean_is_reached_within_limits():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 120, (64, 64), dtype=np.uint8)
    for target in (40, 76, 100):
        res = solve_gamma(g, target)
        assert not res.clamped and abs(res.mean_after - target) < 0.01
        out = apply_gamma(g, res.gamma)
        assert abs(out.mean() - target) < 1.0  # integer rounding


def test_per_image_and_deterministic():
    a = np.full((8, 8), 30, np.uint8)
    b = np.full((8, 8), 200, np.uint8)
    ra, rb = solve_gamma(a, 76), solve_gamma(b, 76)
    assert ra.gamma < 1 < rb.gamma  # brightens the dark image, darkens the bright one
    assert solve_gamma(a, 76) == ra  # no hidden state


def test_clamping_is_reported():
    black = np.zeros((8, 8), np.uint8)
    assert solve_gamma(black, 76).clamped and solve_gamma(black, 76).gamma == GAMMA_MIN
    white = np.full((8, 8), 250, np.uint8)
    r = solve_gamma(white, 10)
    assert r.clamped and r.gamma == GAMMA_MAX


def test_rgb_roundtrip_shape_and_validation():
    rgb = np.random.default_rng(1).integers(0, 100, (16, 16, 3), dtype=np.uint8)
    out, res = match_mean(rgb, 76)
    assert out.shape == rgb.shape and out.dtype == np.uint8 and GAMMA_MIN <= res.gamma <= GAMMA_MAX
    assert gray_of(rgb).shape == (16, 16)
    with pytest.raises(ValueError):
        solve_gamma(np.zeros((2, 2), np.float32), 50)
    with pytest.raises(ValueError):
        solve_gamma(np.zeros((2, 2), np.uint8), 0)
