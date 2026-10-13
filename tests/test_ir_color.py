import cv2
import numpy as np
import pytest

from src.dataset.ir_color import apply_inferno, colorized_to_gray, inferno_lut, is_colorized


def gradient():
    return np.tile(np.arange(256, dtype=np.uint8), (8, 1))


def test_exact_roundtrip_on_gradient():
    g = gradient()
    back = colorized_to_gray(apply_inferno(g))
    assert np.abs(back.astype(int) - g.astype(int)).max() <= 1  # duplicate LUT colors may shift by one


def test_roundtrip_after_jpeg():
    g = gradient()
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(apply_inferno(g), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    rgb = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    err = np.abs(colorized_to_gray(rgb).astype(int) - g.astype(int))
    assert err.mean() < 4 and np.median(err) <= 2


def test_detection():
    g = gradient()
    assert not is_colorized(np.stack([g] * 3, -1))
    assert is_colorized(apply_inferno(g))


def test_lut_shape_and_input_checks():
    assert inferno_lut().shape == (256, 3)
    with pytest.raises(ValueError):
        colorized_to_gray(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        apply_inferno(np.zeros((4, 4), dtype=np.float32))
