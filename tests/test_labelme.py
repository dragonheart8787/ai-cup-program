import copy

import pytest

from src.dataset.labelme import IMAGE_SIZE, LabelmeError, parse_labelme

SQ = [[10, 10], [50, 10], [50, 50], [10, 50]]


def shape(label="solar_panel", points=None, reason=None, **kw):
    s = {"label": label, "points": SQ if points is None else points, "group_id": None, "description": "", "shape_type": "polygon",
         "flags": {}, "mask": None, "reason": reason}  # real data: null on solar_panel
    s.update(kw)
    return s


def doc(modality="EO", shapes=None, **kw):
    w, h = IMAGE_SIZE[modality]
    d = {"version": "5.8.3", "flags": {}, "shapes": shapes if shapes is not None else [shape()],
         "imagePath": "../image/x.jpg", "imageData": None, "imageHeight": h, "imageWidth": w}
    d.update(kw)
    return d


def test_valid_eo_and_ir():
    eo = parse_labelme(doc("EO", [shape(), shape("crack", reason="面板裂損")]), "EO", actual_size=(1920, 1080))
    assert (eo.n_panels, eo.n_defects, eo.has_auto_label) == (1, 1, False)
    ir = parse_labelme(doc("IR", [shape(), shape("hotspot", reason="過熱")]), "IR", actual_size=(640, 512))
    assert ir.n_defects == 1 and ir.defects[0].label == "hotspot"


def test_verified_sizes_are_fixed():
    assert IMAGE_SIZE == {"EO": (1920, 1080), "IR": (640, 512)}


def test_auto_label_counts():
    s = shape(flags={"auto_label": True}, description="auto confidence=0.93 | simplified=4points")
    d = parse_labelme(doc("IR", [s, shape()]), "IR")
    assert d.has_auto_label and d.n_auto_panels == 1


@pytest.mark.parametrize("modality,label", [("EO", "hotspot"), ("IR", "crack"), ("EO", "foo")])
def test_label_not_allowed_for_modality(modality, label):
    with pytest.raises(LabelmeError, match="not allowed"):
        parse_labelme(doc(modality, [shape(label, reason="x")]), modality)


def test_non_polygon_rejected():
    with pytest.raises(LabelmeError, match="polygon"):
        parse_labelme(doc(shapes=[shape(shape_type="rectangle")]), "EO")


@pytest.mark.parametrize("pts", [[[0, 0], [1, 1]], [[0, 0]], []])
def test_too_few_points(pts):
    with pytest.raises(LabelmeError, match="at least 3"):
        parse_labelme(doc(shapes=[shape(points=pts)]), "EO")


def test_bad_points():
    for pts in ([[0, 0], [1, 1], [1]], [[0, 0], [1, 1], ["a", 2]], [[0, 0], [1, 1], [float("nan"), 2]], [[0, 0], [1, 1], [5000, 2]]):
        with pytest.raises(LabelmeError):
            parse_labelme(doc(shapes=[shape(points=pts)]), "EO")


def test_tiny_float_overshoot_tolerated():
    parse_labelme(doc(shapes=[shape(points=[[-5e-13, 0], [100, 0], [100, 1080.0000000000005]])]), "EO")


def test_size_mismatch_with_declared_and_actual():
    with pytest.raises(LabelmeError, match="expected"):
        parse_labelme(doc("EO", imageWidth=640, imageHeight=512), "EO")
    with pytest.raises(LabelmeError, match="image is"):
        parse_labelme(doc("EO"), "EO", actual_size=(1280, 720))


def test_image_data_must_be_null():
    with pytest.raises(LabelmeError, match="imageData"):
        parse_labelme(doc(imageData="abc"), "EO")


def test_missing_keys_and_reason_rules():
    d = doc()
    del d["imageWidth"]
    with pytest.raises(LabelmeError, match="missing"):
        parse_labelme(d, "EO")
    s = shape()
    del s["reason"]
    with pytest.raises(LabelmeError, match="reason"):
        parse_labelme(doc(shapes=[s]), "EO")
    with pytest.raises(LabelmeError, match="non-empty"):
        parse_labelme(doc(shapes=[shape("crack", reason="")]), "EO")
    with pytest.raises(LabelmeError, match="non-empty"):
        parse_labelme(doc(shapes=[shape("crack", reason=None)]), "EO")
    with pytest.raises(LabelmeError, match="must be null"):
        parse_labelme(doc(shapes=[shape(reason="oops")]), "EO")
    with pytest.raises(LabelmeError, match="must be null"):
        parse_labelme(doc(shapes=[shape(reason="")]), "EO")


def test_input_not_mutated():
    d = doc()
    snap = copy.deepcopy(d)
    parse_labelme(d, "EO")
    assert d == snap
