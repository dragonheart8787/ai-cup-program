import json

import numpy as np
import pytest

from src.metrics.proxy import Box, evaluate, polygon_iou
from src.postprocess.pipeline import raw_to_boxes
from src.postprocess.quad import approx4, finalize, minrect, to_polygon
from src.submission.reasons import most_central_sentence
from src.submission.writer import build_entries, write_submission

SQ = np.array([[10, 10], [110, 12], [108, 90], [9, 88]], dtype=float)


def test_quad_methods_on_a_rounded_quad():
    mask_contour = np.array([[10, 10], [60, 9], [110, 12], [109, 50], [108, 90], [60, 91], [9, 88], [10, 50]], dtype=float)
    assert minrect(mask_contour).shape == (4, 2)
    q = approx4(mask_contour)
    assert q is not None and len(q) == 4
    assert polygon_iou(q, SQ) > 0.9 and polygon_iou(minrect(mask_contour), SQ) > 0.9
    with pytest.raises(ValueError):
        to_polygon(mask_contour, "nope")


def test_triangle_has_no_4_point_approx():
    assert approx4(np.array([[0, 0], [100, 0], [50, 80]], dtype=float)) is None


def test_finalize_clips_and_rejects():
    poly, issues = finalize(np.array([[-5, -5], [50, -5], [50, 40], [-5, 40]], dtype=float), 100, 100)
    assert issues == [] and poly.min() >= 0
    bow = np.array([[0, 0], [10, 10], [10, 0], [0, 10]], dtype=float)
    assert finalize(bow, 100, 100)[0] is None
    assert finalize(None, 100, 100)[1] == ["no_polygon"]
    line = np.array([[0, 0], [5, 5], [10, 10], [20, 20]], dtype=float)
    assert finalize(line, 100, 100)[0] is None


def test_raw_to_boxes_threshold_and_drop():
    from collections import Counter

    entry = {"w": 200, "h": 200, "preds": [
        {"label": "crack", "conf": 0.9, "poly": SQ.tolist()},
        {"label": "crack", "conf": 0.1, "poly": SQ.tolist()},
        {"label": "crack", "conf": 0.9, "poly": [[1, 1], [2, 2]]},
    ]}
    drop = Counter()
    boxes = raw_to_boxes(entry, "minrect", 0.5, drop)
    assert len(boxes) == 1 and drop["contour<3"] == 1


def test_proxy_matching():
    gt = {"a": [Box("solar_panel", SQ), Box("crack", SQ + 300)]}
    perfect = {"a": [Box("solar_panel", SQ, 0.9), Box("crack", SQ + 300, 0.8)]}
    r = evaluate(perfect, gt, 0.75)
    assert r["all_f1"] == 1.0 and r["panel_f1"] == 1.0 and r["defect_f1"] == 1.0
    wrong_class = {"a": [Box("hotspot", SQ + 300, 0.8)]}
    r = evaluate(wrong_class, gt, 0.5)
    assert r["defect_tp"] == 0 and r["defect_fp"] == 1 and r["defect_fn"] == 1
    dup = {"a": [Box("solar_panel", SQ, 0.9), Box("solar_panel", SQ, 0.8)]}
    r = evaluate(dup, gt, 0.5)
    assert r["panel_tp"] == 1 and r["panel_fp"] == 1  # one-to-one
    with pytest.raises(KeyError):
        evaluate({"zzz": []}, gt, 0.5)


def test_writer_format_and_rules(tmp_path):
    boxes = [Box("solar_panel", SQ, 0.9), Box("crack", SQ, 0.8)]
    ent = build_entries(boxes, "EO", "占位原因", 1920, 1080)
    assert len(ent) == 1 and ent[0]["label"] == "crack" and "confidence" not in ent[0] and len(ent[0]["points"]) == 4
    with pytest.raises(ValueError):
        build_entries(boxes, "EO", "  ", 1920, 1080)
    with pytest.raises(ValueError):
        build_entries([Box("hotspot", SQ)], "EO", "x", 1920, 1080)
    with pytest.raises(ValueError):
        build_entries([Box("crack", SQ + 5000)], "EO", "x", 1920, 1080)
    keys = ["test/EO/image/test_001.jpg", "test/EO/image/test_002.jpg"]
    p = tmp_path / "s.json"
    write_submission(p, {keys[0]: ent}, keys)
    first = p.read_bytes()
    write_submission(p, {keys[0]: ent}, keys)
    assert p.read_bytes() == first
    d = json.loads(first.decode("utf-8"))
    assert d[keys[1]] == [] and d[keys[0]][0]["reason"] == "占位原因"
    with pytest.raises(KeyError):
        write_submission(p, {"other": []}, keys)


def test_most_central_sentence_deterministic():
    t = ["面板破裂導電受損", "面板破裂導電受損", "模組板破裂影響導電", "完全不同的句子內容"]
    a = most_central_sentence(t)
    assert a == most_central_sentence(list(reversed(t))) and a in t
    with pytest.raises(ValueError):
        most_central_sentence(["", " "])
