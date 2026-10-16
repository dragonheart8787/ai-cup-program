import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from make_v5_variants import pick  # noqa: E402

from src.metrics.proxy import Box  # noqa: E402

SQ = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)


def boxes(*confs):
    return [Box("hotspot", SQ + i * 20, c) for i, c in enumerate(confs)]


def test_pick_rules():
    b = boxes(0.9, 0.8, 0.6, 0.5, 0.35, 0.2)
    assert [x.conf for x in pick(b, "v2", 0.75)] == [0.9, 0.8]
    assert [x.conf for x in pick(b, "conf>=0.30", 0.75)] == [0.9, 0.8, 0.6, 0.5, 0.35]
    assert [x.conf for x in pick(b, "conf>=0.10", 0.75)] == [0.9, 0.8, 0.6, 0.5, 0.35, 0.2]
    # all boxes >= 0.75 are kept, plus the two best boxes >= 0.30 (which are the two boxes already above 0.75 here)
    assert [x.conf for x in pick(b, "conf>=0.75 or top-2 with conf>=0.30", 0.75)] == [0.9, 0.8]
    c = boxes(0.9, 0.8, 0.7, 0.5, 0.35)
    assert [x.conf for x in pick(c, "conf>=0.75 or top-2 with conf>=0.30", 0.75)] == [0.9, 0.8]
    d = boxes(0.6, 0.5, 0.4, 0.2)
    assert [x.conf for x in pick(d, "conf>=0.75 or top-2 with conf>=0.30", 0.75)] == [0.6, 0.5]


def _sub(path: Path, objs: list[dict]) -> None:
    path.write_text(json.dumps({"test/EO/image/test_001.jpg": objs, "test/IR/image/test_001.jpg": []}), encoding="utf-8")


def test_compare_submissions_detects_unexpected_changes(tmp_path):
    panel = {"label": "solar_panel", "points": [[0, 0], [100, 0], [100, 100], [0, 100]]}
    crack = {"label": "crack", "points": [[200, 0], [300, 0], [300, 100], [200, 100]], "reason": "x"}
    new_crack = {"label": "crack", "points": [[0, 0], [100, 0], [100, 100], [0, 100]], "reason": "x"}  # same place as the panel
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    _sub(a, [panel, crack])
    _sub(b, [crack, new_crack])        # panel replaced by a crack: allowed (exclusion rule)
    _sub(c, [panel])                   # an expected-class object removed: not allowed
    run = lambda new: subprocess.run([sys.executable, str(SCRIPTS / "compare_submissions.py"), "--base", str(a), "--new", str(new), "--changed", "EO:crack"],
                                     capture_output=True, text=True, encoding="utf-8")
    assert run(b).returncode == 0 and "PASS" in run(b).stdout
    assert run(c).returncode == 1


def test_compare_submissions_direction_remove(tmp_path):
    panel = {"label": "solar_panel", "points": [[0, 0], [100, 0], [100, 100], [0, 100]]}
    other = {"label": "solar_panel", "points": [[300, 0], [400, 0], [400, 100], [300, 100]]}
    hot = {"label": "hotspot", "points": [[0, 0], [100, 0], [100, 100], [0, 100]], "reason": "x"}
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    base = lambda objs: {"test/EO/image/test_001.jpg": [], "test/IR/image/test_001.jpg": objs}
    a.write_text(json.dumps(base([hot, other])), encoding="utf-8")
    b.write_text(json.dumps(base([other, panel])), encoding="utf-8")   # hotspot removed, the suppressed panel comes back: allowed
    c.write_text(json.dumps(base([panel])), encoding="utf-8")          # a panel disappeared: not allowed
    run = lambda new: subprocess.run([sys.executable, str(SCRIPTS / "compare_submissions.py"), "--base", str(a), "--new", str(new),
                                      "--changed", "IR:hotspot", "--direction", "remove"], capture_output=True, text=True, encoding="utf-8")
    assert run(b).returncode == 0 and "PASS" in run(b).stdout
    assert run(c).returncode == 1
