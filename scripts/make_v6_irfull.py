"""v6_irfull: v2 with the IR predictions replaced by the full-data IR model; EO exactly as v2 (same cached EO predictions and settings).

IR uses the v2 recipe unchanged: hotspot >= 0.75, panels >= 0.85, colorized test images mapped to gray first (done at inference time,
see scripts/predict_oof.py --ir-mode auto), v2 placeholder reasons, panel/defect exclusion rule. Never overwrites an existing file.

Usage: python scripts/make_v6_irfull.py --ir-cache <IR_fold0_test_full.json> --out C:\\aicup_scratch\\submission_v6_irfull.json --test-images <csv>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_submission_local import placeholder_reason, validate_file  # noqa: E402

from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.dataset.workroot import work_root  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.submission.writer import build_entries_with_panels, write_submission  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ir-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-images", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite")
    base = load_config(resolve_repo_path("configs/default.yaml"))
    sub = load_config(resolve_repo_path("configs/submission.yaml"))  # v2 thresholds
    paths = DatasetPaths.from_config(base)
    cd = work_root(base) / "cache"
    expected = [r["filename"] for r in csv.DictReader(open(args.test_images, encoding="utf-8", newline=""))]
    caches = {"EO": json.loads((cd / "EO_fold0_test_run1.json").read_text(encoding="utf-8"))["images"],
              "IR": json.loads(Path(args.ir_cache).read_text(encoding="utf-8"))["images"]}
    by_key: dict[str, list] = {}
    for mod in ("EO", "IR"):
        s = sub[mod]
        reason = placeholder_reason(paths, mod)
        for key, entry in caches[mod].items():
            boxes = raw_to_boxes(entry, s["method"], min(s["conf"], s["panel_conf"]))
            defects = [b for b in boxes if b.label != "solar_panel" and b.conf >= s["conf"]]
            panels = [b for b in boxes if b.label == "solar_panel" and b.conf >= s["panel_conf"]]
            by_key[key] = build_entries_with_panels(defects, panels, mod, reason, entry["w"], entry["h"])
    write_submission(out, by_key, expected)
    rep = validate_file(out, allow_panel=True)
    chk = subprocess.run([sys.executable, str(paths.make_submission_py), "--test-images", args.test_images, "--check", str(out)],
                         capture_output=True, text=True, encoding="utf-8")
    d = json.loads(out.read_text(encoding="utf-8"))
    c = Counter((k.split("/")[1], o["label"]) for k, v in d.items() for o in v)
    print(f"file {out} | {out.stat().st_size:,} bytes | sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    print(f"official --check exit {chk.returncode}: {chk.stdout.strip().splitlines()[0] if chk.stdout.strip() else chk.stderr.strip()[:200]}")
    print(f"own polygon validation: {rep}")
    print("objects: " + ", ".join(f"{m} {l} {n}" for (m, l), n in sorted(c.items())))
    if chk.returncode != 0 or {k: v for k, v in rep.items() if k != "_objects"}:
        raise RuntimeError("v6 failed validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
