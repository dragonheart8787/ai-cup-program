"""Comparison submissions v5a / v5b / v5c built from cached test predictions (no inference), each changing ONE defect rule of v2.

v2 base: old fold-0 models, approx4, EO crack >= 0.65, IR hotspot >= 0.75, panels EO >= 0.80 / IR >= 0.85, panels overlapping a kept
defect with IoU > 0.5 dropped, placeholder reasons.
  v5a_eo010   : EO crack >= 0.10
  v5b_ir030   : IR hotspot >= 0.30
  v5c_ir_top2 : IR hotspot = (all boxes >= 0.75) U (per image the two highest-confidence boxes that are >= 0.30)
  v5d_ir090   : IR hotspot >= 0.90 (a HIGHER threshold: defect boxes disappear and the panel boxes they suppressed come back)
Because the panel/defect exclusion rule is re-applied, EO (v5a) or IR (v5b, v5c) solar_panel boxes that overlap a NEWLY added defect are
removed; scripts/compare_submissions.py verifies that this is the only other difference. Files are written next to the other submissions,
never over an existing file. Reason strings are the placeholders of v2/v4a (checked to be identical).

Usage: python scripts/make_v5_variants.py --out-dir C:\\aicup_scratch --test-images <test_images_local.csv>
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

VARIANTS = {"v5a_eo010": ("EO", "conf>=0.10"), "v5b_ir030": ("IR", "conf>=0.30"), "v5c_ir_top2": ("IR", "conf>=0.75 or top-2 with conf>=0.30"),
            "v5d_ir090": ("IR", "conf>=0.90")}


def pick(defects: list, rule: str, base_conf: float) -> list:
    if rule == "v2":
        return [b for b in defects if b.conf >= base_conf]
    if rule == "conf>=0.10":
        return [b for b in defects if b.conf >= 0.10]
    if rule == "conf>=0.90":
        return [b for b in defects if b.conf >= 0.90]
    if rule == "conf>=0.30":
        return [b for b in defects if b.conf >= 0.30]
    if rule == "conf>=0.75 or top-2 with conf>=0.30":
        keep = {id(b) for b in defects if b.conf >= base_conf}
        keep |= {id(b) for b in sorted(defects, key=lambda b: -b.conf)[:2] if b.conf >= 0.30}
        return [b for b in defects if id(b) in keep]
    raise ValueError(rule)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-images", required=True)
    ap.add_argument("--v2", help="existing v2 file (for the reason-string check)")
    ap.add_argument("--v4a", help="existing v4a file (for the reason-string check)")
    ap.add_argument("--only", nargs="+", choices=sorted(VARIANTS), help="build only these variants (default: all; existing files are never overwritten)")
    args = ap.parse_args()
    base = load_config(resolve_repo_path("configs/default.yaml"))
    sub = load_config(resolve_repo_path("configs/submission.yaml"))  # v2 thresholds
    paths = DatasetPaths.from_config(base)
    cd = work_root(base) / "cache"
    expected = [r["filename"] for r in csv.DictReader(open(args.test_images, encoding="utf-8", newline=""))]
    print("v2 thresholds (configs/submission.yaml): " + ", ".join(f"{m}: defect >= {sub[m]['conf']}, panel >= {sub[m]['panel_conf']}" for m in ("EO", "IR")))
    reasons = {m: placeholder_reason(paths, m) for m in ("EO", "IR")}
    for name in (args.v2, args.v4a):  # reason strings of the existing files must equal the ones used here
        if name:
            d = json.loads(Path(name).read_text(encoding="utf-8"))
            seen = {(k.split("/")[1], o["label"]): o["reason"] for k, v in d.items() for o in v if "reason" in o}
            diff = {k: v for k, v in seen.items() if reasons[k[0]] != v}
            print(f"reason strings in {Path(name).name}: {'identical to the placeholders used here' if not diff else 'DIFFERENT: ' + str(diff)}")
    caches = {"EO": json.loads((cd / "EO_fold0_test_run1.json").read_text(encoding="utf-8"))["images"],
              "IR": json.loads((cd / "IR_fold0_test_run1.json").read_text(encoding="utf-8"))["images"]}
    out_dir = Path(args.out_dir)
    for vname, (changed_mod, rule) in VARIANTS.items():
        if args.only and vname not in args.only:
            continue
        out = out_dir / f"submission_{vname}.json"
        if out.exists():
            raise FileExistsError(f"{out} exists; refusing to overwrite")
        by_key: dict[str, list] = {}
        for mod in ("EO", "IR"):
            s = sub[mod]
            for key, entry in caches[mod].items():
                boxes = raw_to_boxes(entry, s["method"], min(0.30, 0.10, s["conf"]) if mod == changed_mod else s["conf"])
                allp = raw_to_boxes(entry, s["method"], s["panel_conf"])
                defects = [b for b in boxes if b.label != "solar_panel"]
                kept = pick(defects, rule if mod == changed_mod else "v2", s["conf"])
                panels = [b for b in allp if b.label == "solar_panel"]
                by_key[key] = build_entries_with_panels(kept, panels, mod, reasons[mod], entry["w"], entry["h"])
        write_submission(out, by_key, expected)
        rep = validate_file(out, allow_panel=True)
        chk = subprocess.run([sys.executable, str(paths.make_submission_py), "--test-images", args.test_images, "--check", str(out)],
                             capture_output=True, text=True, encoding="utf-8")
        d = json.loads(out.read_text(encoding="utf-8"))
        c = Counter((k.split("/")[1], o["label"]) for k, v in d.items() for o in v)
        print(f"\n### {vname} ({changed_mod}: {rule})\n- file {out} | {out.stat().st_size:,} bytes | sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
        print(f"- official --check exit {chk.returncode}: {chk.stdout.strip().splitlines()[0] if chk.stdout.strip() else chk.stderr.strip()[:200]}")
        print(f"- own polygon validation: {rep}")
        print("- objects: " + ", ".join(f"{m} {l} {n}" for (m, l), n in sorted(c.items())))
        if chk.returncode != 0 or {k: v for k, v in rep.items() if k != '_objects'}:
            raise RuntimeError(f"{vname} failed validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
