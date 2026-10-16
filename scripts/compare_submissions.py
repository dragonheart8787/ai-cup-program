"""Compare two submission files object by object (exact point equality) and verify that only the intended class changed.

For every (modality, label): objects in base / new, objects added, objects removed. Objects of a label that is not expected to change
must be identical, EXCEPT solar_panel objects that were removed because they overlap (IoU > 0.5) a newly added defect (the v2 exclusion rule);
every such removal is checked geometrically. Reasons of unchanged defects must be identical.

Usage: python scripts/compare_submissions.py --base v2.json --new v5a.json --changed EO:crack
Exit code 1 if the verification fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.metrics.proxy import polygon_iou  # noqa: E402


def key_of(o: dict) -> tuple:
    return (o["label"], tuple((round(x, 2), round(y, 2)) for x, y in o["points"]), o.get("reason"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--changed", help="e.g. EO:crack or IR:hotspot (the only class allowed to change, plus explained panel removals)")
    ap.add_argument("--ignore-modality", choices=("EO", "IR"), help="verify that EVERY object of the other modality is identical; "
                    "the ignored modality is only counted (it comes from a different model)")
    args = ap.parse_args()
    if bool(args.changed) == bool(args.ignore_modality):
        ap.error("give exactly one of --changed / --ignore-modality")
    cm, cl = args.changed.split(":") if args.changed else ("", "")
    a = json.loads(Path(args.base).read_text(encoding="utf-8"))
    b = json.loads(Path(args.new).read_text(encoding="utf-8"))
    if set(a) != set(b):
        print("FAIL: different key sets")
        return 1
    stats: dict = {}
    ok = True
    for k in sorted(a):
        mod = k.split("/")[1]
        ka = {key_of(o): o for o in a[k]}
        kb = {key_of(o): o for o in b[k]}
        added = [kb[x] for x in kb if x not in ka]
        removed = [ka[x] for x in ka if x not in kb]
        if args.ignore_modality == mod:
            for o in added:
                stats.setdefault((mod, o["label"]), [0, 0, 0, 0, 0])[2] += 1
            for o in removed:
                stats.setdefault((mod, o["label"]), [0, 0, 0, 0, 0])[3] += 1
            continue
        if args.ignore_modality:  # the compared modality must be identical
            for o in added + removed:
                ok = False
                print(f"FAIL: {mod}/{o['label']} differs in {k}")
            continue
        for o in added:
            stats.setdefault((mod, o["label"]), [0, 0, 0, 0, 0])[2] += 1
        for o in removed:
            stats.setdefault((mod, o["label"]), [0, 0, 0, 0, 0])[3] += 1
        added_defects = [o for o in added if o["label"] == cl and mod == cm]
        for o in added:
            if (mod, o["label"]) != (cm, cl):
                ok = False
                print(f"FAIL: unexpected added object {mod}/{o['label']} in {k}")
        for o in removed:
            if o["label"] == "solar_panel" and mod == cm and any(
                    polygon_iou(np.asarray(o["points"]), np.asarray(d["points"])) > 0.5 for d in added_defects):
                stats.setdefault((mod, "solar_panel"), [0, 0, 0, 0, 0])[4] += 1  # explained by the exclusion rule
            elif (mod, o["label"]) == (cm, cl):
                ok = False
                print(f"FAIL: expected-class object removed in {k} (the change must only add boxes here)")
            else:
                ok = False
                print(f"FAIL: unexplained removal {mod}/{o['label']} in {k}")
    for mod in ("EO", "IR"):
        for lab in ("crack", "hotspot", "solar_panel"):
            if (mod == "EO" and lab == "hotspot") or (mod == "IR" and lab == "crack"):
                continue
            s = stats.setdefault((mod, lab), [0, 0, 0, 0, 0])
            s[0] = sum(1 for k, v in a.items() if k.split("/")[1] == mod for o in v if o["label"] == lab)
            s[1] = sum(1 for k, v in b.items() if k.split("/")[1] == mod for o in v if o["label"] == lab)
    print(f"| modality | label | base | new | added | removed | of which panel removals explained by the exclusion rule |\n|---|---|---|---|---|---|---|")
    for (mod, lab), s in sorted(stats.items()):
        print(f"| {mod} | {lab} | {s[0]} | {s[1]} | {s[2]} | {s[3]} | {s[4] if lab == 'solar_panel' else '-'} |")
    what = (f"only {cm}:{cl} changed; every other difference is an explained panel removal" if args.changed
            else f"every {'EO' if args.ignore_modality == 'IR' else 'IR'} object is identical; the {args.ignore_modality} part is only counted")
    print(f"\nverification: {'PASS' if ok else 'FAIL'} ({what})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
