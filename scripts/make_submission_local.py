"""Build a submission from cached TEST predictions, validate it, and run the official --check.

Usage:
  python scripts/make_submission_local.py --cache-eo <EO test cache> --cache-ir <IR test cache> \
      --test-images <test_images_local.csv> --out <submission.json> [--config configs/submission.yaml]
reason is a PLACEHOLDER (most central training sentence per class).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.dataset.labelme import IMAGE_SIZE, load_labelme  # noqa: E402
from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.postprocess.validate import validate_polygon  # noqa: E402
from src.submission.reasons import PLACEHOLDER_NOTE, most_central_sentence  # noqa: E402
from src.submission.writer import DEFECTS, build_entries, write_submission  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger("submission")


def placeholder_reason(paths: DatasetPaths, modality: str) -> str:
    texts = []
    for p in sorted(paths.label_dir(modality).glob("*.json")):
        doc = load_labelme(p, modality)
        texts += [s.reason for s in doc.defects]
    return most_central_sentence(texts)


def validate_file(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    bad: Counter = Counter()
    n = 0
    for key, objs in data.items():
        mod = key.split("/")[1]
        w, h = IMAGE_SIZE[mod]
        for o in objs:
            n += 1
            res = validate_polygon(np.asarray(o["points"]), w, h, max_vertices=100)
            for i in res.issues:
                bad[i] += 1
            if not o.get("reason", "").strip():
                bad["empty_reason"] += 1
            if o["label"] != DEFECTS[mod]:
                bad["wrong_label"] += 1
    bad["_objects"] = n
    return dict(bad)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-eo", required=True)
    ap.add_argument("--cache-ir", required=True)
    ap.add_argument("--test-images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/submission.yaml")
    ap.add_argument("--default-config", default="configs/default.yaml")
    args = ap.parse_args()
    setup_logging("INFO")
    base = load_config(resolve_repo_path(args.default_config))
    sub = load_config(resolve_repo_path(args.config))
    paths = DatasetPaths.from_config(base)
    expected = [r["filename"] for r in csv.DictReader(open(args.test_images, encoding="utf-8", newline=""))]
    by_key: dict[str, list[dict]] = {}
    drops: dict[str, Counter] = {}
    for mod, cache_path in (("EO", args.cache_eo), ("IR", args.cache_ir)):
        s = sub[mod]
        if s["method"] is None or s["conf"] is None:
            raise ValueError(f"{args.config}: {mod}.method / {mod}.conf are not set")
        cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        if cache["meta"]["modality"] != mod or cache["meta"]["target"] != "test":
            raise ValueError(f"{cache_path} is not a {mod} test cache")
        reason = placeholder_reason(paths, mod)
        logger.info("%s placeholder reason: %s | %s", mod, reason, PLACEHOLDER_NOTE)
        drops[mod] = Counter()
        for key, entry in cache["images"].items():
            boxes = raw_to_boxes(entry, s["method"], float(s["conf"]), drops[mod])
            by_key[key] = build_entries(boxes, mod, reason, entry["w"], entry["h"])
    write_submission(args.out, by_key, expected)
    rep = validate_file(Path(args.out))
    print("own validation:", rep, "| dropped by post-processing:", {k: dict(v) for k, v in drops.items()})
    problems = {k: v for k, v in rep.items() if k != "_objects"}
    if problems:
        raise RuntimeError(f"own polygon validation failed: {problems}")
    r = subprocess.run([sys.executable, str(paths.make_submission_py), "--test-images", args.test_images, "--check", args.out],
                       capture_output=True, text=True, encoding="utf-8")
    print("official --check exit", r.returncode)
    print(r.stdout + r.stderr)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
