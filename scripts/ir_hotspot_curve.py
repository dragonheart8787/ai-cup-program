"""IR hotspot box counts by confidence threshold on the cached TEST predictions (same cache as v2: IR_fold0_test_run1.json).

Counts are post-processed (approx4) boxes, exactly the objects that would enter a submission. Colorized / gray test images are the groups
of src/dataset/ir_color.py:is_colorized (colorized images were mapped back to gray before inference).
Usage: python scripts/ir_hotspot_curve.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.dataset.paths import resolve_repo_path  # noqa: E402
from src.dataset.workroot import work_root  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.utils.config import load_config  # noqa: E402

THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95)


def main() -> int:
    base = load_config(resolve_repo_path("configs/default.yaml"))
    imgs = json.loads((work_root(base) / "cache" / "IR_fold0_test_run1.json").read_text(encoding="utf-8"))["images"]
    meta = pd.read_csv(resolve_repo_path(base["audit"]["metadata_csv"]), dtype={"pair_id": str})
    t = meta[(meta.split == "test") & (meta.modality == "IR")]
    flag = t.is_colorized.astype("boolean").fillna(False).astype(bool)
    grp = {f"test/IR/image/{p}.jpg": ("colorized" if c else "gray") for p, c in zip(t.pair_id, flag)}
    n = {g: sum(1 for v in grp.values() if v == g) for g in ("colorized", "gray")}
    boxes = {k: [b for b in raw_to_boxes(e, "approx4", 0.05) if b.label == "hotspot"] for k, e in imgs.items()}
    print(f"## IR hotspot boxes on the test set by threshold (cache IR_fold0_test_run1.json; {len(imgs)} images: {n['colorized']} colorized, {n['gray']} gray)\n")
    print("| conf >= | boxes (all) | images with >=1 box | boxes colorized | boxes gray | images colorized | images gray |\n|---|---|---|---|---|---|---|")
    for th in THRESHOLDS:
        kept = {k: [b for b in v if b.conf >= th] for k, v in boxes.items()}
        row = {g: (sum(len(v) for k, v in kept.items() if grp[k] == g), sum(1 for k, v in kept.items() if grp[k] == g and v)) for g in ("colorized", "gray")}
        print(f"| {th:.2f} | {sum(len(v) for v in kept.values())} | {sum(1 for v in kept.values() if v)} | {row['colorized'][0]} | {row['gray'][0]} | {row['colorized'][1]} | {row['gray'][1]} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
