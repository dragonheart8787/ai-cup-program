"""IR test diagnostics from cached predictions: hotspot boxes per image at a confidence threshold.

* distribution of hotspot boxes per image (0, 1, 2, 3+)
* for images with >= 3 boxes: is each extra box adjacent to the highest-confidence box (touching / within a small gap)?
* montage of the first N such images (most boxes first), boxes labelled with confidence. Diagnosis only.

Usage: python scripts/diagnose_ir_boxes.py --cache <IR test cache> --conf 0.30 --n 6 --out ir_multibox.png
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--panel-conf", type=float, default=0.85)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="ir_multibox.png")
    ap.add_argument("--method", default="approx4")
    args = ap.parse_args()
    base = load_config(resolve_repo_path("configs/default.yaml"))
    paths = DatasetPaths.from_config(base)
    imgs = json.loads(Path(args.cache).read_text(encoding="utf-8"))["images"]
    per: dict[str, list] = {}
    for k, e in imgs.items():
        per[k] = [b for b in raw_to_boxes(e, args.method, args.conf) if b.label == "hotspot"]
    dist = Counter(min(len(v), 3) for v in per.values())
    print(f"## IR hotspot boxes per image at conf >= {args.conf} (cache {Path(args.cache).name}, {len(per)} images, PROXY/diagnosis)\n")
    print("| boxes per image | images |\n|---|---|")
    for n in (0, 1, 2, 3):
        print(f"| {'3+' if n == 3 else n} | {dist.get(n, 0)} |")
    multi = sorted((k for k, v in per.items() if len(v) >= 3), key=lambda k: (-len(per[k]), k))
    print(f"\nimages with >= 3 boxes: {len(multi)}; total boxes in them: {sum(len(per[k]) for k in multi)}")
    rel = Counter()
    top_conf, other_conf = [], []
    for k in multi:
        bs = sorted(per[k], key=lambda b: -b.conf)
        top = Polygon(bs[0].points).buffer(0)
        top_conf.append(bs[0].conf)
        diag = np.sqrt(top.area)
        for b in bs[1:]:
            other_conf.append(b.conf)
            d = top.distance(Polygon(b.points).buffer(0))
            rel["adjacent (gap < 5% of top-box size)" if d < 0.05 * diag else "not adjacent"] += 1
    if multi:
        print("\n| extra box (not the highest-confidence box) relative to the top box | count |\n|---|---|")
        for k, v in rel.items():
            print(f"| {k} | {v} |")
        print(f"\nconfidence of the top box in those images: median {np.median(top_conf):.2f}; of the extra boxes: median {np.median(other_conf):.2f}")
    sel = multi[: args.n]
    tiles = []
    for k in sel:
        stem = Path(k).stem
        im = Image.open(paths.image_dir("test", "IR") / f"{stem}.jpg").convert("RGB")
        d = ImageDraw.Draw(im)
        for b in raw_to_boxes(imgs[k], args.method, args.panel_conf):
            if b.label == "solar_panel":
                pts = [tuple(p) for p in b.points]
                d.line(pts + [pts[0]], fill=(0, 160, 0), width=1)
        for b in per[k]:
            pts = [tuple(p) for p in b.points]
            d.line(pts + [pts[0]], fill=(255, 0, 255), width=3)
            d.text((pts[0][0] + 3, pts[0][1] + 3), f"{b.conf:.2f}", fill=(255, 255, 0))
        d.rectangle([0, 0, im.width, 14], fill=(0, 0, 0))
        d.text((3, 2), f"{stem}  hotspot boxes={len(per[k])}", fill=(255, 255, 255))
        tiles.append(im)
        print(f"montage: {stem} boxes={len(per[k])} confs={[round(b.conf, 2) for b in sorted(per[k], key=lambda b: -b.conf)]}")
    if tiles:
        cols = 3
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 640, rows * 512))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * 640, (i // cols) * 512))
        out = resolve_repo_path(base.get("visualize", {}).get("out_dir", "outputs/vis")) / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out, optimize=True)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
