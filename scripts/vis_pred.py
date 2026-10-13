"""Draw cached test predictions (conf >= --conf) on selected test images as a 2x2 montage -> outputs/vis/.

Selections (test images): --select brightest | closest --target <value> | colorized | gray
Usage: python scripts/vis_pred.py --modality EO --cache <test.json> --select brightest --out pred_eo_brightest.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.utils.config import load_config  # noqa: E402

COL = {"solar_panel": (0, 255, 0), "crack": (255, 0, 0), "hotspot": (255, 0, 255)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", choices=("EO", "IR"), required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--select", choices=("brightest", "closest", "colorized", "gray"), required=True)
    ap.add_argument("--target", type=float)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = load_config(resolve_repo_path("configs/default.yaml"))
    paths = DatasetPaths.from_config(base)
    meta = pd.read_csv(resolve_repo_path(base["audit"]["metadata_csv"]), dtype={"pair_id": str})
    m = meta[(meta.split == "test") & (meta.modality == args.modality)].copy()
    if args.select == "brightest":
        m = m.sort_values("mean_brightness", ascending=False)
    elif args.select == "closest":
        if args.target is None:
            raise SystemExit("--target required")
        m = m.iloc[(m.mean_brightness - args.target).abs().argsort()]
    elif args.select == "colorized":
        m = m[m.is_colorized.astype("boolean").fillna(False)].sort_values("pair_id")
        m = m.iloc[np.linspace(0, len(m) - 1, args.n).astype(int)]
    else:
        m = m[~m.is_colorized.astype("boolean").fillna(False)].sort_values("pair_id")
        m = m.iloc[np.linspace(0, len(m) - 1, args.n).astype(int)]
    rows = m.head(args.n)
    cache = json.loads(Path(args.cache).read_text(encoding="utf-8"))["images"]
    tiles = []
    for _, r in rows.iterrows():
        key = f"test/{args.modality}/image/{r.pair_id}.jpg"
        e = cache[key]
        im = Image.open(r.image_path).convert("RGB")
        d = ImageDraw.Draw(im)
        n_p = n_d = 0
        for p in e["preds"]:
            if p["conf"] < args.conf:
                continue
            pts = [tuple(x) for x in p["poly"]]
            if len(pts) < 3:
                continue
            defect = p["label"] != "solar_panel"
            d.line(pts + [pts[0]], fill=COL[p["label"]], width=max(3, im.width // 300) if defect else 2)
            if defect:
                n_d += 1
                d.text((pts[0][0] + 3, pts[0][1] + 3), f"{p['label']} {p['conf']:.2f}", fill=COL[p["label"]])
            else:
                n_p += 1
        bar = f"{r.pair_id} mean={r.mean_brightness:.0f} colorized={r.is_colorized} panels={n_p} defects={n_d}"
        d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
        d.text((4, 3), bar, fill=(255, 255, 255))
        print(bar)
        tiles.append(im)
    w = 800 if args.modality == "EO" else 640
    tiles = [t.resize((w, int(t.height * w / t.width))) for t in tiles]
    h = tiles[0].height
    sheet = Image.new("RGB", (2 * w, 2 * h))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * w, (i // 2) * h))
    out = resolve_repo_path(base.get("visualize", {}).get("out_dir", "outputs/vis")) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, optimize=True)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
