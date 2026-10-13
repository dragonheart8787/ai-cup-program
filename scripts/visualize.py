"""Draw GT polygons (train) or the bare image (test) into outputs/vis/.

Examples:
  python scripts/visualize.py --split train --modality EO --pick crack
  python scripts/visualize.py --split train --modality IR --sample train_0001
  python scripts/visualize.py --split train --modality EO --random 4 --seed 0
  python scripts/visualize.py --split test --modality IR --sample test_001
Titles are ASCII only (reasons are not drawn).
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from src.dataset.labelme import load_labelme  # noqa: E402
from src.dataset.paths import MODALITIES, DatasetPaths, resolve_repo_path  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger("visualize")
COLORS = {"solar_panel": (0, 255, 0), "crack": (255, 0, 0), "hotspot": (255, 160, 0)}


def render(image_path: Path, modality: str, label_path: Path | None, out: Path, max_side: int = 1280) -> Path:
    im = Image.open(image_path).convert("RGB")
    size = im.size
    counts: Counter[str] = Counter()
    d = ImageDraw.Draw(im)
    if label_path is not None:
        doc = load_labelme(label_path, modality, actual_size=size)
        for s in doc.shapes:
            counts[s.label] += 1
            pts = list(s.points)
            d.line(pts + [pts[0]], fill=COLORS[s.label], width=2 if s.label == "solar_panel" else max(4, size[0] // 300))
        title = f"{image_path.stem} {modality} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    else:
        title = f"{image_path.stem} {modality} (no GT)"
    bar = max(24, size[1] // 30)
    d.rectangle([0, 0, size[0], bar], fill=(0, 0, 0))
    d.text((6, bar // 4), title, fill=(255, 255, 255))
    scale = min(1.0, max_side / max(size))
    if scale < 1.0:
        im = im.resize((int(size[0] * scale), int(size[1] * scale)), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, optimize=True)
    return out


def pick_sample(meta: pd.DataFrame, modality: str, kind: str) -> str:
    g = meta[(meta.split == "train") & (meta.modality == modality)].copy()
    for c in ("n_panels", "n_defects"):
        g[c] = g[c].astype(int)
    if kind in ("crack", "hotspot"):
        g = g[(g.defect_label == kind) & (g.n_panels <= 10)]
    elif kind == "none":
        g = g[(g.n_defects == 0) & (g.n_panels >= 5)]
    elif kind == "many":
        g = g[g.n_panels > 30]
    else:
        raise ValueError(f"unknown --pick {kind!r}")
    if g.empty:
        raise LookupError(f"no sample matches --pick {kind} for {modality}")
    return str(g.sort_values("pair_id").iloc[0].pair_id)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--modality", choices=MODALITIES, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", help="stem, e.g. train_0001")
    g.add_argument("--pick", choices=("crack", "hotspot", "none", "many"), help="first matching train sample (needs metadata.csv)")
    g.add_argument("--random", type=int, metavar="N")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    setup_logging("INFO")
    cfg = load_config(resolve_repo_path(args.config))
    paths = DatasetPaths.from_config(cfg)
    out_dir = resolve_repo_path(cfg["visualize"]["out_dir"])
    if args.sample:
        stems = [args.sample]
    elif args.pick:
        if args.split != "train":
            raise SystemExit("--pick needs --split train")
        meta = pd.read_csv(resolve_repo_path(cfg["audit"]["metadata_csv"]), dtype={"pair_id": str})
        stems = [pick_sample(meta, args.modality, args.pick)]
    else:
        all_stems = paths.list_stems(args.split, args.modality)
        rng = np.random.default_rng(args.seed)
        stems = sorted(rng.choice(all_stems, size=min(args.random, len(all_stems)), replace=False).tolist())
    for stem in stems:
        img = paths.image_dir(args.split, args.modality) / f"{stem}.jpg"
        if not img.is_file():
            raise FileNotFoundError(img)
        lab = paths.label_dir(args.modality) / f"{stem}.json" if args.split == "train" else None
        out = render(img, args.modality, lab, out_dir / f"{args.split}_{args.modality}_{stem}.png")
        logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
