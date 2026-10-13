"""Cache raw predictions (class, confidence, mask contour) so thresholds / post-processing can be tuned without re-inference.

  --target val   : fold-k validation images of the train set (default)
  --target test  : test images (colorized IR test images are mapped back to gray first)
  --ir-sim       : (val, IR only) simulate colorization first: min-max -> inferno -> JPEG q90 -> inverse inferno

Usage: python scripts/predict_oof.py --config configs/yolo_ir.yaml --weights checkpoints/IR_fold0.pt --tag base
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.dataset.ir_color import colorized_to_gray, is_colorized  # noqa: E402
from src.dataset.ir_sim import simulate_colorized_roundtrip  # noqa: E402
from src.dataset.paths import DatasetPaths, repo_root, resolve_repo_path  # noqa: E402
from src.dataset.workroot import work_root  # noqa: E402
from src.dataset.yolo_export import CLASS_NAMES, build_items  # noqa: E402
from src.utils.config import config_hash, load_config  # noqa: E402
from src.utils.experiments import append_experiment, file_sha256, get_git_commit  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

logger = logging.getLogger("predict")


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def make_loader(path: Path, modality: str, thr: float, ir_sim: bool, gain: float | None = None, gamma: float | None = None):
    def load() -> np.ndarray:
        rgb = _load_rgb(path)
        if gain is not None:  # brightness-shift robustness test (val only): multiply, clip to 255
            rgb = np.clip(rgb.astype(np.float64) * gain, 0, 255).round().astype(np.uint8)
        if gamma is not None:  # out = 255 * (x / 255) ** gamma
            rgb = np.round(255.0 * (rgb.astype(np.float64) / 255.0) ** gamma).astype(np.uint8)
        if modality == "IR":
            if ir_sim:
                rgb = np.repeat(simulate_colorized_roundtrip(rgb)[..., None], 3, axis=2)
            elif is_colorized(rgb, thr):
                rgb = np.repeat(colorized_to_gray(rgb)[..., None], 3, axis=2)
        return rgb
    return load


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", choices=("val", "test"), default="val")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--ir-sim", action="store_true")
    ap.add_argument("--gain", type=float, help="val-only robustness test: multiply pixel values")
    ap.add_argument("--gamma", type=float, help="val-only robustness test: gamma curve on pixel values")
    ap.add_argument("--default-config", default="configs/default.yaml")
    args = ap.parse_args()
    setup_logging("INFO")
    base = load_config(resolve_repo_path(args.default_config))
    cfg = load_config(resolve_repo_path(args.config))
    mod, fold = cfg["modality"], int(cfg["fold"])
    if args.ir_sim and (mod != "IR" or args.target != "val"):
        raise SystemExit("--ir-sim is only for IR validation")
    if (args.gain or args.gamma) and args.target != "val":
        raise SystemExit("--gain/--gamma are for validation images only")
    import torch
    from ultralytics import YOLO

    seed_everything(int(cfg["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    paths = DatasetPaths.from_config(base)
    thr = float(base["ir_color"]["colorized_threshold"])
    if args.target == "val":
        folds = pd.read_csv(resolve_repo_path(base["folds"]["folds_csv"]))
        sel = [it for it in build_items(paths, folds, mod, fold) if it.split == "val"]
        items = [(it.stem, make_loader(it.image, mod, thr, args.ir_sim, args.gain, args.gamma)) for it in sel]
    else:
        items = [(f"test/{mod}/image/{s}.jpg", make_loader(paths.image_dir("test", mod) / f"{s}.jpg", mod, thr, False))
                 for s in paths.list_stems("test", mod)]
    model = YOLO(args.weights)
    from src.models.yolo_infer import predict_raw

    raw = predict_raw(model, items, CLASS_NAMES[mod], imgsz=int(cfg["imgsz"]), conf=args.conf, max_det=int(cfg["max_det"]),
                      batch=int(cfg["batch"]), device=cfg["device"])
    meta = {"modality": mod, "fold": fold, "target": args.target, "weights": args.weights, "weights_sha256": file_sha256(args.weights),
            "conf": args.conf, "imgsz": cfg["imgsz"], "ir_sim": args.ir_sim, "gain": args.gain, "gamma": args.gamma, "config_hash": config_hash(cfg),
            "n_images": len(raw), "n_instances": sum(len(v["preds"]) for v in raw.values())}
    out = work_root(base) / "cache" / f"{mod}_fold{fold}_{args.target}_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "images": raw}), encoding="utf-8")
    logger.info("cached %s", out)
    append_experiment(resolve_repo_path(base["paths"]["experiments_csv"]), {
        "git_commit": get_git_commit(repo_root()), "config_hash": config_hash(cfg),
        "dataset_version": f"folds.csv:{file_sha256(resolve_repo_path(base['folds']['folds_csv']))[:12]}",
        "fold": fold, "seed": cfg["seed"], "model": "yolo11s-seg", "checkpoint_sha256": meta["weights_sha256"],
        "input_modality": mod, "resolution": cfg["imgsz"], "postprocess": "raw-cache",
        "notes": f"predict {args.target} tag={args.tag} conf>={args.conf} ir_sim={args.ir_sim} gain={args.gain} gamma={args.gamma} instances={meta['n_instances']} cache={out.name}",
    })
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
