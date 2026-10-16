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

from src.dataset.closeup import closeup_plan, crop_resize  # noqa: E402
from src.dataset.brightness import match_mean  # noqa: E402
from src.dataset.ir_color import apply_inferno, colorized_to_gray, is_colorized  # noqa: E402
from src.dataset.ir_sim import simulate_colorized_roundtrip  # noqa: E402
from src.dataset.paths import DatasetPaths, repo_root, resolve_repo_path  # noqa: E402
from src.dataset.workroot import work_root  # noqa: E402
from src.dataset.yolo_export import CLASS_NAMES, build_full_items, build_items  # noqa: E402
from src.utils.config import config_hash, load_config  # noqa: E402
from src.utils.experiments import append_experiment, file_sha256, get_git_commit  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

logger = logging.getLogger("predict")


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def make_loader(path: Path, modality: str, thr: float, ir_sim: bool, gain: float | None = None, gamma: float | None = None, crop=None,
                match_mean_target: float | None = None, ir_mode: str = "auto", record: dict | None = None, key: str = ""):
    """ir_mode: auto (colorized -> gray via inferno LUT, the submission pipeline) | raw (feed the colorized RGB as is) |
    lut_roundtrip (gray -> inferno -> inverse LUT) | inferno_raw (gray -> inferno RGB, no inverse). Parameters go to ``record[key]``."""
    def load() -> np.ndarray:
        rgb = _load_rgb(path)
        info: dict = {}
        if crop is not None:  # close-up proxy: center crop zoomed back to full size
            rgb = crop_resize(rgb, crop)
        if gain is not None:  # brightness-shift robustness test (val only): multiply, clip to 255
            rgb = np.clip(rgb.astype(np.float64) * gain, 0, 255).round().astype(np.uint8)
        if gamma is not None:  # out = 255 * (x / 255) ** gamma
            rgb = np.round(255.0 * (rgb.astype(np.float64) / 255.0) ** gamma).astype(np.uint8)
        if match_mean_target is not None:  # per-image gamma so that the mean gray becomes the target (own pixels only)
            rgb, res = match_mean(rgb, match_mean_target)
            info.update(gamma=round(res.gamma, 5), mean_before=round(res.mean_before, 3), mean_after=round(res.mean_after, 3), clamped=res.clamped)
        if modality == "IR":
            colorized = is_colorized(rgb, thr)
            info["colorized"] = bool(colorized)
            if ir_sim:
                rgb = np.repeat(simulate_colorized_roundtrip(rgb)[..., None], 3, axis=2)
            elif ir_mode == "auto":
                if colorized:
                    rgb = np.repeat(colorized_to_gray(rgb)[..., None], 3, axis=2)
                    info["converted"] = "inferno_lut_inverse"
            elif ir_mode == "raw":
                info["converted"] = "none (raw RGB fed to the model)"
            elif ir_mode == "lut_roundtrip":
                g = rgb[..., 0]
                back = colorized_to_gray(apply_inferno(g))
                info["lut_roundtrip_abs_err_mean"] = round(float(np.abs(back.astype(int) - g.astype(int)).mean()), 4)
                info["lut_roundtrip_exact_share"] = round(float((back == g).mean()), 4)
                rgb = np.repeat(back[..., None], 3, axis=2)
            elif ir_mode == "inferno_raw":
                rgb = apply_inferno(rgb[..., 0])
            else:
                raise ValueError(f"unknown ir_mode {ir_mode!r}")
        if record is not None:
            record[key] = info
        return rgb
    return load


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", choices=("val", "test", "train_all"), default="val")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--fold", type=int, help="override the fold in the config")
    ap.add_argument("--ir-sim", action="store_true")
    ap.add_argument("--closeup", action="store_true", help="val-only: 30%% of images center-cropped (area 0.15-0.35) and zoomed back")
    ap.add_argument("--gain", type=float, help="val-only robustness test: multiply pixel values")
    ap.add_argument("--gamma", type=float, help="val-only robustness test: gamma curve on pixel values")
    ap.add_argument("--match-mean", type=float, help="per-image gamma so that the mean gray level becomes this value (own pixels only)")
    ap.add_argument("--ir-mode", choices=("auto", "raw", "lut_roundtrip", "inferno_raw"), default="auto",
                    help="auto = colorized test images mapped back to gray (submission pipeline); raw = colorized RGB fed as is; "
                         "lut_roundtrip / inferno_raw = val diagnostics (gray -> inferno [-> inverse])")
    ap.add_argument("--default-config", default="configs/default.yaml")
    args = ap.parse_args()
    setup_logging("INFO")
    base = load_config(resolve_repo_path(args.default_config))
    cfg = load_config(resolve_repo_path(args.config))
    if args.fold is not None:
        cfg["fold"] = args.fold
    mod, fold = cfg["modality"], int(cfg["fold"])
    if args.ir_sim and (mod != "IR" or args.target != "val"):
        raise SystemExit("--ir-sim is only for IR validation")
    if (args.gain or args.gamma or args.closeup) and args.target != "val":
        raise SystemExit("--gain/--gamma/--closeup are for validation images only")
    if args.ir_mode in ("lut_roundtrip", "inferno_raw") and (mod != "IR" or args.target != "val"):
        raise SystemExit("--ir-mode lut_roundtrip/inferno_raw are for IR validation only")
    if args.ir_mode == "raw" and mod != "IR":
        raise SystemExit("--ir-mode raw is for IR only")
    import torch
    from ultralytics import YOLO

    seed_everything(int(cfg["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    paths = DatasetPaths.from_config(base)
    thr = float(base["ir_color"]["colorized_threshold"])
    tparams: dict = {}
    mm, irm = args.match_mean, args.ir_mode
    if args.target == "train_all":  # sanity check of a full-data model on its own training images (NOT a generalisation metric)
        folds = pd.read_csv(resolve_repo_path(base["folds"]["folds_csv"]))
        items = [(it.stem, make_loader(it.image, mod, thr, False, None, None, None, mm, irm, tparams, it.stem)) for it in build_full_items(paths, folds, mod)]
    elif args.target == "val":
        folds = pd.read_csv(resolve_repo_path(base["folds"]["folds_csv"]))
        sel = [it for it in build_items(paths, folds, mod, fold) if it.split == "val"]
        plan = closeup_plan([it.stem for it in sel], int(cfg["seed"])) if args.closeup else {}
        items = [(it.stem, make_loader(it.image, mod, thr, args.ir_sim, args.gain, args.gamma, plan.get(it.stem), mm, irm, tparams, it.stem))
                 for it in sel]
    else:
        items = [(f"test/{mod}/image/{s}.jpg", make_loader(paths.image_dir("test", mod) / f"{s}.jpg", mod, thr, False, None, None, None, mm, irm, tparams,
                                         f"test/{mod}/image/{s}.jpg"))
                 for s in paths.list_stems("test", mod)]
    model = YOLO(args.weights)
    from src.models.yolo_infer import predict_raw

    raw = predict_raw(model, items, CLASS_NAMES[mod], imgsz=int(cfg["imgsz"]), conf=args.conf, max_det=int(cfg["max_det"]),
                      batch=int(cfg["batch"]), device=cfg["device"])
    for k, v in raw.items():  # transformation parameters of every image are stored with its predictions
        v["transform"] = tparams.get(k, {})
    meta = {"modality": mod, "fold": fold, "target": args.target, "weights": args.weights, "weights_sha256": file_sha256(args.weights),
            "conf": args.conf, "imgsz": cfg["imgsz"], "ir_sim": args.ir_sim, "closeup": args.closeup, "gain": args.gain, "gamma": args.gamma, "match_mean": args.match_mean, "ir_mode": args.ir_mode, "config_hash": config_hash(cfg),
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
        "notes": f"predict {args.target} tag={args.tag} conf>={args.conf} ir_sim={args.ir_sim} gain={args.gain} gamma={args.gamma} match_mean={args.match_mean} ir_mode={args.ir_mode} instances={meta['n_instances']} cache={out.name}",
    })
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
