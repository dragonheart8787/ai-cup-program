"""Read-only dataset audit -> data/metadata.csv and data/audit_summary.md.

Fails (exit 1) when the counts differ from the verified numbers below.

Usage: AICUP_DATA_ROOT=<dir> python scripts/audit_dataset.py [--config configs/default.yaml]
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.dataset.images import measure_image  # noqa: E402
from src.dataset.labelme import DEFECT_LABEL, load_labelme  # noqa: E402
from src.dataset.paths import MODALITIES, DatasetPaths, resolve_repo_path  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger("audit")

UNKNOWN = "unknown"
# Verified on the real data (exploration rounds 1-2).
EXPECTED = {
    "train_pairs": 1046,
    "test_pairs": 696,
    "EO_solar_panel": 29553,
    "EO_crack": 352,
    "EO_images_with_crack": 258,
    "IR_solar_panel": 26257,
    "IR_hotspot": 444,
    "IR_images_with_hotspot": 428,
    "test_ir_colorized": 200,
}


def _process(args: tuple[str, str, str, str | None, str, float]) -> dict:
    split, modality, stem, image_path, label_path, thr = args
    rec = measure_image(Path(image_path), modality, thr)
    row: dict = {
        "sample_id": f"{stem}_{modality}", "pair_id": stem, "split": split, "modality": modality,
        "image_path": image_path, "label_path": label_path or "",
        "width": rec.width, "height": rec.height,
        "mean_brightness": round(rec.mean_brightness, 4), "std_brightness": round(rec.std_brightness, 4),
        "is_colorized": rec.is_colorized, "phash": rec.phash,
        "panel_id": UNKNOWN, "site_id": UNKNOWN, "weather": UNKNOWN, "timestamp": UNKNOWN,
    }
    if label_path:
        doc = load_labelme(label_path, modality, actual_size=(rec.width, rec.height))
        row.update(
            labelme_version=doc.version, n_panels=doc.n_panels, n_defects=doc.n_defects,
            defect_label=DEFECT_LABEL[modality] if doc.n_defects else "none",
            has_auto_label=doc.has_auto_label, n_auto_panels=doc.n_auto_panels,
        )
    else:  # test images have no annotation
        row.update(labelme_version=UNKNOWN, n_panels=UNKNOWN, n_defects=UNKNOWN, defect_label=UNKNOWN,
                   has_auto_label=UNKNOWN, n_auto_panels=UNKNOWN)
    return row


def collect_jobs(paths: DatasetPaths, thr: float) -> list[tuple]:
    jobs = []
    for split in ("train", "test"):
        stems = {m: paths.list_stems(split, m) for m in MODALITIES}
        if stems["EO"] != stems["IR"]:
            only = set(stems["EO"]) ^ set(stems["IR"])
            raise RuntimeError(f"{split}: EO and IR stems differ ({len(only)} unpaired, e.g. {sorted(only)[:3]})")
        for m in MODALITIES:
            for stem in stems[m]:
                lab = None
                if split == "train":
                    lp = paths.label_dir(m) / f"{stem}.json"
                    if not lp.is_file():
                        raise FileNotFoundError(f"missing label for {m}/{stem}: {lp}")
                    lab = str(lp)
                jobs.append((split, m, stem, str(paths.image_dir(split, m) / f"{stem}.jpg"), lab, thr))
    return jobs


def check_expected(df: pd.DataFrame) -> list[str]:
    tr = df[df.split == "train"]
    te = df[df.split == "test"]
    eo, ir = tr[tr.modality == "EO"], tr[tr.modality == "IR"]
    got = {
        "train_pairs": min(len(eo), len(ir)),
        "test_pairs": min(int((te.modality == "EO").sum()), int((te.modality == "IR").sum())),
        "EO_solar_panel": int(eo.n_panels.sum()),
        "EO_crack": int(eo.n_defects.sum()),
        "EO_images_with_crack": int((eo.n_defects > 0).sum()),
        "IR_solar_panel": int(ir.n_panels.sum()),
        "IR_hotspot": int(ir.n_defects.sum()),
        "IR_images_with_hotspot": int((ir.n_defects > 0).sum()),
        "test_ir_colorized": int(te[te.modality == "IR"].is_colorized.fillna(False).astype(bool).sum()),
    }
    diffs = [f"{k}: expected {EXPECTED[k]}, got {got[k]}" for k in EXPECTED if got[k] != EXPECTED[k]]
    train_col = int(tr[tr.modality == "IR"].is_colorized.fillna(False).astype(bool).sum())
    if train_col:
        diffs.append(f"train IR colorized: expected 0, got {train_col}")
    return diffs


def write_summary(df: pd.DataFrame, out: Path, diffs: list[str]) -> None:
    tr = df[df.split == "train"].copy()
    for c in ("n_panels", "n_defects", "n_auto_panels"):
        tr[c] = tr[c].astype(int)
    lines = ["# Dataset audit", "", f"Samples: {len(df)} rows (one per modality per sample).", ""]
    lines += ["## Verified-number check", "", "PASS" if not diffs else "FAIL", ""]
    lines += [f"- {d}" for d in diffs]
    lines += ["", "## Counts", "", "| split | modality | rows | size(s) |", "|---|---|---|---|"]
    for (s, m), g in df.groupby(["split", "modality"]):
        sizes = ", ".join(f"{w}x{h}:{n}" for (w, h), n in g.groupby(["width", "height"]).size().items())
        lines.append(f"| {s} | {m} | {len(g)} | {sizes} |")
    lines += ["", "## Train annotations", "", "| modality | panels | defects | images with defect | auto-label images | auto panels |",
              "|---|---|---|---|---|---|"]
    for m, g in tr.groupby("modality"):
        lines.append(f"| {m} | {g.n_panels.sum()} | {g.n_defects.sum()} | {(g.n_defects > 0).sum()} | "
                     f"{g.has_auto_label.astype(bool).sum()} | {g.n_auto_panels.sum()} |")
    lines += ["", "## labelme version x defect", "", "| modality | version | with defect | no defect |", "|---|---|---|---|"]
    for (m, v), g in tr.groupby(["modality", "labelme_version"]):
        lines.append(f"| {m} | {v} | {(g.n_defects > 0).sum()} | {(g.n_defects == 0).sum()} |")
    lines += ["", "## Brightness (mean of grayscale)", "", "| split | modality | p10 | median | p90 |", "|---|---|---|---|---|"]
    for (s, m), g in df.groupby(["split", "modality"]):
        q = g.mean_brightness.quantile([.1, .5, .9])
        lines.append(f"| {s} | {m} | {q.iloc[0]:.1f} | {q.iloc[1]:.1f} | {q.iloc[2]:.1f} |")
    ir = df[df.modality == "IR"]
    lines += ["", "## IR colorized", "", "| split | colorized | gray |", "|---|---|---|"]
    for s, g in ir.groupby("split"):
        c = int(g.is_colorized.astype(bool).sum())
        lines.append(f"| {s} | {c} | {len(g) - c} |")
    lines += ["", f"Fields {UNKNOWN!r}: panel_id, site_id, weather, timestamp (not present in the data).", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    setup_logging("INFO")
    cfg = load_config(resolve_repo_path(args.config))
    paths = DatasetPaths.from_config(cfg)
    jobs = collect_jobs(paths, float(cfg["ir_color"]["colorized_threshold"]))
    logger.info("auditing %d images", len(jobs))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_process, jobs, chunksize=16))
    cols = ["sample_id", "pair_id", "split", "modality", "image_path", "label_path", "width", "height",
            "labelme_version", "n_panels", "n_defects", "defect_label", "has_auto_label", "n_auto_panels",
            "mean_brightness", "std_brightness", "is_colorized", "phash", "panel_id", "site_id", "weather", "timestamp"]
    df = pd.DataFrame(rows)[cols].sort_values(["split", "pair_id", "modality"]).reset_index(drop=True)
    csv_out = resolve_repo_path(cfg["audit"]["metadata_csv"])
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False, encoding="utf-8")
    diffs = check_expected(df)
    write_summary(df, resolve_repo_path(cfg["audit"]["summary_md"]), diffs)
    logger.info("wrote %s (%d rows)", csv_out, len(df))
    if diffs:
        for d in diffs:
            logger.error("MISMATCH %s", d)
        return 1
    logger.info("all verified numbers match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
