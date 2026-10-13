"""3-fold leak-free split -> data/folds.csv (sample_id, group_id, fold).

Needs data/metadata.csv (run scripts/audit_dataset.py first). EO and IR with the
same number are one pair and always share group and fold. Groups are connected
components of near-duplicate links in EO *or* IR (see src/dataset/groups.py).

Usage: AICUP_DATA_ROOT=<dir> python scripts/make_folds.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.dataset.groups import (LinkRule, assign_folds, group_ids, hamming_matrix, hex_to_bits,  # noqa: E402
                                link_matrix)
from src.dataset.images import thumbnail  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402
from src.dataset.paths import resolve_repo_path  # noqa: E402

logger = logging.getLogger("folds")


def build_pair_table(meta: pd.DataFrame) -> pd.DataFrame:
    tr = meta[meta.split == "train"]
    eo = tr[tr.modality == "EO"].set_index("pair_id").sort_index()
    ir = tr[tr.modality == "IR"].set_index("pair_id").sort_index()
    if not eo.index.equals(ir.index):
        raise RuntimeError("EO and IR train pair ids differ")
    return pd.DataFrame({
        "eo_image": eo.image_path, "ir_image": ir.image_path,
        "eo_phash": eo.phash, "ir_phash": ir.phash,
        "eo_mean": eo.mean_brightness, "ir_mean": ir.mean_brightness,
        "eo_defect": (eo.n_defects.astype(int) > 0), "ir_defect": (ir.n_defects.astype(int) > 0),
        "eo_panels": eo.n_panels.astype(int), "ir_panels": ir.n_panels.astype(int),
        "eo_n_def": eo.n_defects.astype(int), "ir_n_def": ir.n_defects.astype(int),
        "eo_version": eo.labelme_version, "ir_version": ir.labelme_version,
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    setup_logging("INFO")
    cfg = load_config(resolve_repo_path(args.config))
    fc = cfg["folds"]
    seed = int(cfg["seed"])
    meta = pd.read_csv(resolve_repo_path(cfg["audit"]["metadata_csv"]), dtype={"pair_id": str})
    pairs = build_pair_table(meta)
    n = len(pairs)
    thumb_size = tuple(fc["thumb_size"])
    logger.info("computing thumbnails for %d pairs", n)
    t_eo = np.stack([thumbnail(Path(p), thumb_size) for p in pairs.eo_image])
    t_ir = np.stack([thumbnail(Path(p), thumb_size) for p in pairs.ir_image])
    presets = [LinkRule(int(p["phash_dist"]), float(p["thumb_corr_min"]), float(p["thumb_mad_max"])) for p in fc["presets"]]
    strict_mad = min(r.thumb_mad_max for r in presets)
    limit = float(fc["max_group_frac"]) * n

    chosen, groups = None, None
    for rule in presets:
        adj = (link_matrix(pairs.eo_phash.tolist(), t_eo, pairs.eo_mean.to_numpy(), rule, float(fc["low_info_mean"]), strict_mad)
               | link_matrix(pairs.ir_phash.tolist(), t_ir, pairs.ir_mean.to_numpy(), rule, float(fc["low_info_mean"]), strict_mad))
        g = group_ids(adj)
        sizes = np.bincount(g)
        logger.info("rule %s -> %d groups, largest %d (%.1f%% of pairs, limit %.0f)", rule, len(sizes), sizes.max(), 100 * sizes.max() / n, limit)
        if sizes.max() <= limit:
            chosen, groups = rule, g
            break
    if chosen is None:
        raise RuntimeError("no preset keeps the largest group under max_group_frac; tighten the presets")

    strata = pairs.eo_defect.astype(int).to_numpy() * 2 + pairs.ir_defect.astype(int).to_numpy()
    folds = assign_folds(groups, strata, int(fc["n_splits"]), seed)

    # outputs: one row per sample (EO and IR row of a pair share group/fold)
    pair_idx = {pid: i for i, pid in enumerate(pairs.index)}
    tr = meta[meta.split == "train"]
    out = pd.DataFrame({
        "sample_id": tr.sample_id.to_numpy(),
        "group_id": [f"g{groups[pair_idx[p]]:04d}" for p in tr.pair_id],
        "fold": [int(folds[pair_idx[p]]) for p in tr.pair_id],
    }).sort_values("sample_id").reset_index(drop=True)
    csv_out = resolve_repo_path(fc["folds_csv"])
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_out, index=False)

    # ---- report ---------------------------------------------------------
    lines = [f"# Fold report (rule: {chosen}, seed {seed})", "",
             f"pairs={n}, groups={len(np.unique(groups))}, largest group={np.bincount(groups).max()} "
             f"({100 * np.bincount(groups).max() / n:.1f}%)", "",
             "| fold | pairs | samples | EO panels | IR panels | crack | hotspot | EO ver (5.8/6.3/7.1) | IR ver (6.3/7.1) | largest group |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    ver = lambda s, vs: "/".join(str(int(sum(1 for x in s if x.startswith(v))) ) for v in vs)  # noqa: E731
    for k in range(int(fc["n_splits"])):
        m = folds == k
        lines.append(f"| {k} | {int(m.sum())} | {2 * int(m.sum())} | {int(pairs.eo_panels[m].sum())} | {int(pairs.ir_panels[m].sum())} | "
                     f"{int(pairs.eo_n_def[m].sum())} | {int(pairs.ir_n_def[m].sum())} | {ver(pairs.eo_version[m], ('5.8', '6.3', '7.1'))} | "
                     f"{ver(pairs.ir_version[m], ('6.3', '7.1'))} | {int(np.bincount(groups[m]).max())} |")
    lines += ["", "## Strata (EO defect, IR defect) per fold", "", "| fold | none | IR only | EO only | both |", "|---|---|---|---|---|"]
    for k in range(int(fc["n_splits"])):
        c = np.bincount(strata[folds == k], minlength=4)
        lines.append(f"| {k} | {c[0]} | {c[1]} | {c[2]} | {c[3]} |")

    # leakage check: minimum phash distance within vs between folds (EO and IR)
    lines += ["", "## Minimum phash distance (excluding same-group pairs)", "",
              "| modality | within fold, different group | between folds | pairs between folds with phash<=6 |", "|---|---|---|---|"]
    for name, col in (("EO", pairs.eo_phash), ("IR", pairs.ir_phash)):
        D = hamming_matrix(hex_to_bits(col.tolist())).astype(int)
        diff_group = groups[:, None] != groups[None, :]
        same_fold = folds[:, None] == folds[None, :]
        within = D[diff_group & same_fold].min()
        between = D[~same_fold].min()
        n6 = int(((D <= 6) & ~same_fold).sum() // 2)
        lines.append(f"| {name} | {within} | {between} | {n6} |")
    lines += ["", "Pairs between folds with phash<=6 exist by design: they failed the thumbnail/low-information conditions "
              "(different content, not a near-duplicate). Same-group pairs never cross folds (asserted)."]
    rep = "\n".join(lines)
    (csv_out.parent / "folds_summary.md").write_text(rep, encoding="utf-8")
    print(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
