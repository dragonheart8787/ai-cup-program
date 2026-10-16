"""EO/IR pairing diagnostics (weak evidence only, never used as labels).

EO and IR images are paired by the number in the file name (train_0001 <-> train_0001, test_001 <-> test_001).
(a) train: 2x2 table (EO has crack x IR has hotspot) from the labels, conditional probabilities, expected counts under independence.
(b) test: same table from predictions (EO crack >= 0.65, IR hotspot >= 0.75, fold-0 models, cached).
Weak evidence: the association in the training labels supports that equal numbers are the same scene, but it is not proof, and test
predictions are not labels.

Usage: python scripts/pair_diagnostics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2_contingency, fisher_exact  # noqa: E402

from src.dataset.paths import resolve_repo_path  # noqa: E402
from src.dataset.workroot import work_root  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def table(a: np.ndarray, b: np.ndarray, an: str, bn: str) -> None:
    tt, tf, ft, ff = int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())
    n = len(a)
    print(f"| | {bn} yes | {bn} no | total |\n|---|---|---|---|")
    print(f"| {an} yes | {tt} | {tf} | {tt + tf} |\n| {an} no | {ft} | {ff} | {ft + ff} |\n| total | {tt + ft} | {tf + ff} | {n} |")
    exp = (tt + tf) * (tt + ft) / n
    p_b_given_a = tt / (tt + tf) if tt + tf else float("nan")
    p_a_given_b = tt / (tt + ft) if tt + ft else float("nan")
    print(f"\n- P({bn} | {an}) = {p_b_given_a:.3f}; P({bn} | no {an}) = {ft / (ft + ff):.3f}; P({an} | {bn}) = {p_a_given_b:.3f}; "
          f"P({an} | no {bn}) = {tf / (tf + ff):.3f}")
    print(f"- both yes: observed {tt}, expected under independence {exp:.1f}")
    if min(tt + tf, tt + ft) > 0:
        odds, p = fisher_exact([[tt, tf], [ft, ff]])
        chi = chi2_contingency([[tt, tf], [ft, ff]])[0]
        print(f"- odds ratio {odds:.2f}, Fisher p = {p:.2g}, chi2 = {chi:.1f}")


def main() -> int:
    base = load_config(resolve_repo_path("configs/default.yaml"))
    meta = pd.read_csv(resolve_repo_path(base["audit"]["metadata_csv"]), dtype={"pair_id": str})
    tr = meta[meta.split == "train"]
    eo = tr[tr.modality == "EO"].set_index("pair_id").n_defects.astype(int) > 0
    ir = tr[tr.modality == "IR"].set_index("pair_id").n_defects.astype(int) > 0
    ir = ir.reindex(eo.index)
    print("## (a) training labels (1046 pairs)\n")
    table(eo.to_numpy(), ir.to_numpy(), "EO crack", "IR hotspot")
    cd = work_root(base) / "cache"
    e = json.loads((cd / "EO_fold0_test_run1.json").read_text(encoding="utf-8"))["images"]
    i = json.loads((cd / "IR_fold0_test_run1.json").read_text(encoding="utf-8"))["images"]
    stems = sorted(Path(k).stem for k in e)
    ea = np.array([max((p["conf"] for p in e[f"test/EO/image/{s}.jpg"]["preds"] if p["label"] == "crack"), default=0) >= 0.65 for s in stems])
    ib = np.array([max((p["conf"] for p in i[f"test/IR/image/{s}.jpg"]["preds"] if p["label"] == "hotspot"), default=0) >= 0.75 for s in stems])
    print("\n## (b) test predictions (696 pairs; EO crack >= 0.65, IR hotspot >= 0.75; fold-0 models; NOT labels)\n")
    table(ea, ib, "EO crack (pred)", "IR hotspot (pred)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
