"""Text-score PROXY (read only; the official text metric is unknown).

(a) OOF pairs: v2 operating threshold (EO crack >= 0.65, IR hotspot >= 0.75), prediction matched one-to-one (greedy, IoU >= 0.5) with a GT
    defect of the same class; BERTScore-F1 (bert-base-chinese, CPU) of the v2 placeholder sentence against that GT reason, with and without
    baseline rescaling. Distinct GT sentences are scored once and weighted by their number of matched pairs.
(b) the 10 most frequent reasons per class in the training labels, sentence length distribution
(c) share of the matched pairs whose placeholder BERTScore-F1 is >= tau for several tau (to reason about which tau would give F1_TEXT = 0)

Usage: python scripts/reason_proxy.py --v2 C:\\aicup_scratch\\submission_v2_defect_panel.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.dataset.labelme import DEFECT_LABEL, load_labelme  # noqa: E402
from src.dataset.paths import DatasetPaths, resolve_repo_path  # noqa: E402
from src.dataset.workroot import work_root  # noqa: E402
from src.postprocess.pipeline import raw_to_boxes  # noqa: E402
from src.utils.config import load_config  # noqa: E402

CONF = {"EO": 0.65, "IR": 0.75}
TAUS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
MODEL = "bert-base-chinese"


def wq(values: np.ndarray, weights: np.ndarray, qs=(0.1, 0.5, 0.9)) -> list[float]:
    idx = np.argsort(values)
    v, w = values[idx], weights[idx]
    cw = np.cumsum(w) / w.sum()
    return [float(v[min(np.searchsorted(cw, q), len(v) - 1)]) for q in qs]


def matched_gt_reasons(paths, mod: str, cd: Path) -> list[str]:
    """GT reason of every OOF prediction that is matched (greedy, IoU >= 0.5) to a GT defect of the same class."""
    from shapely.geometry import Polygon

    out: list[str] = []
    imgs = {}
    for k in range(3):
        imgs.update(json.loads((cd / f"{mod}_fold{k}_val_base.json").read_text(encoding="utf-8"))["images"])
    for stem in sorted(imgs):
        doc = load_labelme(paths.label_dir(mod) / f"{stem}.json", mod)
        gts = [(Polygon(s.points).buffer(0), s.reason) for s in doc.defects]
        preds = sorted([b for b in raw_to_boxes(imgs[stem], "approx4", CONF[mod]) if b.label == DEFECT_LABEL[mod] and b.conf >= CONF[mod]], key=lambda b: -b.conf)
        used: set[int] = set()
        for b in preds:
            p = Polygon(b.points).buffer(0)
            best, bj = 0.0, -1
            for j, (g, _) in enumerate(gts):
                if j in used:
                    continue
                inter = p.intersection(g).area
                u = p.area + g.area - inter
                v = inter / u if u > 0 else 0.0
                if v > best:
                    best, bj = v, j
            if bj >= 0 and best >= 0.5:
                used.add(bj)
                out.append(gts[bj][1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2", required=True)
    args = ap.parse_args()
    base = load_config(resolve_repo_path("configs/default.yaml"))
    paths = DatasetPaths.from_config(base)
    cd = work_root(base) / "cache"
    v2 = json.loads(Path(args.v2).read_text(encoding="utf-8"))
    placeholder = {}
    for k, objs in v2.items():
        for o in objs:
            if "reason" in o:
                placeholder[k.split("/")[1]] = o["reason"]
    from bert_score import BERTScorer

    scorers = {"raw": BERTScorer(model_type=MODEL, lang="zh", device="cpu", batch_size=64),
               "rescaled": BERTScorer(model_type=MODEL, lang="zh", device="cpu", batch_size=64, rescale_with_baseline=True)}
    print("## (b) most frequent reasons in the training labels\n")
    train_texts = {}
    for mod in ("EO", "IR"):
        texts = []
        for p in sorted(paths.label_dir(mod).glob("*.json")):
            texts += [s.reason for s in load_labelme(p, mod).defects]
        train_texts[mod] = texts
        cnt = Counter(texts)
        ln = np.array([len(t) for t in texts])
        print(f"### {mod} {DEFECT_LABEL[mod]}: {len(texts)} sentences, {len(cnt)} distinct; length (characters) min {ln.min()} / q25 {np.quantile(ln, .25):.0f} / "
              f"median {np.median(ln):.0f} / q75 {np.quantile(ln, .75):.0f} / max {ln.max()} / mean {ln.mean():.1f}; v2 placeholder: {placeholder[mod]} "
              f"(count in training: {cnt.get(placeholder[mod], 0)})\n")
        print("| rank | count | sentence |\n|---|---|---|")
        for i, (t, c) in enumerate(cnt.most_common(10), 1):
            print(f"| {i} | {c} | {t} |")
        print()
    print("## (a) placeholder vs GT reason on matched OOF pairs (BERTScore-F1, bert-base-chinese)\n")
    print("| class | version | matched pairs | mean | p10 | p50 | p90 |\n|---|---|---|---|---|---|---|")
    scores: dict = {}
    for mod in ("EO", "IR"):
        gt_reasons = matched_gt_reasons(paths, mod, cd)
        cnt = Counter(gt_reasons)
        uniq = sorted(cnt)
        w = np.array([cnt[u] for u in uniq], dtype=float)
        for ver, sc in scorers.items():
            _, _, f1 = sc.score([placeholder[mod]] * len(uniq), uniq)
            v = f1.numpy().astype(float)
            scores[(mod, ver)] = (v, w)
            q = wq(v, w)
            print(f"| {DEFECT_LABEL[mod]} ({mod}) | {ver} | {int(w.sum())} | {(v * w).sum() / w.sum():.3f} | {q[0]:.3f} | {q[1]:.3f} | {q[2]:.3f} |")
    print("\n## (c) share of matched OOF pairs with placeholder BERTScore-F1 >= tau\n")
    print("| class | version | " + " | ".join(f"tau {t}" for t in TAUS) + " |\n|---|---|" + "---|" * len(TAUS))
    for mod in ("EO", "IR"):
        for ver in ("raw", "rescaled"):
            v, w = scores[(mod, ver)]
            print(f"| {DEFECT_LABEL[mod]} ({mod}) | {ver} | " + " | ".join(f"{(w[v >= t].sum() / w.sum()):.3f}" for t in TAUS) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
