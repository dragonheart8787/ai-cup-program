"""Choose ONE placeholder reason per defect class by BERTScore (PROXY: the official model / language setting is not published).

For each class (EO crack, IR hotspot) all training reasons are collected; the single sentence with the highest mean
BERTScore-F1 against ALL sentences of that class (duplicates counted, i.e. the medoid of the empirical distribution) is
selected. Also reports pairwise-score statistics as a rough ceiling for any single fixed sentence (not an official score).
CPU only, deterministic. Needs `pip install bert-score` and Hugging Face access to google-bert/bert-base-chinese.

Usage: AICUP_DATA_ROOT=.. AICUP_WORK_ROOT=.. python scripts/select_reason_bertscore.py
Writes <work>/reasons_bertscore.json and logs a row to experiments/experiments.csv.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.dataset.labelme import DEFECT_LABEL, load_labelme  # noqa: E402
from src.dataset.paths import DatasetPaths, repo_root, resolve_repo_path  # noqa: E402
from src.dataset.workroot import work_root  # noqa: E402
from src.submission.reasons import most_central_sentence  # noqa: E402
from src.utils.config import config_hash, load_config  # noqa: E402
from src.utils.experiments import append_experiment, file_sha256, get_git_commit  # noqa: E402
from src.utils.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger("bertscore")
MODEL = "bert-base-chinese"
HF_ID = "google-bert/bert-base-chinese"


def collect(paths: DatasetPaths, modality: str) -> list[str]:
    texts: list[str] = []
    for p in sorted(paths.label_dir(modality).glob("*.json")):
        texts += [s.reason for s in load_labelme(p, modality).defects]
    return texts


def f1_matrix(uniq: list[str]) -> np.ndarray:
    """F[i, j] = BERTScore-F1 of candidate i against reference j over the distinct sentences."""
    from bert_score import BERTScorer

    scorer = BERTScorer(model_type=MODEL, lang="zh", device="cpu", batch_size=64)
    n = len(uniq)
    cands = [uniq[i] for i in range(n) for j in range(n)]
    refs = [uniq[j] for i in range(n) for j in range(n)]
    _, _, f1 = scorer.score(cands, refs)
    return f1.numpy().reshape(n, n).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out")
    args = ap.parse_args()
    setup_logging("INFO")
    base = load_config(resolve_repo_path("configs/default.yaml"))
    paths = DatasetPaths.from_config(base)
    import bert_score
    from huggingface_hub import model_info

    info = model_info(HF_ID)  # raises if Hugging Face is unreachable (no fallback to another source)
    meta = {"model": HF_ID, "hf_sha": info.sha, "license_tags": [t for t in info.tags if t.startswith("license")],
            "bert_score_version": bert_score.__version__}
    logger.info("model %s", meta)
    result: dict = {"meta": meta, "classes": {}}
    for mod in ("EO", "IR"):
        texts = collect(paths, mod)
        cnt = Counter(texts)
        uniq = sorted(cnt)
        w = np.array([cnt[u] for u in uniq], dtype=np.float64)
        F = f1_matrix(uniq)
        mean_vs_all = (F * w[None, :]).sum(1) / w.sum()  # candidate i vs all sentences (duplicates counted)
        order = np.argsort(-mean_vs_all, kind="stable")
        current = most_central_sentence(texts)  # existing char n-gram placeholder
        ci = uniq.index(current)
        # pairwise statistics over distinct instance pairs i != j (weights = counts)
        pair_w = np.outer(w, w) - np.diag(w)  # remove each instance paired with itself
        flat, ww = F.ravel(), pair_w.ravel()
        idx = np.argsort(flat)
        cw = np.cumsum(ww[idx]) / ww.sum()
        qs = {q: float(flat[idx][np.searchsorted(cw, q)]) for q in (0.25, 0.5, 0.75)}
        result["classes"][mod] = {
            "label": DEFECT_LABEL[mod], "n_sentences": len(texts), "n_distinct": len(uniq),
            "selected": uniq[int(order[0])], "selected_mean_f1": float(mean_vs_all[order[0]]),
            "current_placeholder": current, "current_mean_f1": float(mean_vs_all[ci]),
            "top5": [{"text": uniq[int(i)], "mean_f1": float(mean_vs_all[i]), "count": int(w[i])} for i in order[:5]],
            "pairwise_mean_f1": float((flat * ww).sum() / ww.sum()), "pairwise_quartiles": qs,
            "mean_f1_distinct_pairs": float(F[~np.eye(len(uniq), dtype=bool)].mean()),
        }
        logger.info("%s selected: %s (%.4f) | current placeholder %.4f", mod, uniq[int(order[0])], mean_vs_all[order[0]], mean_vs_all[ci])
    out = Path(args.out) if args.out else work_root(base) / "reasons_bertscore.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    cls = result["classes"]
    append_experiment(resolve_repo_path(base["paths"]["experiments_csv"]), {
        "git_commit": get_git_commit(repo_root()), "config_hash": config_hash({"model": HF_ID, "script": "select_reason_bertscore"}),
        "dataset_version": f"folds.csv:{file_sha256(resolve_repo_path(base['folds']['folds_csv']))[:12]}",
        "model": f"{HF_ID}@{meta['hf_sha'][:8]}", "input_modality": "EO+IR", "text_score": round(
            (cls["EO"]["selected_mean_f1"] + cls["IR"]["selected_mean_f1"]) / 2, 4),
        "notes": f"BERTScore PROXY (not official) medoid reason selection; EO mean F1 {cls['EO']['selected_mean_f1']:.4f} (placeholder {cls['EO']['current_mean_f1']:.4f}), "
                 f"IR {cls['IR']['selected_mean_f1']:.4f} (placeholder {cls['IR']['current_mean_f1']:.4f}); out={out.name}",
    })
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
