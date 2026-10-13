"""Near-duplicate grouping and leakage-free fold assignment.

Why phash alone is not enough: very dark / low-information images collapse to
the same perceptual hash (e.g. 21 EO images with mean brightness 7 share one
hash), so a plain Hamming <= 6 rule chains unrelated images together. Two extra
conditions are required before two images are linked:

* thumbnail agreement: Pearson correlation and mean absolute difference of a
  64x36 grayscale thumbnail, which depend on actual pixel content, and
* a stricter rule for low-information images (mean brightness < ``low_info_mean``):
  both images must agree on the strictest preset's thumbnail MAD.

Genuine repeats are kept: the largest identical-hash group of the real data
(30 EO images, mean 59) has a median thumbnail MAD of ~1 gray level.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class LinkRule:
    phash_dist: int
    thumb_corr_min: float
    thumb_mad_max: float


def hex_to_bits(hex_hashes: list[str]) -> np.ndarray:
    """(n, 8) uint8 matrix of 64-bit hashes."""
    arr = np.empty((len(hex_hashes), 8), dtype=np.uint8)
    for i, h in enumerate(hex_hashes):
        if len(h) != 16:
            raise ValueError(f"expected a 64-bit hex phash, got {h!r}")
        arr[i] = np.frombuffer(bytes.fromhex(h), dtype=np.uint8)
    return arr


def hamming_matrix(bits: np.ndarray) -> np.ndarray:
    x = bits[:, None, :] ^ bits[None, :, :]
    return _POPCOUNT[x].sum(-1).astype(np.int16)


def link_matrix(
    hex_hashes: list[str], thumbs: np.ndarray, mean_brightness: np.ndarray, rule: LinkRule,
    low_info_mean: float, strict_mad: float,
) -> np.ndarray:
    """Boolean (n, n) adjacency of 'same scene' for one modality."""
    n = len(hex_hashes)
    if thumbs.shape[0] != n or mean_brightness.shape[0] != n:
        raise ValueError("hashes, thumbs and brightness must have the same length")
    dist = hamming_matrix(hex_to_bits(hex_hashes))
    cand = dist <= rule.phash_dist
    np.fill_diagonal(cand, False)
    adj = np.zeros((n, n), dtype=bool)
    ii, jj = np.nonzero(np.triu(cand))
    if len(ii) == 0:
        return adj
    t = thumbs.astype(np.float64)
    tc = t - t.mean(1, keepdims=True)
    norm = np.linalg.norm(tc, axis=1)
    mad = np.abs(t[ii] - t[jj]).mean(1)
    denom = norm[ii] * norm[jj]
    corr = np.where(denom > 1e-9, (tc[ii] * tc[jj]).sum(1) / np.maximum(denom, 1e-9), 0.0)
    ok = (corr >= rule.thumb_corr_min) & (mad <= rule.thumb_mad_max)
    low = (mean_brightness[ii] < low_info_mean) | (mean_brightness[jj] < low_info_mean)
    ok &= ~low | (mad <= min(strict_mad, rule.thumb_mad_max))
    adj[ii[ok], jj[ok]] = True
    return adj | adj.T


def group_ids(adj: np.ndarray) -> np.ndarray:
    _, labels = connected_components(csr_matrix(adj), directed=False)
    return labels


def assign_folds(group: np.ndarray, strata: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Fold per item with StratifiedGroupKFold; groups never span folds."""
    folds = np.full(len(group), -1, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(group)), strata, group)):
        folds[test_idx] = k
    if (folds < 0).any():
        raise RuntimeError("some items were not assigned to a fold")
    check_no_leak(group, folds)
    return folds


def check_no_leak(group: np.ndarray, folds: np.ndarray) -> None:
    for g in np.unique(group):
        fs = np.unique(folds[group == g])
        if len(fs) != 1:
            raise RuntimeError(f"group {g} spans folds {fs.tolist()}")
