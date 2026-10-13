import numpy as np
import pytest

from src.dataset.groups import LinkRule, assign_folds, check_no_leak, group_ids, link_matrix

RULE = LinkRule(6, 0.9, 6.0)
D = 64 * 36


def test_true_duplicates_link_but_dark_noise_with_same_hash_does_not():
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 255, size=(1, D))
    dup = base + rng.normal(0, 1, base.shape)                      # real near-duplicate
    dark1 = rng.uniform(0, 12, (1, D))
    dark2 = rng.uniform(0, 12, (1, D))                             # unrelated dark frames
    thumbs = np.vstack([base, dup, dark1, dark2]).astype(np.float32)
    hashes = ["f" * 16] * 4                                         # identical (collapsed) hashes
    mean = np.array([120.0, 120.0, 7.0, 7.0])
    adj = link_matrix(hashes, thumbs, mean, RULE, low_info_mean=15.0, strict_mad=3.0)
    assert adj[0, 1] and adj[1, 0]
    assert not adj[2, 3]
    g = group_ids(adj)
    assert g[0] == g[1] and len({g[2], g[3], g[0]}) == 3


def test_different_hash_not_linked():
    rng = np.random.default_rng(1)
    t = np.repeat(rng.uniform(0, 255, size=(1, D)), 2, axis=0).astype(np.float32)
    adj = link_matrix(["0" * 16, "f" * 16], t, np.array([100.0, 100.0]), RULE, 15.0, 3.0)
    assert not adj.any()


def test_folds_no_leak_and_deterministic():
    rng = np.random.default_rng(0)
    n = 300
    group = rng.integers(0, 120, n)
    strata = rng.integers(0, 4, n)
    folds = assign_folds(group, strata, 3, seed=42)
    check_no_leak(group, folds)
    assert set(folds) == {0, 1, 2}
    assert np.array_equal(folds, assign_folds(group, strata, 3, seed=42))


def test_leak_detected():
    with pytest.raises(RuntimeError):
        check_no_leak(np.array([0, 0, 1]), np.array([0, 1, 1]))
