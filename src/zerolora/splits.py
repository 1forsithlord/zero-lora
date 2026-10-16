"""Deterministic, disjoint pool construction + hashing. numpy only (testable without torch)."""
import hashlib

import numpy as np


def stratified_split(labels, fractions, seed):
    """Split indices per class into named, disjoint pools.

    fractions: dict name -> fraction, must sum to 1. Last pool absorbs rounding remainder.
    Returns dict name -> sorted np.ndarray of indices.
    """
    names = list(fractions)
    assert abs(sum(fractions.values()) - 1.0) < 1e-9, "fractions must sum to 1"
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    out = {n: [] for n in names}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        idx = idx[rng.permutation(len(idx))]
        n = len(idx)
        cuts, acc = [], 0
        for name in names[:-1]:
            acc += int(round(fractions[name] * n))
            cuts.append(acc)
        for name, part in zip(names, np.split(idx, cuts)):
            out[name].append(part)
    return {n: np.sort(np.concatenate(v)) for n, v in out.items()}


def assert_disjoint(pools):
    seen = set()
    for name, idx in pools.items():
        s = set(int(i) for i in idx)
        assert not (s & seen), f"pool {name} overlaps an earlier pool"
        seen |= s


def index_hash(idx):
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:16]
