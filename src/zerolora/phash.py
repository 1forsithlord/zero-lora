"""Perceptual hash (pHash, 64-bit DCT) and a pigeonhole near-duplicate index. numpy + PIL only.

pHash follows the usual recipe (grayscale -> 32x32 -> 2D DCT -> top-left 8x8 -> bit = coeff > median).
Near-duplicate = Hamming distance <= thr (protocol: 6). The index uses the pigeonhole principle:
split the 64 bits into thr+1 segments; two hashes within Hamming distance thr must agree exactly on
at least one segment, so only segment-collisions need a full distance check (exact, not approximate).
"""
from collections import defaultdict

import numpy as np

_N = 32


def _dct_matrix(n=_N):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    return np.cos(np.pi * (2 * i + 1) * k / (2 * n))


_C = _dct_matrix()


def phash(img):
    """img: PIL.Image. Returns a Python int in [0, 2**64)."""
    from PIL import Image
    g = img.convert("L").resize((_N, _N), Image.LANCZOS)
    a = np.asarray(g, dtype=np.float64)
    d = _C @ a @ _C.T
    low = d[:8, :8]
    bits = (low > np.median(low)).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def hamming(a, b):
    return bin(a ^ b).count("1")


class NearDupIndex:
    def __init__(self, thr=6):
        self.thr = int(thr)
        edges = np.linspace(0, 64, self.thr + 2).astype(int)
        self.ranges = list(zip(edges[:-1].tolist(), edges[1:].tolist()))
        self.tables = [defaultdict(list) for _ in self.ranges]
        self.hashes, self.labels = [], []

    def _seg(self, h, lo, hi):
        return (h >> lo) & ((1 << (hi - lo)) - 1)

    def add(self, h, label=None):
        i = len(self.hashes)
        self.hashes.append(int(h)); self.labels.append(label)
        for t, (lo, hi) in zip(self.tables, self.ranges):
            t[self._seg(int(h), lo, hi)].append(i)

    def add_many(self, hs, label=None):
        for h in hs:
            self.add(int(h), label)

    def query(self, h):
        """Return (distance, label) of the nearest stored hash within thr, else None."""
        h = int(h)
        cand = set()
        for t, (lo, hi) in zip(self.tables, self.ranges):
            cand.update(t.get(self._seg(h, lo, hi), ()))
        best = None
        for i in cand:
            d = hamming(h, self.hashes[i])
            if d <= self.thr and (best is None or d < best[0]):
                best = (d, self.labels[i])
        return best

    def __len__(self):
        return len(self.hashes)