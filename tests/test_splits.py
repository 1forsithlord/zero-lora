import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from zerolora.splits import stratified_split, assert_disjoint, index_hash


def _labels():
    rng = np.random.RandomState(0)
    return rng.randint(0, 10, size=27000)


def test_disjoint_and_complete():
    y = _labels()
    p = stratified_split(y, {"adaptation": 0.6, "dev": 0.1, "test": 0.3}, 1234)
    assert_disjoint(p)
    assert sum(len(v) for v in p.values()) == len(y)


def test_deterministic():
    y = _labels()
    f = {"adaptation": 0.6, "dev": 0.1, "test": 0.3}
    a = stratified_split(y, f, 1234)
    b = stratified_split(y, f, 1234)
    assert all(index_hash(a[k]) == index_hash(b[k]) for k in a)
    c = stratified_split(y, f, 99)
    assert index_hash(a["test"]) != index_hash(c["test"])


def test_stratified():
    y = _labels()
    p = stratified_split(y, {"adaptation": 0.6, "dev": 0.1, "test": 0.3}, 1)
    for c in range(10):
        n = (y == c).sum()
        got = (y[p["test"]] == c).sum()
        assert abs(got - 0.3 * n) <= 1


if __name__ == "__main__":
    test_disjoint_and_complete(); test_deterministic(); test_stratified()
    print("splits tests passed")
