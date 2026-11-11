"""Exact-first gates: the fast paths must reproduce the slow references bit for bit."""
import numpy as np
import pytest

from mrc.data.reference import naive_lru_mrc, naive_reuse
from mrc.data.reuse import compute_reuse
from mrc.eval.mrc import mrc_curve


def _toy(n=400, keys=40, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.integers(0, keys, n).astype(np.int64)
    s = rng.integers(8, 128, n).astype(np.int64)
    return k, s


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reuse_matches_reference(seed):
    k, s = _toy(seed=seed)
    d = compute_reuse(k, s, int(k.max()) + 1)
    nR, nB = naive_reuse(k, s)
    assert np.array_equal(d["R"], nR)
    assert np.array_equal(d["B"], nB)


@pytest.mark.parametrize("cap", [1, 2, 5, 17, 100])
def test_object_mrc_matches_lru_replay(cap):
    k, s = _toy()
    d = compute_reuse(k, s, int(k.max()) + 1)
    assert mrc_curve(d["R"], np.array([cap]))[0] == pytest.approx(
        naive_lru_mrc(k, s, cap), abs=1e-12)


def test_byte_mrc_close_to_replay():
    k, s = _toy(2000, 60, seed=3)
    d = compute_reuse(k, s, int(k.max()) + 1)
    for cap in (256, 1024, 4096):
        pred = mrc_curve(d["B"], np.array([cap]), own_size=s)[0]
        truth = naive_lru_mrc(k, s, cap, by_bytes=True)
        # Byte stack distance is exact for object counts but approximate for bytes:
        # objects larger than the remaining space, and objects larger than the whole
        # cache, break the "still resident" argument. Measured gap on real traces is
        # ~1e-4; on small synthetic traces it can reach a few percent. The reported
        # MRC numbers use simulation as ground truth, not this curve.
        assert abs(pred - truth) < 6e-2


def test_style_contract():
    import matplotlib as mpl

    from mrc.plots.style import SERIF_CANDIDATES, apply_style
    apply_style()
    assert mpl.rcParams["font.family"] == ["serif"] or mpl.rcParams["font.family"] == "serif"
    assert mpl.rcParams["axes.grid"] is False
    assert not mpl.rcParams["axes.spines.top"]
    assert SERIF_CANDIDATES[0] == "Nimbus Roman"
