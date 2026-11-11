"""Contiguous time-block splits: one boundary function for every stage."""
import numpy as np

from mrc.models.hazard import block_masks


def test_masks_are_contiguous_and_ordered():
    m_tr, m_va, m_te = block_masks(1000)
    tr, va, te = np.flatnonzero(m_tr), np.flatnonzero(m_va), np.flatnonzero(m_te)
    assert tr[0] == 0 and tr[-1] == 699
    assert va[0] == 700 and va[-1] == 849
    assert te[0] == 850 and te[-1] == 999
    # no leakage: every training request precedes every validation request
    assert tr.max() < va.min() < te.min()


def test_masks_cover_every_request_exactly_once():
    m_tr, m_va, m_te = block_masks(1001)
    assert int(m_tr.sum() + m_va.sum() + m_te.sum()) == 1001
    assert not (m_tr & m_va).any()
    assert not (m_va & m_te).any()
    assert not (m_tr & m_te).any()
    assert m_tr.dtype == np.bool_ and m_va.shape == (1001,) and m_te.shape == (1001,)
