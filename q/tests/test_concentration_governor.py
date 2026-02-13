import numpy as np

from qmods.concentration_governor import govern_matrix, govern_row


def test_govern_row_reduces_concentration():
    w = np.array([0.9, 0.05, 0.03, 0.02], dtype=float)
    out, info = govern_row(w, top1_cap=0.30, top3_cap=0.75, max_hhi=0.30)
    g0 = np.sum(np.abs(w))
    g1 = np.sum(np.abs(out))
    assert np.isfinite(out).all()
    assert abs(g1 - g0) < 1e-6
    assert info["hhi_after"] <= info["hhi_before"] + 1e-9
    assert info["top1_after"] <= 0.30 + 1e-3


def test_govern_matrix_shape_and_finiteness():
    rng = np.random.default_rng(12)
    W = rng.normal(0.0, 0.2, size=(120, 12))
    out, stats = govern_matrix(W, top1_cap=0.22, top3_cap=0.50, max_hhi=0.18)
    assert out.shape == W.shape
    assert np.isfinite(out).all()
    assert stats["hhi_after"] <= stats["hhi_before"] + 1e-9
