import numpy as np

import tools.make_returns_and_weights as mrw
import tools.rebuild_asset_matrix as ram


def test_make_returns_sanitize_clips_outliers():
    r = np.array([0.01, -2.0, 1e9, np.nan, np.inf, -np.inf], float)
    out, n_clip = mrw.sanitize_returns(r, clip_abs=0.35)
    assert n_clip >= 2
    assert np.isfinite(out).all()
    assert float(np.min(out)) >= -0.95
    assert float(np.max(out)) <= 0.35


def test_rebuild_asset_matrix_sanitize_clips_outliers():
    r = np.array([-10.0, -0.5, 0.02, 9.0], float)
    out, n_clip = ram._sanitize_returns(r)
    assert n_clip == 2
    assert float(np.min(out)) >= -0.95
    assert float(np.max(out)) <= ram.RET_CLIP_ABS
