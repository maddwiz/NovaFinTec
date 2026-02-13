import numpy as np

from qmods.immune_drill import run_scenarios


def test_run_scenarios_shape_and_keys():
    rng = np.random.default_rng(9)
    W = rng.normal(0.0, 0.05, size=(300, 8))
    A = rng.normal(0.0002, 0.01, size=(300, 8))
    out = run_scenarios(W, A)
    assert "summary" in out and "scenarios" in out
    sc = out["scenarios"]
    for k in ["baseline", "vol_spike_1p5x", "trend_reversal_tail", "flash_crash_monthly", "corr_shock"]:
        assert k in sc
        assert "sharpe" in sc[k]
        assert "max_dd" in sc[k]
        assert "tail_p01" in sc[k]
    sm = out["summary"]
    assert "pass" in sm
    assert "worst_max_dd" in sm
    assert np.isfinite(float(sm["worst_max_dd"]))
