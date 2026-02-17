import numpy as np

import tools.run_cross_section_rank_sleeve as rank_sleeve
import tools.run_regime_moe as regime_moe
import tools.run_uncertainty_sizing as uncertainty_sizing


def test_build_rank_sleeve_is_neutral_and_l1_normalized():
    rng = np.random.default_rng(7)
    asset_returns = rng.normal(0.0, 0.01, size=(120, 8))
    w = rank_sleeve.build_rank_sleeve(
        asset_returns,
        mom_short=10,
        mom_long=30,
        rev_short=4,
        gross_target=1.0,
        per_asset_cap=0.20,
    )
    assert w.shape == asset_returns.shape
    row_net = np.sum(w, axis=1)
    assert float(np.max(np.abs(row_net))) < 1e-8
    row_l1 = np.sum(np.abs(w), axis=1)
    assert float(np.min(row_l1)) > 0.99
    assert float(np.max(row_l1)) < 1.01


def test_build_regime_moe_outputs_bounded_governor():
    t = 240
    x = np.linspace(0.0, 20.0, t)
    daily_returns = 0.002 * np.sin(x) + 0.0007 * np.cos(0.3 * x)
    base_signal = np.sin(0.4 * x)
    out = regime_moe.build_regime_moe(
        daily_returns,
        base_signal,
        governor_alpha=0.30,
        governor_min=0.65,
        governor_max=1.25,
    )
    g = out["governor"]
    assert len(g) == t
    assert float(np.min(g)) >= 0.65 - 1e-9
    assert float(np.max(g)) <= 1.25 + 1e-9
    assert np.isfinite(g).all()


def test_uncertainty_scalar_reacts_to_shock_and_uncertainty():
    t = 20
    conf = np.full(t, 0.9)
    conf[-5:] = 0.2
    dgate = np.full(t, 0.95)
    mprob = np.full(t, 0.9)
    mprob[-5:] = 0.5
    shock = np.zeros(t)
    shock[-5:] = 1.0
    out = uncertainty_sizing.build_uncertainty_scalar(
        t,
        conf_cal=conf,
        conf_raw=conf,
        disagreement_gate=dgate,
        meta_exec_prob=mprob,
        shock_mask=shock,
        beta=0.45,
        shock_penalty=0.25,
        floor=0.50,
        ceiling=1.10,
    )
    s = out["scalar"]
    assert len(s) == t
    assert float(np.min(s)) >= 0.50 - 1e-9
    assert float(np.max(s)) <= 1.10 + 1e-9
    assert float(np.mean(s[-5:])) < float(np.mean(s[:10]))
