import tools.run_governor_param_sweep as rps


def test_objective_penalizes_drawdown_and_can_veto():
    base = {
        "sharpe": 0.75,
        "hit_rate": 0.52,
        "max_drawdown": -0.05,
        "turnover_mean": 0.01,
    }
    bad = {
        "sharpe": 0.90,
        "hit_rate": 0.48,
        "max_drawdown": -0.12,
        "turnover_mean": 0.03,
    }
    score, detail = rps._objective(bad, base)
    assert detail["dd_ratio"] > 2.0
    assert detail["veto"] is True
    assert score < 0.5


def test_profile_from_row_casts_types():
    row = {
        "runtime_total_floor": 0.1,
        "shock_alpha": 0.35,
        "rank_sleeve_blend": 0.06,
        "low_vol_sleeve_blend": 0.04,
        "meta_execution_gate_strength": 0.95,
        "council_gate_strength": 0.9,
        "meta_mix_leverage_strength": 1.15,
        "meta_reliability_strength": 1.1,
        "global_governor_strength": 0.95,
        "heartbeat_scaler_strength": 0.4,
        "quality_governor_strength": 1.05,
        "regime_moe_strength": 1.2,
        "uncertainty_sizing_strength": 0.85,
        "vol_target_strength": 0.7,
        "use_concentration_governor": 1,
        "concentration_top1_cap": 0.18,
        "concentration_top3_cap": 0.42,
        "concentration_max_hhi": 0.14,
    }
    out = rps._profile_from_row(row)
    assert out["runtime_total_floor"] == 0.1
    assert out["shock_alpha"] == 0.35
    assert out["rank_sleeve_blend"] == 0.06
    assert out["low_vol_sleeve_blend"] == 0.04
    assert out["meta_execution_gate_strength"] == 0.95
    assert out["council_gate_strength"] == 0.9
    assert out["meta_mix_leverage_strength"] == 1.15
    assert out["meta_reliability_strength"] == 1.1
    assert out["global_governor_strength"] == 0.95
    assert out["heartbeat_scaler_strength"] == 0.4
    assert out["quality_governor_strength"] == 1.05
    assert out["regime_moe_strength"] == 1.2
    assert out["uncertainty_sizing_strength"] == 0.85
    assert out["vol_target_strength"] == 0.7
    assert out["use_concentration_governor"] is True
    assert out["concentration_top1_cap"] == 0.18
