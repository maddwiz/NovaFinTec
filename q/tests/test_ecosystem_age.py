import numpy as np
import pandas as pd

from qmods.ecosystem_age import govern_hive_weights


def test_govern_hive_weights_actions_and_diag_passthrough_ignored():
    T = 140
    dates = pd.date_range("2025-01-01", periods=T, freq="D")

    w = pd.DataFrame(
        {
            "DATE": dates,
            "EQ": np.full(T, 0.60),
            "FX": np.full(T, 0.35),
            "RATES": np.full(T, 0.05),
            "arb_alpha": np.linspace(1.0, 3.0, T),
            "arb_inertia": np.linspace(0.5, 0.9, T),
        }
    )

    eq_sig = np.sin(np.linspace(0, 20, T)) * 0.40
    rates_sig = eq_sig * 0.95 + 0.01
    fx_sig = np.full(T, -0.10)

    hs_rows = []
    for i, d in enumerate(dates):
        hs_rows.append({"DATE": d, "HIVE": "EQ", "hive_signal": eq_sig[i], "hive_health": 0.8})
        hs_rows.append({"DATE": d, "HIVE": "RATES", "hive_signal": rates_sig[i], "hive_health": 0.5})
        hs_rows.append({"DATE": d, "HIVE": "FX", "hive_signal": fx_sig[i], "hive_health": -0.9})
    hs = pd.DataFrame(hs_rows)

    out, summary = govern_hive_weights(w, hs, split_trigger=0.45)

    assert list(out.columns) == ["DATE", "EQ", "FX", "RATES"]
    vals = out[["EQ", "FX", "RATES"]].values
    assert np.allclose(vals.sum(axis=1), 1.0, atol=1e-6)
    counts = summary.get("event_counts", {})
    assert counts.get("atrophy_applied", 0) > 0
    assert counts.get("split_applied", 0) > 0
    assert counts.get("fusion_applied", 0) > 0
    assert "action_pressure_mean" in summary
    assert "action_pressure_max" in summary
    aps = np.asarray(summary.get("action_pressure_series", []), float)
    assert len(aps) == T
    assert float(np.min(aps)) >= 0.0 - 1e-9


def test_govern_hive_weights_recovery_shield_blocks_over_atrophy():
    T = 120
    dates = pd.date_range("2025-03-01", periods=T, freq="D")
    w = pd.DataFrame(
        {
            "DATE": dates,
            "EQ": np.full(T, 0.55),
            "FX": np.full(T, 0.35),
            "RATES": np.full(T, 0.10),
        }
    )

    # FX starts weak but steadily recovers.
    fx_health = np.linspace(-0.85, 0.45, T)
    hs_rows = []
    for i, d in enumerate(dates):
        hs_rows.append({"DATE": d, "HIVE": "EQ", "hive_signal": 0.20, "hive_health": 0.60})
        hs_rows.append({"DATE": d, "HIVE": "RATES", "hive_signal": 0.08, "hive_health": 0.35})
        hs_rows.append({"DATE": d, "HIVE": "FX", "hive_signal": 0.05, "hive_health": fx_health[i]})
    hs = pd.DataFrame(hs_rows)

    _, summary = govern_hive_weights(
        w,
        hs,
        atrophy_trigger=0.40,
        atrophy_cap=0.08,
        recovery_slope_trigger=0.002,
        split_trigger=0.70,
    )
    counts = summary.get("event_counts", {})
    assert counts.get("recovery_shielded", 0) > 0
