from pathlib import Path

import pytest

import tools.export_aion_signal_pack as ex


def _write_series(path: Path, vals):
    path.write_text("\n".join(str(v) for v in vals), encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolate_aion_feedback_env(monkeypatch, tmp_path: Path):
    # Keep runtime-context tests deterministic regardless of local AION logs.
    monkeypatch.setenv("Q_AION_SHADOW_TRADES", str(tmp_path / "_missing_shadow_trades.csv"))
    monkeypatch.delenv("Q_AION_HOME", raising=False)
    monkeypatch.delenv("Q_AION_FEEDBACK_LOOKBACK", raising=False)
    monkeypatch.delenv("Q_AION_FEEDBACK_MIN_TRADES", raising=False)


def test_runtime_context_uses_governor_components(tmp_path: Path):
    _write_series(tmp_path / "global_governor.csv", [0.9, 0.8])
    _write_series(tmp_path / "quality_governor.csv", [1.0, 0.95])
    _write_series(tmp_path / "quality_runtime_modifier.csv", [1.0, 0.9])
    _write_series(tmp_path / "meta_mix_reliability_governor.csv", [0.98, 1.02])
    _write_series(tmp_path / "hive_persistence_governor.csv", [1.00, 0.96])
    _write_series(tmp_path / "dna_stress_governor.csv", [0.95, 0.90])
    _write_series(tmp_path / "reflex_health_governor.csv", [0.96, 1.00])
    _write_series(tmp_path / "symbolic_governor.csv", [0.94, 0.98])
    _write_series(tmp_path / "novaspine_context_boost.csv", [1.0, 1.05])
    _write_series(tmp_path / "meta_mix_quality.csv", [0.60, 0.75])
    _write_series(tmp_path / "meta_mix_disagreement.csv", [0.30, 0.20])
    _write_series(tmp_path / "meta_mix_alpha.csv", [0.52, 0.58])
    _write_series(tmp_path / "meta_mix_gross.csv", [0.20, 0.26])
    _write_series(tmp_path / "regime_fracture_governor.csv", [1.00, 0.86])
    (tmp_path / "regime_fracture_info.json").write_text(
        '{"state":"fracture_warn","latest_score":0.76}',
        encoding="utf-8",
    )

    ctx = ex._runtime_context(tmp_path)
    assert 0.50 <= ctx["runtime_multiplier"] <= 1.10
    assert ctx["active_component_count"] >= 5
    assert ctx["components"]["global_governor"]["found"] is True
    assert ctx["components"]["meta_mix_reliability_governor"]["found"] is True
    assert ctx["components"]["hive_persistence_governor"]["found"] is True
    assert ctx["components"]["dna_stress_governor"]["found"] is True
    assert ctx["components"]["reflex_health_governor"]["found"] is True
    assert ctx["components"]["symbolic_governor"]["found"] is True
    assert ctx["components"]["meta_mix_quality_modifier"]["found"] is True
    assert ctx["components"]["meta_mix_disagreement_modifier"]["found"] is True
    assert ctx["components"]["meta_mix_alpha_balance_modifier"]["found"] is True
    assert ctx["components"]["meta_mix_gross_modifier"]["found"] is True
    assert ctx["components"]["regime_fracture_governor"]["found"] is True
    assert "fracture_warn" in ctx["risk_flags"]


def test_runtime_context_defaults_to_neutral_when_missing(tmp_path: Path):
    ctx = ex._runtime_context(tmp_path)
    assert ctx["runtime_multiplier"] == 1.0
    assert ctx["active_component_count"] == 0
    assert ctx["regime"] == "risk_on"


def test_runtime_context_includes_drift_and_quality_step_modifiers(tmp_path: Path):
    (tmp_path / "quality_snapshot.json").write_text(
        '{"quality_governor_max_abs_step": 0.16}',
        encoding="utf-8",
    )
    (tmp_path / "portfolio_drift_watch.json").write_text(
        '{"drift":{"status":"alert","latest_over_p95":3.5}}',
        encoding="utf-8",
    )

    ctx = ex._runtime_context(tmp_path)
    assert ctx["components"]["quality_governor_step_modifier"]["found"] is True
    assert ctx["components"]["portfolio_drift_modifier"]["found"] is True
    assert "drift_alert" in ctx["risk_flags"]
    assert "quality_governor_step_spike" in ctx["risk_flags"]
    assert 0.50 <= ctx["runtime_multiplier"] <= 1.0


def test_runtime_context_flags_council_divergence_alert(tmp_path: Path):
    _write_series(tmp_path / "meta_mix_disagreement.csv", [0.30, 0.86])
    ctx = ex._runtime_context(tmp_path)
    assert ctx["components"]["meta_mix_disagreement_modifier"]["found"] is True
    assert "council_divergence_alert" in ctx["risk_flags"]
    assert "council_divergence_warn" not in ctx["risk_flags"]


def test_runtime_context_includes_execution_adaptive_risk_modifier(tmp_path: Path):
    (tmp_path / "execution_constraints_info.json").write_text(
        '{"adaptive_risk_scale": 0.52}',
        encoding="utf-8",
    )
    ctx = ex._runtime_context(tmp_path)
    assert ctx["components"]["execution_adaptive_risk_modifier"]["found"] is True
    assert "exec_risk_hard" in ctx["risk_flags"]


def test_runtime_context_flags_nested_leakage_alert(tmp_path: Path):
    (tmp_path / "nested_wf_summary.json").write_text(
        (
            '{"assets":12,"avg_outer_fold_utilization":0.28,'
            '"low_utilization_assets":9,"avg_train_ratio_mean":0.55,'
            '"params":{"purge_embargo_ratio":0.92}}'
        ),
        encoding="utf-8",
    )
    ctx = ex._runtime_context(tmp_path)
    assert ctx["components"]["nested_wf_leakage_modifier"]["found"] is True
    assert "nested_leakage_alert" in ctx["risk_flags"]


def test_runtime_context_parses_dated_governor_csvs(tmp_path: Path):
    (tmp_path / "hive_diversification_governor.csv").write_text(
        "DATE,hive_diversification_governor\n2026-01-01,1.01\n2026-01-02,0.99\n",
        encoding="utf-8",
    )
    (tmp_path / "hive_persistence_governor.csv").write_text(
        "DATE,hive_persistence_governor\n2026-01-01,1.02\n2026-01-02,1.00\n",
        encoding="utf-8",
    )
    (tmp_path / "dream_coherence_governor.csv").write_text(
        "DATE,dream_coherence_governor\n2026-01-01,1.04\n2026-01-02,0.97\n",
        encoding="utf-8",
    )
    (tmp_path / "heartbeat_exposure_scaler.csv").write_text(
        "DATE,heartbeat_exposure_scaler\n2026-01-01,0.92\n2026-01-02,0.88\n",
        encoding="utf-8",
    )
    (tmp_path / "dna_stress_governor.csv").write_text(
        "DATE,dna_stress,dna_stress_governor\n2026-01-01,0.1,1.03\n2026-01-02,0.3,0.95\n",
        encoding="utf-8",
    )
    (tmp_path / "symbolic_governor.csv").write_text(
        "DATE,symbolic_stress,symbolic_governor\n2026-01-01,0.2,1.01\n2026-01-02,0.4,0.93\n",
        encoding="utf-8",
    )

    ctx = ex._runtime_context(tmp_path)
    assert ctx["components"]["hive_diversification_governor"]["found"] is True
    assert ctx["components"]["hive_persistence_governor"]["found"] is True
    assert ctx["components"]["dream_coherence_governor"]["found"] is True
    assert ctx["components"]["heartbeat_exposure_scaler"]["found"] is True
    assert ctx["components"]["dna_stress_governor"]["found"] is True
    assert ctx["components"]["symbolic_governor"]["found"] is True


def test_runtime_context_includes_hive_ecosystem_stress(tmp_path: Path):
    (tmp_path / "cross_hive_summary.json").write_text(
        (
            '{"adaptive_diagnostics":{'
            '"mean_disagreement":0.82,'
            '"mean_stability_dispersion":0.81,'
            '"mean_regime_fracture":0.34},'
            '"entropy_adaptive_diagnostics":{'
            '"entropy_target_mean":0.78,'
            '"entropy_target_max":0.89,'
            '"entropy_strength_mean":0.84,'
            '"entropy_strength_max":0.94},'
            '"crowding_penalty_mean":{"EQ":0.66,"FX":0.60,"RATES":0.58}}'
        ),
        encoding="utf-8",
    )
    (tmp_path / "hive_evolution.json").write_text(
        (
            '{"action_pressure_mean":0.22,'
            '"latest_vitality":{"EQ":0.38,"FX":0.44,"RATES":0.41}}'
        ),
        encoding="utf-8",
    )

    ctx = ex._runtime_context(tmp_path)
    assert ctx["components"]["hive_ecosystem_stability_modifier"]["found"] is True
    assert ctx["components"]["hive_crowding_modifier"]["found"] is True
    assert ctx["components"]["hive_entropy_pressure_modifier"]["found"] is True
    assert ctx["components"]["hive_evolution_modifier"]["found"] is True
    assert "hive_stress_alert" in ctx["risk_flags"]
    assert "hive_crowding_alert" in ctx["risk_flags"]
    assert "hive_entropy_alert" in ctx["risk_flags"]
    assert ctx["risk_flags"].count("hive_stress_alert") == 1
    assert "hive_stress_warn" not in ctx["risk_flags"]


def test_runtime_context_includes_heartbeat_stress_flags(tmp_path: Path):
    _write_series(tmp_path / "heartbeat_stress.csv", [0.55, 0.86])
    _write_series(tmp_path / "heartbeat_bpm.csv", [88, 89, 91, 93, 96, 99])

    ctx = ex._runtime_context(tmp_path)
    assert ctx["components"]["heartbeat_stress_modifier"]["found"] is True
    assert "heartbeat_alert" in ctx["risk_flags"]
    assert "heartbeat_warn" not in ctx["risk_flags"]


def test_runtime_context_includes_novaspine_memory_feedback(tmp_path: Path):
    (tmp_path / "novaspine_context.json").write_text(
        (
            '{"enabled": true, "status": "ok", '
            '"context_resonance": 0.04, "context_boost": 0.92}'
        ),
        encoding="utf-8",
    )
    (tmp_path / "novaspine_hive_feedback.json").write_text(
        (
            '{"enabled": true, "status": "ok", "global_boost": 0.93}'
        ),
        encoding="utf-8",
    )

    ctx = ex._runtime_context(tmp_path)
    mf = ctx.get("memory_feedback", {})
    assert mf.get("active") is True
    assert mf.get("status") in {"warn", "alert"}
    assert float(mf.get("risk_scale")) < 1.0
    assert ctx["components"]["novaspine_memory_feedback_modifier"]["found"] is True
    assert any(f in ctx["risk_flags"] for f in ["memory_feedback_warn", "memory_feedback_alert"])


def test_runtime_context_includes_aion_outcome_feedback(tmp_path: Path, monkeypatch):
    shadow = tmp_path / "shadow_trades.csv"
    shadow.write_text(
        "\n".join(
            [
                "timestamp,symbol,side,pnl",
                "2026-02-01 10:00:00,AAPL,EXIT_SELL,-12.0",
                "2026-02-01 10:05:00,MSFT,EXIT_BUY,-8.0",
                "2026-02-01 10:10:00,NVDA,EXIT_SELL,-11.0",
                "2026-02-01 10:15:00,TSLA,EXIT_BUY,-9.0",
                "2026-02-01 10:20:00,AMZN,EXIT_SELL,-10.0",
                "2026-02-01 10:25:00,META,EXIT_BUY,-7.0",
                "2026-02-01 10:30:00,GOOG,EXIT_SELL,-9.5",
                "2026-02-01 10:35:00,AMD,EXIT_BUY,-8.5",
                "2026-02-01 10:40:00,NFLX,EXIT_SELL,-10.5",
                "2026-02-01 10:45:00,IBM,EXIT_BUY,-6.0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("Q_AION_SHADOW_TRADES", str(shadow))
    monkeypatch.setenv("Q_AION_FEEDBACK_LOOKBACK", "30")
    monkeypatch.setenv("Q_AION_FEEDBACK_MIN_TRADES", "8")

    ctx = ex._runtime_context(tmp_path)
    assert ctx["components"]["aion_outcome_modifier"]["found"] is True
    assert "aion_outcome_alert" in ctx["risk_flags"]
    afb = ctx.get("aion_feedback", {})
    assert afb.get("active") is True
    assert afb.get("status") == "alert"
    assert float(afb.get("risk_scale", 1.0)) < 1.0
