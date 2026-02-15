from pathlib import Path

import tools.export_aion_signal_pack as ex


def _write_series(path: Path, vals):
    path.write_text("\n".join(str(v) for v in vals), encoding="utf-8")


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
