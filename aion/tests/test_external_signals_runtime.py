import json
from pathlib import Path

from aion.brain.external_signals import load_external_signal_bundle, runtime_overlay_scale


def test_load_external_signal_bundle_reads_runtime_context(tmp_path: Path):
    p = tmp_path / "overlay.json"
    p.write_text(
        json.dumps(
            {
                "global": {"bias": 0.2, "confidence": 0.8},
                "signals": {"AAPL": {"bias": 0.5, "confidence": 0.7}},
                "runtime_context": {"runtime_multiplier": 0.84, "regime": "balanced", "risk_flags": ["drift_warn"]},
                "source_mode": "wf_table",
                "degraded_safe_mode": False,
                "quality_gate": {"ok": True},
            }
        ),
        encoding="utf-8",
    )

    b = load_external_signal_bundle(p, min_confidence=0.55, max_bias=0.9)
    assert "AAPL" in b["signals"]
    assert "__GLOBAL__" in b["signals"]
    assert b["runtime_multiplier"] == 0.84
    assert b["regime"] == "balanced"
    assert b["source_mode"] == "wf_table"
    assert "drift_warn" in b["risk_flags"]
    assert b["quality_gate_ok"] is True


def test_runtime_overlay_scale_penalizes_flags_and_degraded():
    scale, diag = runtime_overlay_scale(
        {
            "runtime_multiplier": 0.95,
            "risk_flags": ["drift_alert", "quality_governor_step_spike"],
            "degraded_safe_mode": True,
            "quality_gate_ok": False,
            "regime": "defensive",
            "source_mode": "final_weights_fallback",
        },
        min_scale=0.55,
        max_scale=1.05,
        degraded_scale=0.70,
        quality_fail_scale=0.82,
        flag_scale=0.90,
    )
    assert 0.55 <= scale < 0.95
    assert diag["active"] is True
    assert diag["degraded"] is True
    assert diag["quality_gate_ok"] is False
    assert diag["source_mode"] == "final_weights_fallback"
    assert "drift_alert" in diag["flags"]


def test_runtime_overlay_scale_applies_fracture_alert_penalty():
    scale_warn, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["fracture_warn"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.40,
        max_scale=1.10,
        flag_scale=0.95,
        fracture_warn_scale=0.90,
        fracture_alert_scale=0.70,
    )
    scale_alert, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["fracture_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.40,
        max_scale=1.10,
        flag_scale=0.95,
        fracture_warn_scale=0.90,
        fracture_alert_scale=0.70,
    )
    assert scale_alert < scale_warn < 1.0


def test_runtime_overlay_scale_applies_exec_risk_hard_penalty():
    scale_tight, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["exec_risk_tight"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.35,
        max_scale=1.10,
        flag_scale=0.96,
        exec_risk_tight_scale=0.86,
        exec_risk_hard_scale=0.70,
    )
    scale_hard, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["exec_risk_hard"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.35,
        max_scale=1.10,
        flag_scale=0.96,
        exec_risk_tight_scale=0.86,
        exec_risk_hard_scale=0.70,
    )
    assert scale_hard < scale_tight < 1.0


def test_runtime_overlay_scale_applies_nested_leakage_penalty():
    scale_warn, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["nested_leakage_warn"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.35,
        max_scale=1.10,
        flag_scale=0.96,
        nested_leak_warn_scale=0.90,
        nested_leak_alert_scale=0.76,
    )
    scale_alert, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["nested_leakage_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.35,
        max_scale=1.10,
        flag_scale=0.96,
        nested_leak_warn_scale=0.90,
        nested_leak_alert_scale=0.76,
    )
    assert scale_alert < scale_warn < 1.0


def test_runtime_overlay_scale_applies_drift_and_quality_step_penalties():
    scale_warn, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["drift_warn"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.35,
        max_scale=1.10,
        flag_scale=0.97,
        drift_warn_scale=0.92,
        drift_alert_scale=0.80,
    )
    scale_alert, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["drift_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.35,
        max_scale=1.10,
        flag_scale=0.97,
        drift_warn_scale=0.92,
        drift_alert_scale=0.80,
    )
    scale_step, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["quality_governor_step_spike"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.35,
        max_scale=1.10,
        flag_scale=0.97,
        quality_step_spike_scale=0.85,
    )
    assert scale_alert < scale_warn < 1.0
    assert 0.35 <= scale_step < 1.0


def test_runtime_overlay_scale_applies_hive_stress_penalty():
    scale_warn, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["hive_stress_warn"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.35,
        max_scale=1.10,
        flag_scale=0.96,
        hive_stress_warn_scale=0.90,
        hive_stress_alert_scale=0.74,
    )
    scale_alert, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["hive_stress_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.35,
        max_scale=1.10,
        flag_scale=0.96,
        hive_stress_warn_scale=0.90,
        hive_stress_alert_scale=0.74,
    )
    assert scale_alert < scale_warn < 1.0
