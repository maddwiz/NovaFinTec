import json
import os
from datetime import datetime, timedelta, timezone
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
    assert isinstance(b["overlay_age_hours"], float)
    assert b["overlay_age_source"] in {"payload", "mtime", "unknown"}


def test_load_external_signal_bundle_parses_memory_feedback(tmp_path: Path):
    p = tmp_path / "overlay.json"
    p.write_text(
        json.dumps(
            {
                "signals": {"AAPL": {"bias": 0.3, "confidence": 0.8}},
                "runtime_context": {
                    "runtime_multiplier": 0.92,
                    "risk_flags": ["memory_feedback_warn"],
                    "memory_feedback": {
                        "active": True,
                        "status": "warn",
                        "risk_scale": 0.90,
                        "max_trades_scale": 0.88,
                        "max_open_scale": 0.90,
                        "block_new_entries": False,
                        "reasons": ["memory_partial_or_disabled"],
                    },
                },
                "quality_gate": {"ok": True},
            }
        ),
        encoding="utf-8",
    )

    b = load_external_signal_bundle(p, min_confidence=0.55, max_bias=0.9)
    mf = b.get("memory_feedback", {})
    assert mf.get("active") is True
    assert mf.get("status") == "warn"
    assert float(mf.get("risk_scale")) < 1.0
    assert "memory_feedback_warn" in b.get("risk_flags", [])


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


def test_runtime_overlay_scale_applies_hive_crowding_penalty():
    scale_warn, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["hive_crowding_warn"],
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
            "risk_flags": ["hive_crowding_alert"],
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


def test_runtime_overlay_scale_applies_hive_entropy_penalty():
    scale_warn, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["hive_entropy_warn"],
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
            "risk_flags": ["hive_entropy_alert"],
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


def test_runtime_overlay_scale_applies_heartbeat_penalty():
    scale_warn, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["heartbeat_warn"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.35,
        max_scale=1.10,
        flag_scale=0.96,
        heartbeat_warn_scale=0.88,
        heartbeat_alert_scale=0.72,
    )
    scale_alert, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["heartbeat_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.35,
        max_scale=1.10,
        flag_scale=0.96,
        heartbeat_warn_scale=0.88,
        heartbeat_alert_scale=0.72,
    )
    assert scale_alert < scale_warn < 1.0


def test_runtime_overlay_scale_applies_council_divergence_penalty():
    scale_warn, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["council_divergence_warn"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.35,
        max_scale=1.10,
        flag_scale=0.96,
        council_divergence_warn_scale=0.90,
        council_divergence_alert_scale=0.74,
    )
    scale_alert, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["council_divergence_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.35,
        max_scale=1.10,
        flag_scale=0.96,
        council_divergence_warn_scale=0.90,
        council_divergence_alert_scale=0.74,
    )
    assert scale_alert < scale_warn < 1.0


def test_runtime_overlay_scale_ignores_duplicate_flags():
    base, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["hive_stress_alert", "drift_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.30,
        max_scale=1.10,
        flag_scale=0.95,
    )
    dupes, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["hive_stress_alert", "hive_stress_alert", "drift_alert", "drift_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.30,
        max_scale=1.10,
        flag_scale=0.95,
    )
    assert dupes == base


def test_runtime_overlay_scale_uses_stronger_flag_when_warn_and_alert_both_present():
    scale_alert, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["hive_stress_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.30,
        max_scale=1.10,
        flag_scale=0.95,
    )
    scale_both, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["hive_stress_warn", "hive_stress_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.30,
        max_scale=1.10,
        flag_scale=0.95,
    )
    assert scale_both == scale_alert


def test_runtime_overlay_scale_uses_stronger_crowding_flag_when_warn_and_alert_present():
    scale_alert, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["hive_crowding_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.30,
        max_scale=1.10,
        flag_scale=0.95,
    )
    scale_both, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["hive_crowding_warn", "hive_crowding_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.30,
        max_scale=1.10,
        flag_scale=0.95,
    )
    assert scale_both == scale_alert


def test_runtime_overlay_scale_uses_stronger_entropy_flag_when_warn_and_alert_present():
    scale_alert, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["hive_entropy_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.30,
        max_scale=1.10,
        flag_scale=0.95,
    )
    scale_both, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["hive_entropy_warn", "hive_entropy_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.30,
        max_scale=1.10,
        flag_scale=0.95,
    )
    assert scale_both == scale_alert


def test_runtime_overlay_scale_applies_aion_outcome_penalty():
    scale_warn, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["aion_outcome_warn"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.30,
        max_scale=1.10,
        flag_scale=0.96,
        aion_outcome_warn_scale=0.88,
        aion_outcome_alert_scale=0.72,
    )
    scale_alert, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["aion_outcome_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.30,
        max_scale=1.10,
        flag_scale=0.96,
        aion_outcome_warn_scale=0.88,
        aion_outcome_alert_scale=0.72,
    )
    assert scale_alert < scale_warn < 1.0


def test_runtime_overlay_scale_uses_stronger_aion_outcome_flag_when_warn_and_alert_present():
    scale_alert, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["aion_outcome_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.30,
        max_scale=1.10,
        flag_scale=0.95,
    )
    scale_both, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["aion_outcome_warn", "aion_outcome_alert"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
        },
        min_scale=0.30,
        max_scale=1.10,
        flag_scale=0.95,
    )
    assert scale_both == scale_alert


def test_load_external_signal_bundle_marks_stale_overlay_and_drops_signals(tmp_path: Path):
    p = tmp_path / "overlay.json"
    p.write_text(
        json.dumps(
            {
                "signals": {"AAPL": {"bias": 0.4, "confidence": 0.8}},
                "runtime_context": {"runtime_multiplier": 1.0, "risk_flags": []},
                "quality_gate": {"ok": True},
            }
        ),
        encoding="utf-8",
    )
    old_ts = 946684800  # 2000-01-01
    os.utime(p, (old_ts, old_ts))

    b = load_external_signal_bundle(p, min_confidence=0.55, max_bias=0.9, max_age_hours=1.0)
    assert b["overlay_stale"] is True
    assert "overlay_stale" in b["risk_flags"]
    assert b["signals"] == {}


def test_runtime_overlay_scale_penalizes_overlay_stale():
    clean, _ = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": [],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
            "overlay_stale": False,
        },
        min_scale=0.30,
        max_scale=1.10,
    )
    stale, diag = runtime_overlay_scale(
        {
            "runtime_multiplier": 1.0,
            "risk_flags": ["overlay_stale"],
            "degraded_safe_mode": False,
            "quality_gate_ok": True,
            "overlay_stale": True,
        },
        min_scale=0.30,
        max_scale=1.10,
        overlay_stale_scale=0.80,
    )
    assert stale < clean
    assert diag["overlay_stale"] is True
    assert "overlay_age_hours" in diag
    assert "overlay_age_source" in diag


def test_load_external_signal_bundle_prefers_payload_timestamp_over_mtime(tmp_path: Path):
    p = tmp_path / "overlay.json"
    fresh = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    p.write_text(
        json.dumps(
            {
                "generated_at_utc": fresh,
                "signals": {"AAPL": {"bias": 0.2, "confidence": 0.8}},
                "runtime_context": {"runtime_multiplier": 1.0, "risk_flags": []},
                "quality_gate": {"ok": True},
            }
        ),
        encoding="utf-8",
    )
    old_ts = 946684800  # 2000-01-01
    os.utime(p, (old_ts, old_ts))

    b = load_external_signal_bundle(p, min_confidence=0.55, max_bias=0.9, max_age_hours=1.0)
    assert b["overlay_age_source"] == "payload"
    assert b["overlay_stale"] is False
    assert "AAPL" in b["signals"]


def test_load_external_signal_bundle_marks_stale_from_payload_timestamp(tmp_path: Path):
    p = tmp_path / "overlay.json"
    stale = (datetime.now(timezone.utc) - timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    p.write_text(
        json.dumps(
            {
                "generated_at_utc": stale,
                "signals": {"AAPL": {"bias": 0.2, "confidence": 0.8}},
                "runtime_context": {"runtime_multiplier": 1.0, "risk_flags": []},
                "quality_gate": {"ok": True},
            }
        ),
        encoding="utf-8",
    )
    # Make mtime fresh to ensure stale decision comes from payload timestamp.
    now_ts = datetime.now(timezone.utc).timestamp()
    os.utime(p, (now_ts, now_ts))

    b = load_external_signal_bundle(p, min_confidence=0.55, max_bias=0.9, max_age_hours=12.0)
    assert b["overlay_age_source"] == "payload"
    assert b["overlay_stale"] is True
    assert "overlay_stale" in b["risk_flags"]
    assert b["signals"] == {}
