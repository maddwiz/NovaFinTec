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
