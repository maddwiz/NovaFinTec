import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from aion.exec.doctor import _overlay_remediation, check_external_overlay


def _write(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_check_external_overlay_ok(tmp_path: Path):
    p = tmp_path / "overlay.json"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write(
        p,
        {
            "generated_at_utc": ts,
            "signals": {"AAPL": {"bias": 0.2, "confidence": 0.7}},
            "quality_gate": {"ok": True},
            "runtime_context": {"runtime_multiplier": 0.9, "risk_flags": []},
            "degraded_safe_mode": False,
        },
    )
    ok, msg, details = check_external_overlay(p, max_age_hours=24.0, require_runtime_context=True)
    assert ok is True
    assert "healthy" in msg.lower()
    assert details["signals"] == 1
    assert details["runtime_context_present"] is True
    assert details["age_source"] == "payload"
    assert details["generated_at_utc"] == ts


def test_check_external_overlay_flags_degraded_and_qgate(tmp_path: Path):
    p = tmp_path / "overlay.json"
    _write(
        p,
        {
            "signals": {"AAPL": {"bias": 0.2, "confidence": 0.7}},
            "quality_gate": {"ok": False},
            "runtime_context": {"runtime_multiplier": 0.7, "risk_flags": ["drift_alert"]},
            "degraded_safe_mode": True,
        },
    )
    ok, msg, details = check_external_overlay(p, max_age_hours=24.0, require_runtime_context=True)
    assert ok is False
    assert "degraded_safe_mode=true" in msg
    assert "quality_gate_not_ok" in msg
    assert details["quality_gate_ok"] is False


def test_check_external_overlay_requires_runtime_context(tmp_path: Path):
    p = tmp_path / "overlay.json"
    _write(
        p,
        {
            "signals": {"AAPL": {"bias": 0.2, "confidence": 0.7}},
            "quality_gate": {"ok": True},
            "degraded_safe_mode": False,
        },
    )
    ok, msg, _details = check_external_overlay(p, max_age_hours=24.0, require_runtime_context=True)
    assert ok is False
    assert "runtime_context_missing" in msg


def test_check_external_overlay_adds_overlay_stale_risk_flag(tmp_path: Path):
    p = tmp_path / "overlay.json"
    _write(
        p,
        {
            "signals": {"AAPL": {"bias": 0.2, "confidence": 0.7}},
            "quality_gate": {"ok": True},
            "runtime_context": {"runtime_multiplier": 0.9, "risk_flags": []},
            "degraded_safe_mode": False,
        },
    )
    old_ts = 946684800  # 2000-01-01 UTC
    os.utime(p, (old_ts, old_ts))
    ok, msg, details = check_external_overlay(p, max_age_hours=1.0, require_runtime_context=False)
    assert ok is False
    assert "stale>" in msg
    assert "overlay_stale" in details["risk_flags"]


def test_overlay_remediation_provides_actionable_tips(tmp_path: Path):
    checks = [
        {
            "name": "external_overlay",
            "ok": False,
            "details": {
                "exists": False,
                "age_hours": 25.0,
                "max_age_hours": 12.0,
                "degraded_safe_mode": True,
                "quality_gate_ok": False,
                "runtime_context_present": False,
                "risk_flags": ["fracture_alert"],
            },
        }
    ]
    tips = _overlay_remediation(checks, tmp_path / "q_signal_overlay.json")
    joined = " | ".join(tips).lower()
    assert "missing" in joined
    assert "stale" in joined
    assert "degraded" in joined
    assert "quality gate" in joined
    assert "runtime_context" in joined
    assert "fracture" in joined


def test_check_external_overlay_prefers_payload_timestamp_over_mtime(tmp_path: Path):
    p = tmp_path / "overlay.json"
    fresh = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write(
        p,
        {
            "generated_at_utc": fresh,
            "signals": {"AAPL": {"bias": 0.2, "confidence": 0.7}},
            "quality_gate": {"ok": True},
            "runtime_context": {"runtime_multiplier": 0.9, "risk_flags": []},
            "degraded_safe_mode": False,
        },
    )
    old_ts = 946684800
    os.utime(p, (old_ts, old_ts))
    ok, _msg, details = check_external_overlay(p, max_age_hours=1.0, require_runtime_context=True)
    assert ok is True
    assert details["age_source"] == "payload"


def test_check_external_overlay_stale_from_payload_timestamp(tmp_path: Path):
    p = tmp_path / "overlay.json"
    stale = (datetime.now(timezone.utc) - timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write(
        p,
        {
            "generated_at_utc": stale,
            "signals": {"AAPL": {"bias": 0.2, "confidence": 0.7}},
            "quality_gate": {"ok": True},
            "runtime_context": {"runtime_multiplier": 0.9, "risk_flags": []},
            "degraded_safe_mode": False,
        },
    )
    now_ts = datetime.now(timezone.utc).timestamp()
    os.utime(p, (now_ts, now_ts))
    ok, msg, details = check_external_overlay(p, max_age_hours=12.0, require_runtime_context=True)
    assert ok is False
    assert "stale>" in msg
    assert details["age_source"] == "payload"
    assert "overlay_stale" in details["risk_flags"]
