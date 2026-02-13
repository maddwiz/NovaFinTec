import json

import numpy as np

import tools.sync_novaspine_memory as sm


def _write_json(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_build_events_includes_governance_audit_events(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "RUNS", tmp_path)

    _write_json(
        tmp_path / "q_signal_overlay.json",
        {
            "global": {"confidence": 0.7, "bias": 0.2},
            "coverage": {"symbols": 5},
            "runtime_context": {"runtime_multiplier": 0.9, "regime": "balanced"},
        },
    )
    _write_json(tmp_path / "system_health.json", {"health_score": 88})
    _write_json(tmp_path / "health_alerts.json", {"ok": True, "alerts": []})
    _write_json(tmp_path / "quality_snapshot.json", {"quality_score": 0.66, "quality_governor_mean": 0.84})
    _write_json(
        tmp_path / "meta_mix_info.json",
        {
            "mean_confidence_raw": 0.58,
            "mean_confidence_calibrated": 0.62,
            "brier_raw": 0.241,
            "brier_calibrated": 0.229,
        },
    )
    _write_json(tmp_path / "dream_coherence_info.json", {"status": "ok", "signals": ["reflex_latent", "meta_mix"], "mean_coherence": 0.62, "mean_governor": 0.93})
    _write_json(tmp_path / "dna_stress_info.json", {"status": "ok", "mean_stress": 0.41, "max_stress": 0.77})
    _write_json(tmp_path / "reflex_health_info.json", {"health_mean": 0.9, "health_max": 1.8, "governor_mean": 0.94})
    _write_json(tmp_path / "symbolic_governor_info.json", {"status": "ok", "mean_stress": 0.37, "max_stress": 0.72})
    _write_json(tmp_path / "cross_hive_summary.json", {"hives": ["EQ", "FX"], "mean_turnover": 0.2, "latest_weights": {"EQ": 0.6}})
    _write_json(tmp_path / "hive_evolution.json", {"events": [{"event": "split_applied"}]})
    _write_json(tmp_path / "execution_constraints_info.json", {"gross_after_mean": 0.9})
    _write_json(tmp_path / "immune_drill.json", {"pass": True, "ok": True})
    _write_json(tmp_path / "guardrails_summary.json", {"turnover_cost": {"turnover_budget": {"enabled": True, "limit": 1.0}}})
    _write_json(tmp_path / "final_portfolio_info.json", {"steps": ["quality_governor", "concentration_governor"]})
    _write_json(tmp_path / "concentration_governor_info.json", {"enabled": True, "top1_cap": 0.18, "top3_cap": 0.42, "max_hhi": 0.14, "stats": {"hhi_after": 0.11}})
    _write_json(tmp_path / "shock_mask_info.json", {"shock_days": 10, "shock_rate": 0.05, "params": {"z": 2.5}})
    _write_json(tmp_path / "pipeline_status.json", {"failed_count": 0})
    _write_json(tmp_path / "novaspine_context.json", {"status": "ok", "context_resonance": 0.6, "context_boost": 1.03})
    _write_json(tmp_path / "novaspine_hive_feedback.json", {"status": "ok", "global_boost": 1.02, "per_hive": {"EQ": {"boost": 1.04}}})
    _write_json(tmp_path / "hive_transparency.json", {"summary": {"hive_count": 2}})
    _write_json(tmp_path / "portfolio_drift_watch.json", {"drift": {"status": "ok", "latest_l1": 0.4, "mean_l1": 0.2, "p95_l1": 0.35}})
    np.savetxt(tmp_path / "final_governor_trace.csv", np.array([[1.0, 0.9], [1.0, 0.85]], float), delimiter=",")
    np.savetxt(tmp_path / "heartbeat_stress.csv", np.array([0.45, 0.60], float), delimiter=",")
    np.savetxt(tmp_path / "meta_mix_reliability_governor.csv", np.array([0.95, 1.01], float), delimiter=",")
    np.savetxt(tmp_path / "dna_stress_governor.csv", np.array([1.00, 0.92], float), delimiter=",")
    np.savetxt(tmp_path / "reflex_health_governor.csv", np.array([0.95, 1.01], float), delimiter=",")
    np.savetxt(tmp_path / "symbolic_governor.csv", np.array([0.94, 0.99], float), delimiter=",")

    np.savetxt(tmp_path / "portfolio_weights_final.csv", np.array([[0.1, -0.1], [0.2, -0.2]], float), delimiter=",")

    events = sm.build_events()
    types = {e.get("event_type") for e in events}

    assert "governance.risk_controls" in types
    assert "decision.runtime_context" in types
    assert "memory.feedback_state" in types
    assert "governance.immune_drill" in types
    rc = [e for e in events if e.get("event_type") == "governance.risk_controls"][0]
    rts = rc.get("payload", {}).get("runtime_total_scalar", {})
    assert float(rts.get("latest")) > 0.0
    pdw = rc.get("payload", {}).get("portfolio_drift_watch", {})
    assert pdw.get("status") == "ok"
    assert float(pdw.get("latest_l1")) > 0.0
    dream = rc.get("payload", {}).get("dream_coherence", {})
    assert dream.get("status") == "ok"
    assert float(dream.get("mean_coherence")) > 0.0
    hb = rc.get("payload", {}).get("heartbeat_stress", {})
    assert float(hb.get("latest")) > 0.0
    assert float(hb.get("mean")) > 0.0
    mmr = rc.get("payload", {}).get("meta_mix_reliability", {})
    assert float(mmr.get("mean_governor")) > 0.0
    assert float(mmr.get("mean_confidence_calibrated")) > 0.0
    dna = rc.get("payload", {}).get("dna_stress", {})
    assert dna.get("status") == "ok"
    assert float(dna.get("mean_stress")) > 0.0
    rx = rc.get("payload", {}).get("reflex_health", {})
    assert float(rx.get("health_mean")) > 0.0
    assert float(rx.get("mean_governor")) > 0.0
    sym = rc.get("payload", {}).get("symbolic", {})
    assert sym.get("status") == "ok"
    assert float(sym.get("mean_stress")) > 0.0
    trusts = [float(e.get("trust", 0.0)) for e in events]
    assert all(0.0 <= t <= 1.0 for t in trusts)


def test_build_events_works_with_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "RUNS", tmp_path)
    events = sm.build_events()
    assert len(events) >= 4
    assert any(e.get("event_type") == "governance.health_gate" for e in events)
