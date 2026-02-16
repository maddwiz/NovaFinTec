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
            "runtime_context": {
                "runtime_multiplier": 0.9,
                "regime": "balanced",
                "aion_feedback": {
                    "active": True,
                    "status": "warn",
                    "risk_scale": 0.88,
                    "closed_trades": 16,
                    "hit_rate": 0.43,
                    "profit_factor": 0.91,
                    "expectancy": -0.8,
                    "drawdown_norm": 1.9,
                    "age_hours": 18.0,
                    "max_age_hours": 72.0,
                    "stale": False,
                    "last_closed_ts": "2026-02-16T15:35:00Z",
                    "reasons": ["low_profit_factor_warn"],
                    "path": "/tmp/shadow_trades.csv",
                },
            },
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
            "adaptive_enabled": True,
            "mean_alpha": 0.54,
            "mean_gross_dynamic": 0.21,
            "mean_quality_mix": 0.61,
            "mean_disagreement_norm": 0.34,
        },
    )
    _write_json(tmp_path / "dream_coherence_info.json", {"status": "ok", "signals": ["reflex_latent", "meta_mix"], "mean_coherence": 0.62, "mean_governor": 0.93})
    _write_json(tmp_path / "dna_stress_info.json", {"status": "ok", "mean_stress": 0.41, "max_stress": 0.77})
    _write_json(tmp_path / "reflex_health_info.json", {"health_mean": 0.9, "health_max": 1.8, "governor_mean": 0.94})
    _write_json(tmp_path / "symbolic_governor_info.json", {"status": "ok", "mean_stress": 0.37, "max_stress": 0.72})
    _write_json(
        tmp_path / "cross_hive_summary.json",
        {
            "hives": ["EQ", "FX"],
            "mean_turnover": 0.2,
            "latest_weights": {"EQ": 0.6},
            "crowding_penalty_mean": {"EQ": 0.62, "FX": 0.48},
            "adaptive_diagnostics": {
                "mean_disagreement": 0.41,
                "mean_stability_dispersion": 0.37,
                "mean_regime_fracture": 0.19,
            },
            "entropy_adaptive_diagnostics": {
                "entropy_target_mean": 0.76,
                "entropy_target_max": 0.89,
                "entropy_strength_mean": 0.80,
                "entropy_strength_max": 0.93,
            },
        },
    )
    _write_json(tmp_path / "hive_evolution.json", {"events": [{"event": "split_applied"}], "action_pressure_mean": 0.17})
    _write_json(tmp_path / "execution_constraints_info.json", {"gross_after_mean": 0.9})
    _write_json(tmp_path / "immune_drill.json", {"pass": True, "ok": True})
    _write_json(tmp_path / "guardrails_summary.json", {"turnover_cost": {"turnover_budget": {"enabled": True, "limit": 1.0}}})
    _write_json(tmp_path / "final_portfolio_info.json", {"steps": ["quality_governor", "concentration_governor"]})
    _write_json(tmp_path / "concentration_governor_info.json", {"enabled": True, "top1_cap": 0.18, "top3_cap": 0.42, "max_hhi": 0.14, "stats": {"hhi_after": 0.11}})
    _write_json(tmp_path / "shock_mask_info.json", {"shock_days": 10, "shock_rate": 0.05, "params": {"z": 2.5}})
    _write_json(tmp_path / "regime_fracture_info.json", {"state": "fracture_warn", "latest_score": 0.76, "latest_governor": 0.84, "risk_flags": ["fracture_warn"]})
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
    np.savetxt(tmp_path / "hive_persistence_governor.csv", np.array([1.00, 0.97], float), delimiter=",")

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
    af = rc.get("payload", {}).get("aion_feedback", {})
    assert af.get("active") is True
    assert af.get("status") == "warn"
    assert af.get("source") == "overlay"
    assert int(af.get("closed_trades")) == 16
    assert float(af.get("risk_scale")) > 0.0
    assert float(af.get("age_hours")) >= 0.0
    assert af.get("stale") is False
    assert af.get("last_closed_ts") == "2026-02-16T15:35:00Z"
    dream = rc.get("payload", {}).get("dream_coherence", {})
    assert dream.get("status") == "ok"
    assert float(dream.get("mean_coherence")) > 0.0
    hb = rc.get("payload", {}).get("heartbeat_stress", {})
    assert float(hb.get("latest")) > 0.0
    assert float(hb.get("mean")) > 0.0
    frac = rc.get("payload", {}).get("regime_fracture", {})
    assert frac.get("state") == "fracture_warn"
    assert float(frac.get("latest_score")) > 0.0
    mmr = rc.get("payload", {}).get("meta_mix_reliability", {})
    assert float(mmr.get("mean_governor")) > 0.0
    assert float(mmr.get("mean_confidence_calibrated")) > 0.0
    assert float(mmr.get("mean_alpha")) > 0.0
    assert float(mmr.get("mean_quality_mix")) > 0.0
    dna = rc.get("payload", {}).get("dna_stress", {})
    assert dna.get("status") == "ok"
    assert float(dna.get("mean_stress")) > 0.0
    rx = rc.get("payload", {}).get("reflex_health", {})
    assert float(rx.get("health_mean")) > 0.0
    assert float(rx.get("mean_governor")) > 0.0
    sym = rc.get("payload", {}).get("symbolic", {})
    assert sym.get("status") == "ok"
    assert float(sym.get("mean_stress")) > 0.0
    hp = rc.get("payload", {}).get("hive_persistence", {})
    assert float(hp.get("mean_governor")) > 0.0
    hc = rc.get("payload", {}).get("hive_crowding", {})
    assert hc.get("top_hive") == "EQ"
    assert float(hc.get("mean_penalty")) > 0.0
    he = rc.get("payload", {}).get("hive_entropy", {})
    assert float(he.get("entropy_target_mean")) > 0.0
    assert float(he.get("entropy_strength_max")) > 0.0
    chs = rc.get("payload", {}).get("cross_hive_stability", {})
    assert float(chs.get("mean_turnover")) > 0.0
    assert float(chs.get("mean_disagreement")) > 0.0
    drc = [e for e in events if e.get("event_type") == "decision.runtime_context"][0]
    dfrac = drc.get("payload", {}).get("regime_fracture", {})
    assert dfrac.get("state") == "fracture_warn"
    dch = drc.get("payload", {}).get("cross_hive", {})
    assert dch.get("crowding", {}).get("top_hive") == "EQ"
    assert float(dch.get("entropy", {}).get("entropy_strength_mean")) > 0.0
    assert float(dch.get("stability", {}).get("mean_turnover")) > 0.0
    mfb = [e for e in events if e.get("event_type") == "memory.feedback_state"][0]
    af2 = mfb.get("payload", {}).get("aion_feedback", {})
    assert af2.get("status") == "warn"
    assert af2.get("source") == "overlay"
    assert int(af2.get("closed_trades")) == 16
    assert float(af2.get("age_hours")) >= 0.0
    assert af2.get("last_closed_ts") == "2026-02-16T15:35:00Z"
    trusts = [float(e.get("trust", 0.0)) for e in events]
    assert all(0.0 <= t <= 1.0 for t in trusts)


def test_build_events_works_with_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "RUNS", tmp_path)
    events = sm.build_events()
    assert len(events) >= 4
    assert any(e.get("event_type") == "governance.health_gate" for e in events)


def test_build_events_falls_back_to_shadow_trades_aion_feedback(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "RUNS", tmp_path)
    shadow = tmp_path / "shadow_trades.csv"
    shadow.write_text(
        "\n".join(
            [
                "timestamp,symbol,side,pnl",
                "2026-02-16 10:00:00,AAPL,EXIT_BUY,-6.0",
                "2026-02-16 10:05:00,MSFT,EXIT_SELL,-5.0",
                "2026-02-16 10:10:00,NVDA,PARTIAL_BUY,-4.0",
                "2026-02-16 10:15:00,TSLA,EXIT_SELL,-3.0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("Q_AION_SHADOW_TRADES", str(shadow))
    monkeypatch.setenv("Q_AION_FEEDBACK_MIN_TRADES", "3")
    _write_json(tmp_path / "q_signal_overlay.json", {"global": {"confidence": 0.55, "bias": 0.1}})

    events = sm.build_events()
    rc = [e for e in events if e.get("event_type") == "governance.risk_controls"][0]
    af = rc.get("payload", {}).get("aion_feedback", {})
    assert af.get("source") == "shadow_trades"
    assert af.get("active") is True
    assert int(af.get("closed_trades")) == 4
    assert float(af.get("risk_scale")) < 1.0
