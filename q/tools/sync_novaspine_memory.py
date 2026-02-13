#!/usr/bin/env python3
# Optional NovaSpine bridge:
# - Builds enriched decision/governance events from runs_plus artifacts.
# - Publishes to NovaSpine API (preferred), filesystem outbox, or custom HTTP endpoint.
#
# Writes:
#   runs_plus/novaspine_last_batch.json
#   runs_plus/novaspine_events.jsonl
#   runs_plus/novaspine_sync_status.json

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.novaspine_adapter import publish_events  # noqa: E402

RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_matrix(path: Path):
    if not path.exists():
        return None
    try:
        a = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _load_series(path: Path):
    m = _load_matrix(path)
    if m is None or m.size == 0:
        return None
    return m[:, -1].ravel()


def _append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


def _latest_overlay_sample(limit: int = 12):
    p = RUNS / "q_signal_overlay.csv"
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p)
    except Exception:
        return []
    if "symbol" not in df.columns:
        return []
    cols = [c for c in ["symbol", "score", "confidence", "horizon"] if c in df.columns]
    if not cols:
        return []
    out = df[cols].head(max(1, int(limit))).copy()
    rec = []
    for _, r in out.iterrows():
        x = {k: r[k] for k in cols}
        for k, v in list(x.items()):
            if isinstance(v, (np.floating, np.integer)):
                x[k] = float(v)
        rec.append(x)
    return rec


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def build_events():
    ts = _now_iso()
    overlay = _load_json(RUNS / "q_signal_overlay.json") or {}
    health = _load_json(RUNS / "system_health.json") or {}
    alerts = _load_json(RUNS / "health_alerts.json") or {}
    quality = _load_json(RUNS / "quality_snapshot.json") or {}
    mix = _load_json(RUNS / "meta_mix_info.json") or {}
    dream = _load_json(RUNS / "dream_coherence_info.json") or {}
    cross = _load_json(RUNS / "cross_hive_summary.json") or {}
    eco = _load_json(RUNS / "hive_evolution.json") or {}
    constraints = _load_json(RUNS / "execution_constraints_info.json") or {}
    immune = _load_json(RUNS / "immune_drill.json") or {}
    guardrails = _load_json(RUNS / "guardrails_summary.json") or {}
    final_info = _load_json(RUNS / "final_portfolio_info.json") or {}
    conc = _load_json(RUNS / "concentration_governor_info.json") or {}
    shock = _load_json(RUNS / "shock_mask_info.json") or {}
    pipeline = _load_json(RUNS / "pipeline_status.json") or {}
    nctx = _load_json(RUNS / "novaspine_context.json") or {}
    nhive = _load_json(RUNS / "novaspine_hive_feedback.json") or {}
    htx = _load_json(RUNS / "hive_transparency.json") or {}
    drift_watch = _load_json(RUNS / "portfolio_drift_watch.json") or {}
    gov_trace = _load_series(RUNS / "final_governor_trace.csv")
    hb_stress = _load_series(RUNS / "heartbeat_stress.csv")
    meta_rel = _load_series(RUNS / "meta_mix_reliability_governor.csv")

    W = _load_matrix(RUNS / "portfolio_weights_final.csv")
    weights_info = {}
    if W is not None and W.size > 0:
        weights_info = {
            "rows": int(W.shape[0]),
            "cols": int(W.shape[1]),
            "abs_mean": float(np.mean(np.abs(W))),
            "gross_last": float(np.sum(np.abs(W[-1]))),
        }

    signal_rows = 0
    if isinstance(overlay, dict):
        signal_rows = int(overlay.get("signals_count", 0) or 0)
        if signal_rows <= 0:
            cov = overlay.get("coverage", {})
            if isinstance(cov, dict):
                signal_rows = int(cov.get("symbols", 0) or 0)
        if signal_rows <= 0:
            sig_map = overlay.get("signals", {})
            if isinstance(sig_map, dict):
                signal_rows = int(len(sig_map))
    fallback_mode = bool(overlay.get("degraded_safe_mode", False)) if isinstance(overlay, dict) else False
    overlay_conf = 0.0
    overlay_bias = 0.0
    if isinstance(overlay, dict):
        g = overlay.get("global", {})
        if isinstance(g, dict):
            overlay_conf = float(g.get("confidence", 0.0) or 0.0)
            overlay_bias = float(g.get("bias", 0.0) or 0.0)
        else:
            overlay_conf = float(overlay.get("global_confidence", 0.0) or 0.0)
            overlay_bias = float(overlay.get("global_bias", 0.0) or 0.0)

    latest_weights = cross.get("latest_weights", {}) if isinstance(cross, dict) else {}
    eco_events = eco.get("events", []) if isinstance(eco, dict) else []
    runtime_ctx = overlay.get("runtime_context", {}) if isinstance(overlay, dict) else {}
    turnover_budget = {}
    if isinstance(guardrails, dict):
        tc = guardrails.get("turnover_cost", {})
        if isinstance(tc, dict):
            tb = tc.get("turnover_budget", {})
            if isinstance(tb, dict):
                turnover_budget = tb
    conc_stats = conc.get("stats", {}) if isinstance(conc, dict) else {}
    quality_score = float(quality.get("quality_score", 0.5)) if isinstance(quality, dict) else 0.5
    quality_gov_mean = float(quality.get("quality_governor_mean", 1.0)) if isinstance(quality, dict) else 1.0

    events = [
        {
            "event_type": "decision.signal_export",
            "namespace": "private/c3/decisions",
            "ts_utc": ts,
            "payload": {
                "signals_count": signal_rows,
                "global_confidence": overlay_conf,
                "global_bias": overlay_bias,
                "degraded_safe_mode": fallback_mode,
                "signals_sample": _latest_overlay_sample(limit=12),
            },
            "trust": float(np.clip(overlay_conf, 0.0, 1.0)),
        },
        {
            "event_type": "governance.health_gate",
            "namespace": "private/c3/governance",
            "ts_utc": ts,
            "payload": {
                "health_score": float(health.get("health_score", 0.0)),
                "alerts_ok": bool(alerts.get("ok", False)),
                "alerts": list(alerts.get("alerts", []) or []),
                "quality_score": quality_score,
                "quality_governor_mean": quality_gov_mean,
            },
            "trust": float(np.clip(float(health.get("health_score", 0.0)) / 100.0, 0.0, 1.0)),
        },
        {
            "event_type": "ecosystem.hive_state",
            "namespace": "private/nova/actions",
            "ts_utc": ts,
            "payload": {
                "hives": list(cross.get("hives", []) or []),
                "latest_hive_weights": latest_weights,
                "mean_hive_turnover": float(cross.get("mean_turnover", 0.0)),
                "ecosystem_events_count": int(len(eco_events)),
                "ecosystem_events_sample": eco_events[:10],
            },
            "trust": float(np.clip(1.0 - min(1.0, float(cross.get("mean_turnover", 0.0))), 0.0, 1.0)),
        },
        {
            "event_type": "portfolio.runtime_state",
            "namespace": "private/nova/actions",
            "ts_utc": ts,
            "payload": {
                "weights": weights_info,
                "execution_constraints": constraints,
            },
            "trust": 0.8,
        },
        {
            "event_type": "governance.risk_controls",
            "namespace": "private/c3/governance",
            "ts_utc": ts,
            "payload": {
                "shock_mask": {
                    "shock_days": int(shock.get("shock_days", 0)) if isinstance(shock, dict) else 0,
                    "shock_rate": _safe_float((shock or {}).get("shock_rate", 0.0)),
                    "params": (shock or {}).get("params", {}) if isinstance(shock, dict) else {},
                },
                "turnover_budget": turnover_budget,
                "concentration": {
                    "enabled": bool((conc or {}).get("enabled", False)) if isinstance(conc, dict) else False,
                    "params": {
                        "top1_cap": _safe_float((conc or {}).get("top1_cap", 0.0)),
                        "top3_cap": _safe_float((conc or {}).get("top3_cap", 0.0)),
                        "max_hhi": _safe_float((conc or {}).get("max_hhi", 0.0)),
                    },
                    "stats": conc_stats,
                },
                "dream_coherence": {
                    "status": str((dream or {}).get("status", "na")) if isinstance(dream, dict) else "na",
                    "signals": int(len((dream or {}).get("signals", []) or [])) if isinstance(dream, dict) else 0,
                    "mean_coherence": _safe_float((dream or {}).get("mean_coherence", 0.0)),
                    "mean_governor": _safe_float((dream or {}).get("mean_governor", 0.0)),
                },
                "meta_mix_reliability": {
                    "mean_governor": float(np.mean(meta_rel)) if meta_rel is not None and len(meta_rel) else None,
                    "mean_confidence_raw": _safe_float((mix or {}).get("mean_confidence_raw", 0.0)),
                    "mean_confidence_calibrated": _safe_float((mix or {}).get("mean_confidence_calibrated", 0.0)),
                    "brier_raw": _safe_float((mix or {}).get("brier_raw", 0.0)),
                    "brier_calibrated": _safe_float((mix or {}).get("brier_calibrated", 0.0)),
                },
                "final_steps": list((final_info or {}).get("steps", []) or []),
                "pipeline_failed_count": int((pipeline or {}).get("failed_count", 0)),
                "runtime_total_scalar": {
                    "latest": float(gov_trace[-1]) if gov_trace is not None and len(gov_trace) else None,
                    "mean": float(np.mean(gov_trace)) if gov_trace is not None and len(gov_trace) else None,
                    "min": float(np.min(gov_trace)) if gov_trace is not None and len(gov_trace) else None,
                    "max": float(np.max(gov_trace)) if gov_trace is not None and len(gov_trace) else None,
                },
                "portfolio_drift_watch": {
                    "status": str((drift_watch.get("drift", {}) if isinstance(drift_watch, dict) else {}).get("status", "na")),
                    "latest_l1": _safe_float((drift_watch.get("drift", {}) if isinstance(drift_watch, dict) else {}).get("latest_l1", 0.0)),
                    "mean_l1": _safe_float((drift_watch.get("drift", {}) if isinstance(drift_watch, dict) else {}).get("mean_l1", 0.0)),
                    "p95_l1": _safe_float((drift_watch.get("drift", {}) if isinstance(drift_watch, dict) else {}).get("p95_l1", 0.0)),
                },
                "heartbeat_stress": {
                    "latest": float(hb_stress[-1]) if hb_stress is not None and len(hb_stress) else None,
                    "mean": float(np.mean(hb_stress)) if hb_stress is not None and len(hb_stress) else None,
                    "max": float(np.max(hb_stress)) if hb_stress is not None and len(hb_stress) else None,
                },
            },
            "trust": float(
                np.clip(
                    0.45 * np.clip(_safe_float(health.get("health_score", 0.0)) / 100.0, 0.0, 1.0)
                    + 0.30 * np.clip(quality_score, 0.0, 1.0)
                    + 0.25 * (1.0 if int((pipeline or {}).get("failed_count", 0)) == 0 else 0.0),
                    0.0,
                    1.0,
                )
            ),
        },
        {
            "event_type": "decision.runtime_context",
            "namespace": "private/c3/decisions",
            "ts_utc": ts,
            "payload": {
                "runtime_context": runtime_ctx if isinstance(runtime_ctx, dict) else {},
                "hive_transparency_summary": (htx or {}).get("summary", {}) if isinstance(htx, dict) else {},
            },
            "trust": float(np.clip(_safe_float((runtime_ctx or {}).get("runtime_multiplier", 1.0), 1.0), 0.0, 1.0)),
        },
        {
            "event_type": "memory.feedback_state",
            "namespace": "private/c3/governance",
            "ts_utc": ts,
            "payload": {
                "novaspine_context": {
                    "status": str((nctx or {}).get("status", "na")) if isinstance(nctx, dict) else "na",
                    "context_resonance": _safe_float((nctx or {}).get("context_resonance", 0.0)),
                    "context_boost": _safe_float((nctx or {}).get("context_boost", 1.0), 1.0),
                },
                "novaspine_hive_feedback": {
                    "status": str((nhive or {}).get("status", "na")) if isinstance(nhive, dict) else "na",
                    "global_boost": _safe_float((nhive or {}).get("global_boost", 1.0), 1.0),
                    "hives": int(len((nhive or {}).get("per_hive", {}) or {})) if isinstance(nhive, dict) else 0,
                },
            },
            "trust": float(
                np.clip(
                    0.5 * np.clip(_safe_float((nctx or {}).get("context_boost", 1.0), 1.0), 0.0, 1.2) / 1.2
                    + 0.5 * np.clip(_safe_float((nhive or {}).get("global_boost", 1.0), 1.0), 0.0, 1.2) / 1.2,
                    0.0,
                    1.0,
                )
            ),
        },
    ]
    if isinstance(immune, dict) and immune:
        events.append(
            {
                "event_type": "governance.immune_drill",
                "namespace": "private/c3/governance",
                "ts_utc": ts,
                "payload": immune,
                "trust": 0.9 if bool(immune.get("pass", False)) else 0.3,
            }
        )
    return events


if __name__ == "__main__":
    enabled = str(os.getenv("C3_MEMORY_ENABLE", "0")).strip().lower() in {"1", "true", "yes", "on"}
    backend = str(os.getenv("C3_MEMORY_BACKEND", "novaspine_api")).strip().lower()
    namespace = str(os.getenv("C3_MEMORY_NAMESPACE", "private/nova/actions")).strip() or "private/nova/actions"
    outbox = Path(str(os.getenv("C3_MEMORY_DIR", str(RUNS / "novaspine_outbox"))))
    http_url = str(os.getenv("C3_MEMORY_HTTP_URL", "")).strip() or None
    http_token = str(os.getenv("C3_MEMORY_TOKEN", "")).strip() or None
    novaspine_url = str(os.getenv("C3_MEMORY_NOVASPINE_URL", "http://127.0.0.1:8420")).strip() or "http://127.0.0.1:8420"
    novaspine_token = str(os.getenv("C3AE_API_TOKEN", "")).strip() or http_token

    events = build_events()
    ns_counts = {}
    type_counts = {}
    for ev in events:
        ns = str(ev.get("namespace", "private/nova/actions"))
        et = str(ev.get("event_type", "event"))
        ns_counts[ns] = int(ns_counts.get(ns, 0) + 1)
        type_counts[et] = int(type_counts.get(et, 0) + 1)
    # Always materialize local batch artifacts for auditability.
    jsonl_path = RUNS / "novaspine_events.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=True) + "\n")
    batch = {
        "timestamp_utc": _now_iso(),
        "enabled": bool(enabled),
        "backend": backend,
        "namespace": namespace,
        "novaspine_url": novaspine_url,
        "events_count": int(len(events)),
        "namespaces": ns_counts,
        "event_types": type_counts,
        "events": events,
    }
    (RUNS / "novaspine_last_batch.json").write_text(json.dumps(batch, indent=2))

    if enabled:
        res = publish_events(
            events=events,
            backend=backend,
            namespace=namespace,
            outbox_dir=outbox,
            http_url=http_url,
            http_token=http_token,
            novaspine_base_url=novaspine_url,
            novaspine_token=novaspine_token,
        )
        sync = {
            "timestamp_utc": _now_iso(),
            "enabled": True,
            "backend": backend,
            "namespace": namespace,
            "novaspine_url": novaspine_url,
            "published": int(res.published),
            "queued": int(res.queued),
            "failed": int(res.failed),
            "outbox_file": res.outbox_file,
            "error": res.error,
            "events_count": int(len(events)),
            "namespaces": ns_counts,
            "event_types": type_counts,
        }
    else:
        sync = {
            "timestamp_utc": _now_iso(),
            "enabled": False,
            "backend": backend,
            "namespace": namespace,
            "novaspine_url": novaspine_url,
            "published": 0,
            "queued": int(len(events)),
            "failed": 0,
            "outbox_file": None,
            "error": "disabled",
            "events_count": int(len(events)),
            "namespaces": ns_counts,
            "event_types": type_counts,
        }

    (RUNS / "novaspine_sync_status.json").write_text(json.dumps(sync, indent=2))

    _append_card(
        "NovaSpine Bridge ✔",
        (
            f"<p>enabled={sync['enabled']}, backend={sync['backend']}, namespace={sync['namespace']}</p>"
            f"<p>events={sync['events_count']}, published={sync['published']}, queued={sync['queued']}, failed={sync['failed']}</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'novaspine_events.jsonl'}")
    print(f"✅ Wrote {RUNS/'novaspine_last_batch.json'}")
    print(f"✅ Wrote {RUNS/'novaspine_sync_status.json'}")
