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


def build_events():
    ts = _now_iso()
    overlay = _load_json(RUNS / "q_signal_overlay.json") or {}
    health = _load_json(RUNS / "system_health.json") or {}
    alerts = _load_json(RUNS / "health_alerts.json") or {}
    quality = _load_json(RUNS / "quality_snapshot.json") or {}
    cross = _load_json(RUNS / "cross_hive_summary.json") or {}
    eco = _load_json(RUNS / "hive_evolution.json") or {}
    constraints = _load_json(RUNS / "execution_constraints_info.json") or {}
    immune = _load_json(RUNS / "immune_drill.json") or {}

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
                "quality_score": float(quality.get("quality_score", 0.5)),
                "quality_governor_mean": float(quality.get("quality_governor_mean", 1.0)),
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
    for ev in events:
        ns = str(ev.get("namespace", "private/nova/actions"))
        ns_counts[ns] = int(ns_counts.get(ns, 0) + 1)
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
