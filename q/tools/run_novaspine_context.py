#!/usr/bin/env python3
# Pull context from NovaSpine augment API and convert to a risk boost scalar.
#
# Writes:
#   runs_plus/novaspine_context.json
#   runs_plus/novaspine_context_boost.csv

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib import parse, request

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.novaspine_context import (  # noqa: E402
    apply_turnover_dampener,
    build_context_query,
    context_boost,
    context_resonance,
    turnover_pressure,
)

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


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _infer_length() -> int:
    for rel in [
        "portfolio_weights_final.csv",
        "weights_regime.csv",
        "weights_tail_blend.csv",
        "portfolio_weights.csv",
    ]:
        p = RUNS / rel
        if not p.exists():
            continue
        try:
            a = np.loadtxt(p, delimiter=",")
        except Exception:
            try:
                a = np.loadtxt(p, delimiter=",", skiprows=1)
            except Exception:
                continue
        a = np.asarray(a, float)
        if a.ndim == 1:
            return int(len(a))
        if a.ndim == 2:
            return int(a.shape[0])
    return 0


def _json_request(url: str, payload: dict, token: str | None, timeout_sec: float = 6.0):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = request.Request(url, method="POST", data=body, headers=headers)
    with request.urlopen(req, timeout=float(timeout_sec)) as resp:
        code = int(getattr(resp, "status", 200))
        raw = resp.read()
        data = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
        return code, data


def _append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    enabled = str(os.getenv("C3_MEMORY_RECALL_ENABLE", "1")).strip().lower() in {"1", "true", "yes", "on"}
    backend = str(os.getenv("C3_MEMORY_BACKEND", "filesystem")).strip().lower()
    base = str(os.getenv("C3_MEMORY_NOVASPINE_URL", "http://127.0.0.1:8420")).strip() or "http://127.0.0.1:8420"
    token = str(os.getenv("C3AE_API_TOKEN", "")).strip() or None
    top_k = int(max(1, int(float(os.getenv("C3_MEMORY_RECALL_TOPK", "6")))))
    min_score = float(os.getenv("C3_MEMORY_RECALL_MIN_SCORE", "0.005"))
    include_alerts = str(os.getenv("C3_MEMORY_RECALL_INCLUDE_ALERTS", "0")).strip().lower() in {"1", "true", "yes", "on"}
    mean_turnover_limit = float(os.getenv("Q_NOVASPINE_TURNOVER_MEAN_LIMIT", "0.45"))
    max_turnover_limit = float(os.getenv("Q_NOVASPINE_TURNOVER_MAX_LIMIT", "1.00"))
    rolling_turnover_limit = float(os.getenv("Q_NOVASPINE_TURNOVER_ROLLING_LIMIT", "1.25"))
    turnover_max_cut = float(os.getenv("Q_NOVASPINE_TURNOVER_MAX_CUT", "0.12"))

    quality = _load_json(RUNS / "quality_snapshot.json") or {}
    cross = _load_json(RUNS / "cross_hive_summary.json") or {}
    alerts = _load_json(RUNS / "health_alerts.json") or {}

    top_hives = []
    latest = cross.get("latest_weights", {}) if isinstance(cross, dict) else {}
    if isinstance(latest, dict) and latest:
        top_hives = [k for k, _ in sorted(latest.items(), key=lambda kv: float(kv[1]), reverse=True)]
    query = build_context_query(
        top_hives=top_hives,
        quality_score=float(quality.get("quality_score", 0.5)),
        alerts=(list(alerts.get("alerts", []) or []) if include_alerts else []),
    )

    status = "disabled"
    err = None
    recall_count = 0
    scores = []
    context_text = ""
    memories = []

    if enabled and backend in {"novaspine_api", "novaspine", "c3ae"}:
        try:
            if not base.startswith(("http://", "https://")):
                base = "http://" + base
            url = parse.urljoin(base.rstrip("/") + "/", "api/v1/memory/augment")
            payload = {
                "query": query,
                "top_k": top_k,
                "min_score": float(min_score),
                "format": "plain",
                "roles": ["user", "assistant"],
            }
            code, data = _json_request(url, payload=payload, token=token)
            if 200 <= code < 300:
                status = "ok"
                recall_count = int(data.get("count", 0) or 0)
                context_text = str(data.get("context", "") or "")
                memories = list(data.get("memories", []) or [])
                for m in memories:
                    try:
                        scores.append(float(m.get("score", 0.0)))
                    except Exception:
                        continue
            else:
                status = f"http_{code}"
                err = f"augment_status_{code}"
        except Exception as e:
            status = "unreachable"
            err = str(e)

    elif enabled:
        status = "local_only"

    res = context_resonance(recall_count, top_k, scores=scores)
    boost_raw = context_boost(res, status_ok=(status == "ok"))
    cross_mean = _safe_float(cross.get("mean_turnover", 0.0), 0.0) if isinstance(cross, dict) else 0.0
    cross_max = _safe_float(cross.get("max_turnover", cross_mean), cross_mean) if isinstance(cross, dict) else cross_mean
    cross_roll = _safe_float(cross.get("rolling_turnover_max", cross_max), cross_max) if isinstance(cross, dict) else cross_max
    pressure = turnover_pressure(
        cross_mean,
        cross_max,
        cross_roll,
        mean_limit=mean_turnover_limit,
        max_limit=max_turnover_limit,
        rolling_limit=rolling_turnover_limit,
    )
    boost = apply_turnover_dampener(boost_raw, pressure=pressure, max_cut=turnover_max_cut)

    T = _infer_length()
    if T > 0:
        vec = np.full(T, float(boost), dtype=float)
        np.savetxt(RUNS / "novaspine_context_boost.csv", vec, delimiter=",")

    out = {
        "timestamp_utc": _now_iso(),
        "enabled": bool(enabled),
        "backend": backend,
        "status": status,
        "novaspine_url": base,
        "query": query,
        "top_k": int(top_k),
        "min_score": float(min_score),
        "retrieved_count": int(recall_count),
        "avg_memory_score": float(np.mean(scores)) if scores else 0.0,
        "context_resonance": float(res),
        "context_boost_raw": float(boost_raw),
        "context_boost": float(boost),
        "turnover_pressure": float(pressure),
        "turnover_dampener": float(boost_raw - boost),
        "cross_hive_turnover": {
            "mean_turnover": float(cross_mean),
            "max_turnover": float(cross_max),
            "rolling_turnover_max": float(cross_roll),
            "limits": {
                "mean": float(mean_turnover_limit),
                "max": float(max_turnover_limit),
                "rolling": float(rolling_turnover_limit),
            },
            "max_cut": float(turnover_max_cut),
        },
        "length": int(T),
        "context_excerpt": context_text[:1200],
        "memories_sample": memories[:5],
        "error": err,
    }
    (RUNS / "novaspine_context.json").write_text(json.dumps(out, indent=2))

    _append_card(
        "NovaSpine Recall Loop ✔",
        (
            f"<p>status={status}, retrieved={recall_count}, resonance={res:.3f}, boost={boost:.3f}, pressure={pressure:.3f}</p>"
            f"<p>query: {query[:180]}</p>"
        ),
    )
    if T > 0:
        print(f"✅ Wrote {RUNS/'novaspine_context_boost.csv'}")
    print(f"✅ Wrote {RUNS/'novaspine_context.json'}")
