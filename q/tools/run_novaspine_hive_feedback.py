#!/usr/bin/env python3
# Query NovaSpine per hive and build bounded multipliers for cross-hive arbitration.
#
# Writes:
#   runs_plus/novaspine_hive_feedback.json
#   runs_plus/novaspine_hive_boost.csv   (global scalar path for final portfolio scaling)

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib import parse, request

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.novaspine_hive import build_hive_query, hive_boost, hive_resonance  # noqa: E402
from qmods.novaspine_context import apply_turnover_dampener, turnover_pressure  # noqa: E402

RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _infer_length() -> int:
    p = RUNS / "hive_signals.csv"
    if p.exists():
        try:
            h = pd.read_csv(p)
            if "DATE" in h.columns:
                return int(len(pd.to_datetime(h["DATE"], errors="coerce").dropna().unique()))
        except Exception:
            pass
    for rel in ["portfolio_weights_final.csv", "weights_tail_blend.csv", "portfolio_weights.csv"]:
        path = RUNS / rel
        if not path.exists():
            continue
        try:
            a = np.loadtxt(path, delimiter=",")
        except Exception:
            try:
                a = np.loadtxt(path, delimiter=",", skiprows=1)
            except Exception:
                continue
        a = np.asarray(a, float)
        return int(a.shape[0] if a.ndim == 2 else len(a))
    return 0


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
    top_k = int(max(1, int(float(os.getenv("C3_MEMORY_HIVE_RECALL_TOPK", "4")))))
    min_score = float(os.getenv("C3_MEMORY_HIVE_RECALL_MIN_SCORE", "0.005"))
    mean_turnover_limit = float(os.getenv("Q_NOVASPINE_TURNOVER_MEAN_LIMIT", "0.45"))
    max_turnover_limit = float(os.getenv("Q_NOVASPINE_TURNOVER_MAX_LIMIT", "1.00"))
    rolling_turnover_limit = float(os.getenv("Q_NOVASPINE_TURNOVER_ROLLING_LIMIT", "1.25"))
    turnover_max_cut = float(os.getenv("Q_NOVASPINE_TURNOVER_MAX_CUT", "0.12"))
    per_hive_max_cut = float(os.getenv("Q_NOVASPINE_HIVE_PER_HIVE_MAX_CUT", str(max(0.0, turnover_max_cut * 0.75))))
    cross = {}
    p_cross = RUNS / "cross_hive_summary.json"
    if p_cross.exists():
        try:
            cross = json.loads(p_cross.read_text(encoding="utf-8"))
        except Exception:
            cross = {}

    m = RUNS / "hive_wf_metrics.csv"
    per_hive = {}
    status = "disabled"
    err = None
    if m.exists():
        try:
            met = pd.read_csv(m)
        except Exception:
            met = pd.DataFrame()
    else:
        met = pd.DataFrame()

    if enabled and backend in {"novaspine_api", "novaspine", "c3ae"} and len(met):
        try:
            if not base.startswith(("http://", "https://")):
                base = "http://" + base
            url = parse.urljoin(base.rstrip("/") + "/", "api/v1/memory/augment")
            for _, row in met.iterrows():
                hive = str(row.get("HIVE", "")).strip().upper()
                if not hive:
                    continue
                q = build_hive_query(
                    hive=hive,
                    sharpe=float(row.get("sharpe_oos", 0.0)),
                    hit=float(row.get("hit_rate", 0.5)),
                    max_dd=float(row.get("max_dd", 0.0)),
                )
                payload = {
                    "query": q,
                    "top_k": top_k,
                    "min_score": float(min_score),
                    "format": "plain",
                    "roles": ["user", "assistant"],
                }
                code, data = _json_request(url, payload=payload, token=token)
                if not (200 <= code < 300):
                    per_hive[hive] = {
                        "status": f"http_{code}",
                        "retrieved_count": 0,
                        "avg_score": 0.0,
                        "resonance": 0.0,
                        "boost": 1.0,
                    }
                    continue
                mems = list(data.get("memories", []) or [])
                scores = []
                for mm in mems:
                    try:
                        scores.append(float(mm.get("score", 0.0)))
                    except Exception:
                        continue
                cnt = int(data.get("count", len(mems)) or 0)
                res = hive_resonance(cnt, top_k, scores)
                bst_raw = hive_boost(res, status_ok=True)
                per_hive[hive] = {
                    "status": "ok",
                    "retrieved_count": cnt,
                    "avg_score": float(np.mean(scores)) if scores else 0.0,
                    "resonance": float(res),
                    "boost_raw": float(bst_raw),
                    "boost": float(bst_raw),
                }
            status = "ok"
        except Exception as e:
            status = "unreachable"
            err = str(e)
    elif enabled:
        status = "local_only"
    else:
        status = "disabled"

    # Fill absent hives with neutral multipliers.
    if len(met):
        for h in met["HIVE"].astype(str).str.upper().tolist():
            per_hive.setdefault(
                h,
                {
                    "status": status,
                    "retrieved_count": 0,
                    "avg_score": 0.0,
                    "resonance": 0.0,
                    "boost_raw": 1.0,
                    "boost": 1.0,
                },
            )

    cross_mean = _safe_float((cross or {}).get("mean_turnover", 0.0), 0.0)
    cross_max = _safe_float((cross or {}).get("max_turnover", cross_mean), cross_mean)
    cross_roll = _safe_float((cross or {}).get("rolling_turnover_max", cross_max), cross_max)
    pressure = turnover_pressure(
        cross_mean,
        cross_max,
        cross_roll,
        mean_limit=mean_turnover_limit,
        max_limit=max_turnover_limit,
        rolling_limit=rolling_turnover_limit,
    )

    for hive_payload in per_hive.values():
        hive_payload["boost"] = apply_turnover_dampener(
            hive_payload.get("boost_raw", hive_payload.get("boost", 1.0)),
            pressure=pressure,
            max_cut=per_hive_max_cut,
        )

    boosts_raw = [float(v.get("boost_raw", 1.0)) for v in per_hive.values()] if per_hive else [1.0]
    boosts = [float(v.get("boost", 1.0)) for v in per_hive.values()] if per_hive else [1.0]
    global_boost_raw = float(np.clip(np.mean(boosts_raw), 0.90, 1.10))
    global_boost = float(np.clip(np.mean(boosts), 0.90, 1.10))

    T = _infer_length()
    if T > 0:
        np.savetxt(RUNS / "novaspine_hive_boost.csv", np.full(T, global_boost, float), delimiter=",")

    out = {
        "timestamp_utc": _now_iso(),
        "enabled": bool(enabled),
        "backend": backend,
        "status": status,
        "novaspine_url": base,
        "top_k": int(top_k),
        "min_score": float(min_score),
        "global_boost_raw": float(global_boost_raw),
        "global_boost": float(global_boost),
        "turnover_pressure": float(pressure),
        "turnover_dampener": float(global_boost_raw - global_boost),
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
            "per_hive_max_cut": float(per_hive_max_cut),
        },
        "per_hive": per_hive,
        "error": err,
    }
    (RUNS / "novaspine_hive_feedback.json").write_text(json.dumps(out, indent=2))

    _append_card(
        "NovaSpine Hive Feedback ✔",
        f"<p>status={status}, hives={len(per_hive)}, global_boost={global_boost:.3f}, pressure={pressure:.3f}</p>",
    )
    if T > 0:
        print(f"✅ Wrote {RUNS/'novaspine_hive_boost.csv'}")
    print(f"✅ Wrote {RUNS/'novaspine_hive_feedback.json'}")
