#!/usr/bin/env python3
# Build per-hive transparency artifact and append report card.
#
# Writes:
#   runs_plus/hive_transparency.json

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.hive_transparency import build_hive_snapshot  # noqa: E402

RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _append_card(title: str, html: str):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;">\n<h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


def _load_metrics() -> tuple[list[str], dict[str, dict]]:
    p = RUNS / "hive_wf_metrics.csv"
    if not p.exists():
        return [], {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return [], {}
    if "HIVE" not in df.columns:
        return [], {}
    names = []
    by = {}
    for _, row in df.iterrows():
        h = str(row.get("HIVE", "")).strip().upper()
        if not h:
            continue
        if h not in names:
            names.append(h)
        by[h] = {
            "sharpe_oos": row.get("sharpe_oos", 0.0),
            "hit_rate": row.get("hit_rate", 0.0),
            "max_dd": row.get("max_dd", 0.0),
        }
    return names, by


def _load_latest_weights() -> dict[str, float]:
    p = RUNS / "cross_hive_weights.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}
    if df.empty:
        return {}
    row = df.iloc[-1].to_dict()
    out = {}
    for k, v in row.items():
        if str(k).upper() == "DATE":
            continue
        try:
            out[str(k).strip().upper()] = float(v)
        except Exception:
            continue
    return out


def _load_feedback() -> dict[str, dict]:
    p = RUNS / "novaspine_hive_feedback.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return {}
    ph = obj.get("per_hive", {}) if isinstance(obj, dict) else {}
    if not isinstance(ph, dict):
        return {}
    out = {}
    for k, v in ph.items():
        out[str(k).strip().upper()] = v if isinstance(v, dict) else {}
    return out


def _table_html(rows: list[dict]) -> str:
    if not rows:
        return "<p>No hive data available yet.</p>"
    head = (
        "<tr><th>Hive</th><th>Weight</th><th>Sharpe OOS</th><th>Hit</th>"
        "<th>MaxDD</th><th>NovaSpine Boost</th><th>NovaSpine Status</th></tr>"
    )
    body = []
    for r in rows:
        body.append(
            "<tr>"
            f"<td>{r['hive']}</td>"
            f"<td>{float(r['weight']):.3f}</td>"
            f"<td>{float(r['sharpe_oos']):.3f}</td>"
            f"<td>{float(r['hit_rate']):.3f}</td>"
            f"<td>{float(r['max_dd']):.3f}</td>"
            f"<td>{float(r['novaspine_boost']):.3f}</td>"
            f"<td>{r['novaspine_status']}</td>"
            "</tr>"
        )
    return '<table border="1" cellspacing="0" cellpadding="4">' + head + "".join(body) + "</table>"


if __name__ == "__main__":
    names, metrics_by = _load_metrics()
    latest_weights = _load_latest_weights()
    feedback = _load_feedback()

    snap = build_hive_snapshot(
        hive_names=names,
        metrics_by_hive=metrics_by,
        latest_weights=latest_weights,
        feedback_by_hive=feedback,
    )

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **snap,
    }
    out_path = RUNS / "hive_transparency.json"
    out_path.write_text(json.dumps(out, indent=2))

    s = out["summary"]
    html = (
        f"<p>hives={s['hive_count']}, top={s['top_hive']}, top_weight={float(s['top_weight']):.3f}, "
        f"mean_sharpe={float(s['mean_sharpe_oos']):.3f}, mean_novaspine_boost={float(s['mean_novaspine_boost']):.3f}</p>"
        + _table_html(out["rows"])
    )
    _append_card("Hive Transparency ✔", html)
    print(f"✅ Wrote {out_path}")
