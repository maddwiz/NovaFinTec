#!/usr/bin/env python3
# System health snapshot for Q pipeline.
#
# Writes:
#   runs_plus/system_health.json

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _now():
    return datetime.now(timezone.utc)


def _hours_since(path: Path):
    if not path.exists():
        return None
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return float((_now() - ts).total_seconds() / 3600.0)


def _load_series(path: Path):
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
    if a.ndim == 2 and a.shape[1] >= 1:
        a = a[:, -1]
    return a.ravel()


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


if __name__ == "__main__":
    required = [
        RUNS / "portfolio_weights_final.csv",
        RUNS / "meta_mix.csv",
        RUNS / "global_governor.csv",
        RUNS / "heartbeat_exposure_scaler.csv",
        RUNS / "legacy_exposure.csv",
        RUNS / "cross_hive_weights.csv",
        RUNS / "weights_cross_hive_governed.csv",
    ]
    optional = [
        RUNS / "hive_wf_metrics.csv",
        RUNS / "hive_diversification_governor.csv",
        RUNS / "reflex_signal_gated.csv",
        RUNS / "synapses_summary.json",
        RUNS / "meta_stack_summary.json",
        RUNS / "tune_best_config.json",
        RUNS / "portfolio_weights_exec.csv",
        RUNS / "execution_constraints_info.json",
        RUNS / "q_signal_overlay.json",
        RUNS / "q_signal_overlay.csv",
    ]

    checks = []
    for p in required:
        checks.append(
            {
                "file": p.name,
                "required": True,
                "exists": p.exists(),
                "hours_since_update": _hours_since(p),
            }
        )
    for p in optional:
        checks.append(
            {
                "file": p.name,
                "required": False,
                "exists": p.exists(),
                "hours_since_update": _hours_since(p),
            }
        )

    w = _load_matrix(RUNS / "portfolio_weights_final.csv")
    daily = _load_series(RUNS / "daily_returns.csv")
    gov = _load_series(RUNS / "global_governor.csv")
    hive_gov = _load_series(RUNS / "hive_diversification_governor.csv")

    shape = {}
    if w is not None:
        shape["weights_rows"] = int(w.shape[0])
        shape["weights_cols"] = int(w.shape[1])
        shape["weights_abs_mean"] = float(np.mean(np.abs(w)))
    if daily is not None:
        shape["daily_returns_rows"] = int(len(daily))
    if gov is not None:
        shape["global_governor_rows"] = int(len(gov))
        shape["global_governor_mean"] = float(np.mean(gov))
    if hive_gov is not None:
        shape["hive_governor_rows"] = int(len(hive_gov))
        shape["hive_governor_mean"] = float(np.mean(hive_gov))

    # Alignment diagnostics
    issues = []
    if w is None:
        issues.append("missing portfolio_weights_final.csv")
    if daily is not None and w is not None and len(daily) < int(0.5 * w.shape[0]):
        issues.append("daily_returns much shorter than final weights")
    if gov is not None and w is not None and len(gov) < int(0.5 * w.shape[0]):
        issues.append("global_governor much shorter than final weights")
    if w is not None:
        bad = np.isnan(w).any() or np.isinf(w).any()
        if bad:
            issues.append("portfolio_weights_final contains NaN/Inf")

    required_ok = sum(1 for c in checks if c["required"] and c["exists"])
    required_total = sum(1 for c in checks if c["required"])
    optional_ok = sum(1 for c in checks if (not c["required"]) and c["exists"])
    optional_total = sum(1 for c in checks if not c["required"])
    health_score = 100.0 * (0.75 * (required_ok / max(1, required_total)) + 0.25 * (optional_ok / max(1, optional_total)))
    if issues:
        health_score = max(0.0, health_score - 10.0 * len(issues))

    out = {
        "timestamp_utc": _now().isoformat(),
        "health_score": float(health_score),
        "required_ok": int(required_ok),
        "required_total": int(required_total),
        "optional_ok": int(optional_ok),
        "optional_total": int(optional_total),
        "checks": checks,
        "shape": shape,
        "issues": issues,
    }
    (RUNS / "system_health.json").write_text(json.dumps(out, indent=2))

    html = (
        f"<p>Health score: <b>{health_score:.1f}</b> "
        f"(required {required_ok}/{required_total}, optional {optional_ok}/{optional_total})</p>"
        f"<p>Issues: {', '.join(issues) if issues else 'none'}</p>"
    )
    _append_card("System Health ✔", html)
    print(f"✅ Wrote {RUNS/'system_health.json'}")
