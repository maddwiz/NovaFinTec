#!/usr/bin/env python3
"""
Publish an investor-friendly results snapshot under repository-level `results/`.

Outputs:
  - results/walkforward_metrics.json
  - results/walkforward_equity.csv
  - results/benchmarks_metrics.csv
  - results/governor_compound_summary.json
  - results/README.md
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
RUNS = ROOT / "runs_plus"
DATA = ROOT / "data"
RESULTS = REPO_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def _load_vec(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            arr = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    arr = np.asarray(arr, float).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr if arr.size > 0 else None


def _metrics(r: np.ndarray) -> dict[str, float | int]:
    v = np.asarray(r, float).reshape(-1)
    if v.size == 0:
        return {
            "n": 0,
            "sharpe": 0.0,
            "hit_rate": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "max_drawdown": 0.0,
        }
    mu = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if v.size > 1 else float(np.std(v))
    sharpe = float((mu / (sd + 1e-12)) * np.sqrt(252.0))
    hit = float(np.mean(v > 0.0))
    ann_return = float(mu * 252.0)
    ann_vol = float(sd * np.sqrt(252.0))
    eq = np.cumprod(1.0 + v)
    peak = np.maximum.accumulate(eq)
    dd = (eq / (peak + 1e-12)) - 1.0
    max_dd = float(np.min(dd))
    return {
        "n": int(v.size),
        "sharpe": sharpe,
        "hit_rate": hit,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_drawdown": max_dd,
    }


def _close_series(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    date_col = None
    for c in df.columns:
        if str(c).strip().lower() in {"date", "time", "timestamp"} or "date" in str(c).lower():
            date_col = c
            break
    close_col = None
    for cand in ["Adj Close", "Close", "adj_close", "close", "PX_LAST", "value", "VALUE"]:
        for c in df.columns:
            if str(c).strip().lower() == cand.lower():
                close_col = c
                break
        if close_col:
            break
    if date_col is None or close_col is None:
        return None
    s = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce"), "close": pd.to_numeric(df[close_col], errors="coerce")})
    s = s.dropna().sort_values("date")
    if s.empty:
        return None
    return pd.Series(s["close"].values, index=s["date"].values)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _write_equity_csv(path: Path, returns: np.ndarray) -> None:
    eq = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(eq)
    dd = (eq / (peak + 1e-12)) - 1.0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "equity", "drawdown"])
        for i, (e, d) in enumerate(zip(eq, dd)):
            w.writerow([int(i), float(e), float(d)])


def _compute_equity_at_cost(returns: np.ndarray, cost_bps: float) -> np.ndarray:
    r = np.asarray(returns, float).ravel()
    drag = float(cost_bps) / 10000.0
    net = r - drag
    return np.cumprod(1.0 + net)


def _drawdown(equity: np.ndarray) -> np.ndarray:
    eq = np.asarray(equity, float).ravel()
    if eq.size == 0:
        return np.zeros(0, dtype=float)
    peak = np.maximum.accumulate(eq)
    return (eq / (peak + 1e-12)) - 1.0


def _write_detailed_equity_chart(path: Path, returns: np.ndarray) -> bool:
    if plt is None:
        return False
    r = np.asarray(returns, float).ravel()
    if r.size <= 0:
        return False
    dates = np.arange(len(r))
    curves = {}
    for bps, label in [(0, "Zero cost"), (5, "5 bps"), (10, "10 bps"), (20, "20 bps")]:
        curves[label] = _compute_equity_at_cost(r, bps)

    base_eq = curves["Zero cost"]
    dd = _drawdown(base_eq)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1], sharex=True)
    for label, eq in curves.items():
        alpha = 1.0 if label == "Zero cost" else 0.85
        lw = 2.0 if label == "Zero cost" else 1.4
        ax1.plot(dates, eq, label=label, alpha=alpha, linewidth=lw)
    ax1.set_title("Walk-Forward OOS Equity Curve (Cost Sensitivity)")
    ax1.legend(loc="best")
    ax1.grid(alpha=0.20)

    ax2.fill_between(dates, dd, 0.0, color="red", alpha=0.25)
    ax2.plot(dates, dd, color="red", linewidth=1.0)
    ax2.set_title("Drawdown")
    ax2.grid(alpha=0.20)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _summarize_governor_compound() -> dict:
    p = RUNS / "final_governor_trace.csv"
    if not p.exists():
        return {"ok": False, "reason": "missing_final_governor_trace"}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {"ok": False, "reason": "unreadable_final_governor_trace"}
    if df.empty:
        return {"ok": False, "reason": "empty_final_governor_trace"}
    cols = [c for c in df.columns if str(c).strip().lower() != "runtime_total_scalar"]
    if not cols:
        return {"ok": False, "reason": "no_governor_columns"}
    mat = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0).values.astype(float)
    compound = np.clip(np.prod(mat, axis=1), 0.0, 5.0)
    return {
        "ok": True,
        "rows": int(len(compound)),
        "mean": float(np.mean(compound)),
        "median": float(np.median(compound)),
        "p10": float(np.percentile(compound, 10)),
        "min": float(np.min(compound)),
        "max": float(np.max(compound)),
        "below_0_20_share": float(np.mean(compound < 0.20)),
        "below_0_10_share": float(np.mean(compound < 0.10)),
    }


def main() -> int:
    primary = _load_vec(RUNS / "wf_oos_returns.csv")
    source = "runs_plus/wf_oos_returns.csv"
    if primary is None:
        primary = _load_vec(RUNS / "daily_returns.csv")
        source = "runs_plus/daily_returns.csv"
    if primary is None:
        print("(!) Missing wf_oos_returns.csv and daily_returns.csv. Nothing to publish.")
        return 0

    primary_metrics = _metrics(primary)
    _write_equity_csv(RESULTS / "walkforward_equity.csv", primary)
    chart_ok = _write_detailed_equity_chart(RESULTS / "equity_curve_detailed.png", primary)

    strict = _read_json(RUNS / "strict_oos_validation.json")
    stress = _read_json(RUNS / "cost_stress_validation.json")
    external_holdout = _read_json(RUNS / "external_holdout_validation.json")
    final_info = _read_json(RUNS / "final_portfolio_info.json")
    governor_compound = _summarize_governor_compound()
    (RESULTS / "governor_compound_summary.json").write_text(json.dumps(governor_compound, indent=2), encoding="utf-8")

    bench_rows = []
    horizon = int(primary_metrics.get("n", 0))
    for sym in ["SPY", "QQQ"]:
        s = _close_series(DATA / f"{sym}.csv")
        if s is None or len(s) < 3:
            continue
        r = s.pct_change().dropna().values.astype(float)
        if horizon > 0 and len(r) > horizon:
            r = r[-horizon:]
        m = _metrics(r)
        m["symbol"] = sym
        bench_rows.append(m)

    with (RESULTS / "benchmarks_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["symbol", "n", "sharpe", "hit_rate", "ann_return", "ann_vol", "max_drawdown"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in bench_rows:
            w.writerow({k: row.get(k, "") for k in cols})

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_returns": source,
        "primary_metrics": primary_metrics,
        "equity_curve_detailed_png": bool(chart_ok),
        "strict_oos": strict,
        "cost_stress": stress,
        "external_holdout": external_holdout,
        "final_portfolio_info": final_info,
        "governor_compound": governor_compound,
        "benchmarks": bench_rows,
    }
    (RESULTS / "walkforward_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Results Snapshot",
        "",
        f"- Generated (UTC): `{payload['generated_at_utc']}`",
        f"- Source returns: `{source}`",
        "",
        "## Primary Metrics",
        "",
        f"- N: `{primary_metrics['n']}`",
        f"- Sharpe: `{primary_metrics['sharpe']:.3f}`",
        f"- Hit rate: `{primary_metrics['hit_rate']:.3f}`",
        f"- Annualized return: `{primary_metrics['ann_return']:.3f}`",
        f"- Annualized vol: `{primary_metrics['ann_vol']:.3f}`",
        f"- Max drawdown: `{primary_metrics['max_drawdown']:.3f}`",
        "",
        "## Governor Compound Diagnostics",
        "",
        f"- Mean compound scalar: `{governor_compound.get('mean', 0.0):.3f}`",
        f"- Min compound scalar: `{governor_compound.get('min', 0.0):.3f}`",
        f"- Share below 0.20: `{governor_compound.get('below_0_20_share', 0.0):.3f}`",
        "",
        "## External Holdout",
        "",
        f"- Present: `{bool(external_holdout)}`",
        f"- OK: `{bool(external_holdout.get('ok', False)) if isinstance(external_holdout, dict) else False}`",
        (
            f"- Sharpe: `{float((external_holdout.get('metrics_external_holdout_net', {}) or {}).get('sharpe', 0.0)):.3f}`"
            if isinstance(external_holdout, dict)
            else "- Sharpe: `0.000`"
        ),
        "",
        "## Files",
        "",
        "- `results/walkforward_metrics.json`",
        "- `results/walkforward_equity.csv`",
        "- `results/equity_curve_detailed.png`",
        "- `results/benchmarks_metrics.csv`",
        "- `results/governor_compound_summary.json`",
    ]
    (RESULTS / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"✅ Wrote {RESULTS/'walkforward_metrics.json'}")
    print(f"✅ Wrote {RESULTS/'walkforward_equity.csv'}")
    if chart_ok:
        print(f"✅ Wrote {RESULTS/'equity_curve_detailed.png'}")
    print(f"✅ Wrote {RESULTS/'benchmarks_metrics.csv'}")
    print(f"✅ Wrote {RESULTS/'governor_compound_summary.json'}")
    print(f"✅ Wrote {RESULTS/'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
