#!/usr/bin/env python3
"""
Strict OOS validation on costed returns.

Reads:
  - runs_plus/daily_returns.csv (net)
  - runs_plus/daily_returns_gross.csv (optional)
  - runs_plus/daily_costs.csv (optional)

Writes:
  - runs_plus/wf_oos_returns.csv
  - runs_plus/strict_oos_validation.json
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _load_series(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        x = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            x = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    x = np.asarray(x, float).ravel()
    if x.size == 0:
        return None
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _metrics(r: np.ndarray) -> dict:
    v = np.asarray(r, float).ravel()
    if v.size == 0:
        return {
            "n": 0,
            "mean_daily": 0.0,
            "vol_daily": 0.0,
            "sharpe": 0.0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
            "ann_return_arith": 0.0,
            "ann_vol": 0.0,
        }
    mu = float(np.mean(v))
    if v.size > 1:
        sd = float(np.std(v, ddof=1))
    else:
        sd = 0.0
    sd = float(sd + 1e-12)
    sh = float((mu / sd) * math.sqrt(252.0))
    hit = float(np.mean(v > 0.0))
    eq = np.cumsum(v)
    peak = np.maximum.accumulate(eq)
    mdd = float(np.min(eq - peak))
    return {
        "n": int(v.size),
        "mean_daily": mu,
        "vol_daily": sd,
        "sharpe": sh,
        "hit_rate": hit,
        "max_drawdown": mdd,
        "ann_return_arith": float(mu * 252.0),
        "ann_vol": float(sd * math.sqrt(252.0)),
    }


def _append_card(title: str, html: str) -> None:
    if str(os.getenv("Q_DISABLE_REPORT_CARDS", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        return
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


def _build_split_index(T: int, train_frac: float, min_train: int, min_test: int) -> int:
    split = max(int(min_train), int(T * train_frac))
    if (T - split) < int(min_test):
        split = max(int(min_train), T - int(min_test))
    split = int(np.clip(split, 1, max(1, T - 1)))
    return split


def _robust_splits(T: int, min_train: int, min_test: int, n_splits: int) -> list[int]:
    lo = int(np.clip(int(min_train), 1, max(1, T - 1)))
    hi = int(np.clip(T - int(min_test), 1, max(1, T - 1)))
    if hi < lo:
        return [int(np.clip(hi, 1, max(1, T - 1)))]
    ns = int(np.clip(int(n_splits), 1, 16))
    idx = np.linspace(lo, hi, ns, dtype=int)
    out = sorted(set(int(np.clip(i, 1, max(1, T - 1))) for i in idx.tolist()))
    return out if out else [int(np.clip(hi, 1, max(1, T - 1)))]


def _latest_holdout_window(T: int, requested: int, min_window: int) -> int:
    if T <= 1:
        return 0
    lo = int(np.clip(int(min_window), 10, max(10, T - 1)))
    req = int(np.clip(int(requested), lo, max(lo, T - 1)))
    return int(np.clip(req, lo, max(lo, T - 1)))


def _aggregate_robust(metrics: list[dict]) -> dict:
    if not metrics:
        return _metrics(np.asarray([], float))
    sh = np.asarray([float(m.get("sharpe", 0.0)) for m in metrics], float)
    hit = np.asarray([float(m.get("hit_rate", 0.0)) for m in metrics], float)
    dd_abs = np.asarray([abs(float(m.get("max_drawdown", 0.0))) for m in metrics], float)
    ns = np.asarray([int(m.get("n", 0)) for m in metrics], int)
    return {
        "n": int(np.min(ns)) if ns.size else 0,
        "sharpe": float(np.median(sh)),
        "hit_rate": float(np.median(hit)),
        # Conservative drawdown: 75th percentile of drawdown severity.
        "max_drawdown": float(-np.quantile(dd_abs, 0.75)),
        "sharpe_p25": float(np.quantile(sh, 0.25)),
        "hit_rate_p25": float(np.quantile(hit, 0.25)),
        "max_drawdown_worst": float(-np.max(dd_abs)),
        "num_splits": int(len(metrics)),
    }


def main() -> int:
    r = _load_series(RUNS / "daily_returns.csv")
    if r is None:
        print("(!) Missing runs_plus/daily_returns.csv; run make_daily_from_weights.py first.")
        return 0

    T = len(r)
    train_frac = float(np.clip(float(os.getenv("Q_STRICT_OOS_TRAIN_FRAC", "0.75")), 0.50, 0.95))
    min_train = int(np.clip(int(float(os.getenv("Q_STRICT_OOS_MIN_TRAIN", "756"))), 100, 100000))
    min_test = int(np.clip(int(float(os.getenv("Q_STRICT_OOS_MIN_TEST", "252"))), 50, 100000))

    split = _build_split_index(T, train_frac, min_train, min_test)

    train = r[:split]
    oos = r[split:]
    np.savetxt(RUNS / "wf_oos_returns.csv", oos, delimiter=",")

    robust_n_splits = int(np.clip(int(float(os.getenv("Q_STRICT_OOS_ROBUST_SPLITS", "5"))), 1, 16))
    split_ix = _robust_splits(T, min_train=min_train, min_test=min_test, n_splits=robust_n_splits)
    split_rows = []
    split_metrics = []
    for s in split_ix:
        seg = r[s:]
        m = _metrics(seg)
        row = {
            "split_index": int(s),
            "train_rows": int(s),
            "oos_rows": int(len(seg)),
            "metrics_oos_net": m,
        }
        split_rows.append(row)
        split_metrics.append(m)
    robust = _aggregate_robust(split_metrics)
    (RUNS / "strict_oos_splits.json").write_text(
        json.dumps(
            {
                "rows_total": int(T),
                "min_train": int(min_train),
                "min_test": int(min_test),
                "num_splits_requested": int(robust_n_splits),
                "splits": split_rows,
                "metrics_oos_robust": robust,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    latest_holdout_req = int(np.clip(int(float(os.getenv("Q_STRICT_OOS_LATEST_HOLDOUT_DAYS", "252"))), 20, 20000))
    latest_holdout_min = int(np.clip(int(float(os.getenv("Q_STRICT_OOS_LATEST_HOLDOUT_MIN", "126"))), 20, 20000))
    latest_n = _latest_holdout_window(T, requested=latest_holdout_req, min_window=latest_holdout_min)
    latest = r[-latest_n:] if latest_n > 0 else np.asarray([], float)
    latest_metrics = _metrics(latest)

    gross = _load_series(RUNS / "daily_returns_gross.csv")
    costs = _load_series(RUNS / "daily_costs.csv")
    cost_info = {}
    if gross is not None and len(gross) >= T:
        gv = gross[:T]
        cost_info["gross_full"] = _metrics(gv)
    if costs is not None and len(costs) >= T:
        cv = costs[:T]
        cost_info["cost"] = {
            "mean_daily": float(np.mean(cv)),
            "ann_cost_estimate": float(np.mean(cv) * 252.0),
            "sum": float(np.sum(cv)),
        }

    out = {
        "method": "strict_holdout_costed",
        "source": "runs_plus/daily_returns.csv",
        "rows_total": int(T),
        "split_index": int(split),
        "train_frac_requested": float(train_frac),
        "train_rows": int(len(train)),
        "oos_rows": int(len(oos)),
        "metrics_full_net": _metrics(r),
        "metrics_train_net": _metrics(train),
        "metrics_oos_net": _metrics(oos),
        "metrics_oos_robust": robust,
        "metrics_oos_latest": latest_metrics,
        "latest_holdout_days_requested": int(latest_holdout_req),
        "latest_holdout_days_used": int(latest_n),
        "robust_oos_splits_file": str(RUNS / "strict_oos_splits.json"),
        "cost_context": cost_info,
    }
    (RUNS / "strict_oos_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    m = out["metrics_oos_net"]
    mr = out["metrics_oos_robust"]
    ml = out["metrics_oos_latest"]
    html = (
        f"<p>Strict OOS net validation: rows={out['oos_rows']}, "
        f"Sharpe={m['sharpe']:.3f}, Hit={m['hit_rate']:.3f}, MaxDD={m['max_drawdown']:.3f}.</p>"
        f"<p>Split: train={out['train_rows']} / oos={out['oos_rows']}.</p>"
        f"<p>Robust OOS ({mr['num_splits']} splits): Sharpe={mr['sharpe']:.3f}, "
        f"Hit={mr['hit_rate']:.3f}, MaxDD={mr['max_drawdown']:.3f}.</p>"
        f"<p>Latest holdout ({latest_n} rows): Sharpe={ml['sharpe']:.3f}, "
        f"Hit={ml['hit_rate']:.3f}, MaxDD={ml['max_drawdown']:.3f}.</p>"
    )
    _append_card("Strict OOS Validation ✔", html)

    print(f"✅ Wrote {RUNS/'wf_oos_returns.csv'}")
    print(f"✅ Wrote {RUNS/'strict_oos_splits.json'}")
    print(f"✅ Wrote {RUNS/'strict_oos_validation.json'}")
    print(
        f"OOS net: Sharpe={m['sharpe']:.3f} Hit={m['hit_rate']:.3f} MaxDD={m['max_drawdown']:.3f} N={m['n']}"
    )
    print(
        f"OOS robust: Sharpe={mr['sharpe']:.3f} Hit={mr['hit_rate']:.3f} "
        f"MaxDD={mr['max_drawdown']:.3f} Splits={mr['num_splits']}"
    )
    print(
        f"OOS latest: Sharpe={ml['sharpe']:.3f} Hit={ml['hit_rate']:.3f} "
        f"MaxDD={ml['max_drawdown']:.3f} N={ml['n']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
