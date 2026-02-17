#!/usr/bin/env python3
"""
Capacity/market-impact proxy guard.

Estimates participation pressure from position size vs ADV proxies and emits a
bounded exposure scalar for optional use in final portfolio construction.

Writes:
  - runs_plus/capacity_impact_scalar.csv
  - runs_plus/capacity_impact_proxy.csv
  - runs_plus/capacity_impact_info.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
DATA = ROOT / "data"
RUNS.mkdir(parents=True, exist_ok=True)


def _load_mat(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            arr = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    arr = np.asarray(arr, float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _first_weights() -> np.ndarray | None:
    cands = [
        RUNS / "weights_tail_blend.csv",
        RUNS / "weights_regime.csv",
        RUNS / "portfolio_weights.csv",
        ROOT / "portfolio_weights.csv",
    ]
    for p in cands:
        w = _load_mat(p)
        if w is not None:
            return w
    return None


def _load_asset_names(n: int) -> list[str]:
    p = RUNS / "asset_names.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            c = df.columns[0]
            out = [str(x).strip().upper() for x in df[c].tolist()]
            out = [x if x else f"A{i:03d}" for i, x in enumerate(out)]
            if len(out) >= n:
                return out[:n]
        except Exception:
            pass
    return [f"A{i:03d}" for i in range(n)]


def _read_adv_series(sym: str) -> pd.Series:
    p = DATA / f"{sym}.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.Series(dtype=float)
    if df.empty:
        return pd.Series(dtype=float)

    dcol = next((c for c in ["DATE", "Date", "date", "timestamp", "Timestamp"] if c in df.columns), None)
    if dcol is None:
        return pd.Series(dtype=float)
    pcol = next((c for c in ["Adj Close", "Close", "adj_close", "close", "PRICE", "price", "value", "Value"] if c in df.columns), None)
    vcol = next((c for c in ["Volume", "volume", "VOL", "vol"] if c in df.columns), None)
    if pcol is None or vcol is None:
        return pd.Series(dtype=float)

    d = pd.to_datetime(df[dcol], errors="coerce")
    px = pd.to_numeric(df[pcol], errors="coerce")
    vol = pd.to_numeric(df[vcol], errors="coerce")
    adv = pd.Series((px * vol).values, index=d).replace([np.inf, -np.inf], np.nan).dropna()
    if adv.empty:
        return pd.Series(dtype=float)
    adv = adv[~adv.index.duplicated(keep="last")].sort_index()
    adv = adv.clip(lower=0.0)
    return adv.astype(float)


def _target_len() -> int:
    a = _load_mat(RUNS / "asset_returns.csv")
    if a is None:
        return 0
    return int(a.shape[0])


def _align_tail(x: np.ndarray, t: int, fill: float) -> np.ndarray:
    arr = np.asarray(x, float).ravel()
    if t <= 0:
        return arr
    if arr.size >= t:
        return arr[-t:]
    out = np.full(t, float(fill), dtype=float)
    if arr.size > 0:
        out[-arr.size :] = arr
        out[: t - arr.size] = float(arr[0])
    return out


def _build_adv_matrix(names: list[str], t: int, lookback: int) -> np.ndarray:
    n = len(names)
    adv = np.full((t, n), np.nan, dtype=float)
    for i, nm in enumerate(names):
        s = _read_adv_series(nm)
        if s.empty:
            continue
        s_roll = s.rolling(lookback, min_periods=max(5, lookback // 4)).mean().dropna()
        if s_roll.empty:
            continue
        v = _align_tail(s_roll.values, t, float(s_roll.iloc[0]))
        adv[:, i] = v

    finite = np.isfinite(adv) & (adv > 0.0)
    if not np.any(finite):
        return np.full((t, n), 1.0e7, dtype=float)

    row_med = np.nanmedian(np.where(finite, adv, np.nan), axis=1)
    finite_vals = adv[finite]
    global_med = float(np.median(finite_vals)) if finite_vals.size else 1.0e7
    row_med = np.where(np.isfinite(row_med), row_med, global_med)
    for i in range(n):
        col = adv[:, i]
        bad = ~np.isfinite(col) | (col <= 0.0)
        if np.any(bad):
            col = col.copy()
            col[bad] = row_med[bad]
            adv[:, i] = col

    adv = np.where(np.isfinite(adv) & (adv > 0.0), adv, max(global_med, 1.0e5))
    return adv


def main() -> int:
    w = _first_weights()
    t = _target_len()
    if w is None or t <= 0:
        print("(!) Missing weights/returns for capacity impact guard; writing neutral series.")
        if t > 0:
            np.savetxt(RUNS / "capacity_impact_proxy.csv", np.zeros(t), delimiter=",")
            np.savetxt(RUNS / "capacity_impact_scalar.csv", np.ones(t), delimiter=",")
            (RUNS / "capacity_impact_info.json").write_text(
                json.dumps({"ok": False, "reason": "missing_inputs", "rows": int(t)}, indent=2),
                encoding="utf-8",
            )
        return 0

    t = min(int(t), int(w.shape[0]))
    n = int(w.shape[1])
    w = np.asarray(w[:t, :n], float)
    names = _load_asset_names(n)

    lookback = int(np.clip(float(os.getenv("Q_CAPACITY_ADV_LOOKBACK", "20")), 5, 252))
    book_usd = float(np.clip(float(os.getenv("Q_CAPACITY_BOOK_USD", "100000")), 1000.0, 1.0e10))
    beta = float(np.clip(float(os.getenv("Q_CAPACITY_IMPACT_BETA", "0.35")), 0.0, 5.0))
    floor = float(np.clip(float(os.getenv("Q_CAPACITY_IMPACT_FLOOR", "0.70")), 0.1, 1.2))
    ceil = float(np.clip(float(os.getenv("Q_CAPACITY_IMPACT_CEIL", "1.05")), floor, 2.0))
    base = float(np.clip(float(os.getenv("Q_CAPACITY_IMPACT_BASE", "1.00")), 0.5, 1.5))
    sqrt_scale = float(np.clip(float(os.getenv("Q_CAPACITY_SQRT_SCALE", "20.0")), 0.1, 5000.0))

    adv = _build_adv_matrix(names, t, lookback)
    gross = np.sum(np.abs(w), axis=1)
    gross_safe = np.where(gross > 1e-9, gross, 1.0)

    # Approx participation: weighted average of per-asset book-vs-ADV usage.
    pos_usd = np.abs(w) * book_usd
    part = pos_usd / np.clip(adv, 1.0, np.inf)
    part_w = np.sum(part * np.abs(w), axis=1) / gross_safe
    part_w = np.clip(np.nan_to_num(part_w, nan=0.0, posinf=10.0, neginf=0.0), 0.0, 10.0)

    impact_proxy = np.sqrt(part_w * sqrt_scale)
    impact_proxy = np.clip(impact_proxy, 0.0, 3.0)
    scalar = np.clip(base - beta * impact_proxy, floor, ceil)

    np.savetxt(RUNS / "capacity_impact_proxy.csv", impact_proxy.astype(float), delimiter=",")
    np.savetxt(RUNS / "capacity_impact_scalar.csv", scalar.astype(float), delimiter=",")

    info = {
        "ok": True,
        "rows": int(t),
        "assets": int(n),
        "params": {
            "adv_lookback": lookback,
            "book_usd": book_usd,
            "beta": beta,
            "base": base,
            "floor": floor,
            "ceil": ceil,
            "sqrt_scale": sqrt_scale,
        },
        "impact_proxy_mean": float(np.mean(impact_proxy)),
        "impact_proxy_p95": float(np.percentile(impact_proxy, 95)),
        "scalar_mean": float(np.mean(scalar)),
        "scalar_min": float(np.min(scalar)),
        "scalar_max": float(np.max(scalar)),
    }
    (RUNS / "capacity_impact_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"✅ Wrote {RUNS/'capacity_impact_proxy.csv'}")
    print(f"✅ Wrote {RUNS/'capacity_impact_scalar.csv'}")
    print(f"✅ Wrote {RUNS/'capacity_impact_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
