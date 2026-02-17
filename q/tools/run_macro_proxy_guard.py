#!/usr/bin/env python3
"""
Macro/forward-looking proxy guard.

Builds a market stress proxy from:
  - Vol term structure (VIX9D vs VIX3M or VIX/VIX3M fallback)
  - Yield-curve inversion (2Y - 10Y)
  - Credit stress (LQD/HYG ratio)
  - Front-end rate pressure (3M yield)

Writes:
  - runs_plus/macro_shock_proxy.csv   (0..1 stress)
  - runs_plus/macro_risk_scalar.csv   (bounded exposure scalar)
  - runs_plus/macro_proxy_info.json
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
RUNS.mkdir(exist_ok=True)


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


def _read_level(sym: str) -> pd.Series:
    p = DATA / f"{sym}.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.Series(dtype=float)
    if df.empty:
        return pd.Series(dtype=float)
    dcol = None
    for c in ["DATE", "Date", "date", "timestamp", "Timestamp"]:
        if c in df.columns:
            dcol = c
            break
    if dcol is None:
        return pd.Series(dtype=float)
    vcol = None
    for c in ["Close", "Adj Close", "close", "adj_close", "value", "Value", "PRICE", "price"]:
        if c in df.columns:
            vcol = c
            break
    if vcol is None:
        # fallback: first non-date numeric-ish column
        for c in df.columns:
            if c == dcol:
                continue
            try:
                pd.to_numeric(df[c], errors="raise")
                vcol = c
                break
            except Exception:
                continue
    if vcol is None:
        return pd.Series(dtype=float)
    d = pd.to_datetime(df[dcol], errors="coerce")
    v = pd.to_numeric(df[vcol], errors="coerce")
    s = pd.Series(v.values, index=d).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.astype(float)


def _roll_z(s: pd.Series, lookback: int, minp: int) -> pd.Series:
    if s.empty:
        return s
    mu = s.rolling(lookback, min_periods=minp).mean()
    sd = s.rolling(lookback, min_periods=minp).std(ddof=1).replace(0.0, np.nan)
    z = (s - mu) / (sd + 1e-12)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _target_len() -> int:
    p = RUNS / "asset_returns.csv"
    if not p.exists():
        return 0
    try:
        a = np.loadtxt(p, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(p, delimiter=",", skiprows=1)
        except Exception:
            return 0
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return int(a.shape[0])


def _align_tail(v: np.ndarray, t: int, fill: float) -> np.ndarray:
    x = np.asarray(v, float).ravel()
    if t <= 0:
        return x
    if x.size >= t:
        return x[-t:]
    out = np.full(t, float(fill), dtype=float)
    if x.size > 0:
        out[-x.size :] = x
        out[: t - x.size] = float(x[0])
    return out


def main() -> int:
    vix = _read_level("VIX9D")
    vix3m = _read_level("VIX3M")
    vix_spot = _read_level("VIXCLS")
    if vix_spot.empty:
        vix_spot = _read_level("VIX")
    dgs10 = _read_level("DGS10")
    dgs2 = _read_level("DGS2")
    dgs3m = _read_level("DGS3MO")
    hyg = _read_level("HYG")
    lqd = _read_level("LQD")

    idx = None
    for s in [vix, vix3m, vix_spot, dgs10, dgs2, dgs3m, hyg, lqd]:
        if not s.empty:
            idx = s.index if idx is None else idx.union(s.index)
    if idx is None or len(idx) == 0:
        print("(!) Macro proxy inputs missing; writing neutral series.")
        t = _target_len()
        if t <= 0:
            return 0
        shock = np.zeros(t, dtype=float)
        scalar = np.ones(t, dtype=float)
        np.savetxt(RUNS / "macro_shock_proxy.csv", shock, delimiter=",")
        np.savetxt(RUNS / "macro_risk_scalar.csv", scalar, delimiter=",")
        (RUNS / "macro_proxy_info.json").write_text(
            json.dumps({"ok": False, "reason": "missing_inputs", "rows": int(t)}, indent=2),
            encoding="utf-8",
        )
        return 0

    idx = pd.DatetimeIndex(idx).sort_values()
    df = pd.DataFrame(index=idx)
    for nm, s in [
        ("vix9d", vix),
        ("vix3m", vix3m),
        ("vix", vix_spot),
        ("dgs10", dgs10),
        ("dgs2", dgs2),
        ("dgs3m", dgs3m),
        ("hyg", hyg),
        ("lqd", lqd),
    ]:
        if not s.empty:
            df[nm] = s.reindex(idx).ffill()

    # Feature engineering with robust fallbacks.
    if {"vix9d", "vix3m"}.issubset(df.columns):
        vol_term = (df["vix9d"] - df["vix3m"]) / (df["vix3m"].abs() + 1e-6)
    elif {"vix", "vix3m"}.issubset(df.columns):
        vol_term = (df["vix"] - df["vix3m"]) / (df["vix3m"].abs() + 1e-6)
    else:
        vol_term = pd.Series(0.0, index=df.index)

    if "vix" in df.columns:
        vix_level = _roll_z(df["vix"], lookback=252, minp=40)
    else:
        vix_level = pd.Series(0.0, index=df.index)

    if {"dgs2", "dgs10"}.issubset(df.columns):
        curve_inv = _roll_z(df["dgs2"] - df["dgs10"], lookback=252, minp=60)
    else:
        curve_inv = pd.Series(0.0, index=df.index)

    if {"lqd", "hyg"}.issubset(df.columns):
        credit = _roll_z(np.log((df["lqd"] + 1e-6) / (df["hyg"] + 1e-6)), lookback=252, minp=60)
    else:
        credit = pd.Series(0.0, index=df.index)

    if "dgs3m" in df.columns:
        front_end = _roll_z(df["dgs3m"], lookback=252, minp=60)
    else:
        front_end = pd.Series(0.0, index=df.index)

    term_z = _roll_z(vol_term, lookback=252, minp=60)

    raw = (
        0.33 * np.clip(vix_level.values, -4.0, 4.0)
        + 0.27 * np.clip(term_z.values, -4.0, 4.0)
        + 0.22 * np.clip(curve_inv.values, -4.0, 4.0)
        + 0.12 * np.clip(credit.values, -4.0, 4.0)
        + 0.06 * np.clip(front_end.values, -4.0, 4.0)
    )
    raw = np.clip(raw, -6.0, 6.0)
    # logistic to 0..1, then smooth
    shock = 1.0 / (1.0 + np.exp(-raw))
    shock = pd.Series(shock, index=df.index).ewm(alpha=0.15, adjust=False).mean().values
    shock = np.clip(shock, 0.0, 1.0)

    beta = float(np.clip(float(os.getenv("Q_MACRO_PROXY_BETA", "0.28")), 0.0, 1.2))
    base = float(np.clip(float(os.getenv("Q_MACRO_PROXY_BASE", "1.02")), 0.5, 1.5))
    floor = float(np.clip(float(os.getenv("Q_MACRO_PROXY_FLOOR", "0.78")), 0.2, 1.2))
    ceil = float(np.clip(float(os.getenv("Q_MACRO_PROXY_CEIL", "1.05")), floor, 1.6))
    scalar = np.clip(base - beta * shock, floor, ceil)

    t = _target_len()
    if t > 0:
        shock = _align_tail(shock, t, 0.0)
        scalar = _align_tail(scalar, t, 1.0)

    np.savetxt(RUNS / "macro_shock_proxy.csv", np.asarray(shock, float), delimiter=",")
    np.savetxt(RUNS / "macro_risk_scalar.csv", np.asarray(scalar, float), delimiter=",")

    info = {
        "ok": True,
        "rows": int(len(shock)),
        "params": {
            "beta": beta,
            "base": base,
            "floor": floor,
            "ceil": ceil,
        },
        "inputs_present": {
            "vix": bool("vix" in df.columns),
            "vix3m": bool("vix3m" in df.columns),
            "vix9d": bool("vix9d" in df.columns),
            "dgs2": bool("dgs2" in df.columns),
            "dgs10": bool("dgs10" in df.columns),
            "dgs3m": bool("dgs3m" in df.columns),
            "hyg": bool("hyg" in df.columns),
            "lqd": bool("lqd" in df.columns),
        },
        "shock_mean": float(np.mean(shock)),
        "shock_max": float(np.max(shock)),
        "scalar_mean": float(np.mean(scalar)),
        "scalar_min": float(np.min(scalar)),
        "scalar_max": float(np.max(scalar)),
    }
    (RUNS / "macro_proxy_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Macro Proxy Guard ✔",
        (
            f"<p>Macro proxy rows={len(shock)}, shock_mean={info['shock_mean']:.3f}, "
            f"scalar_mean={info['scalar_mean']:.3f}, scalar_range=[{info['scalar_min']:.3f},{info['scalar_max']:.3f}].</p>"
        ),
    )
    print(f"✅ Wrote {RUNS/'macro_shock_proxy.csv'}")
    print(f"✅ Wrote {RUNS/'macro_risk_scalar.csv'}")
    print(f"✅ Wrote {RUNS/'macro_proxy_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
