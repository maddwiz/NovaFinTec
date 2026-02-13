#!/usr/bin/env python3
"""
run_vol_overlay.py  (v3 robust: no look-ahead, leverage clamp, winsorized tails)

Keeps the volatility gate + tiny direction cue, but fixes stability:
- EWMA vol estimates (smoother)
- Scale uses *lagged* port vol (no look-ahead)
- Clamp scale within [MIN_LEV, MAX_LEV]
- Winsorize daily returns to cut tail spikes

Env overrides (optional):
  CALM_TH=0.30     # annualized realized vol threshold for "calm" (default 0.30)
  STRESS_CUT=0.6   # exposure multiplier in stress (default 0.6)
  TARGET_VOL=0.10  # target portfolio vol (default 0.10)
  MOM_WIN=21       # momentum window for tiny direction signal (default 21)
  VOL_WIN=21       # window for realized vol per-asset (default 21)
  RP_WIN=63        # window for risk-parity vol (default 63)
  EWMA_HALFLIFE=30 # EWMA half-life for port vol scaling (default 30)
  MIN_LEV=0.5      # min leverage clamp (default 0.5)
  MAX_LEV=1.5      # max leverage clamp (default 1.5)
  WINSOR=0.01      # winsorize tails at +/-1% quantiles (default 0.01)
"""

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True, parents=True)

def load_series(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "DATE" not in df.columns or "VALUE" not in df.columns:
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date", list(df.columns)[0])
        vcol = cols.get("value", list(df.columns)[-1])
        df = pd.DataFrame({"DATE": df[dcol], "VALUE": df[vcol]})
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna().sort_values("DATE")
    return df

def daily_returns(price: pd.Series):
    return price.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0)

def realized_vol(r, window=21):
    return r.rolling(window).std() * np.sqrt(252)

def ann_sharpe(r):
    r = pd.Series(r).dropna()
    s = r.std()
    if s == 0 or np.isnan(s): return 0.0
    return float((r.mean() / s) * np.sqrt(252.0))

def winsorize(s: pd.Series, q=0.01):
    lo, hi = s.quantile(q), s.quantile(1-q)
    return s.clip(lo, hi)

if __name__ == "__main__":
    CALM_TH = float(os.getenv("CALM_TH", "0.30"))
    STRESS_CUT = float(os.getenv("STRESS_CUT", "0.6"))
    TARGET_VOL = float(os.getenv("TARGET_VOL", "0.10"))
    MOM_WIN = int(os.getenv("MOM_WIN", "21"))
    VOL_WIN = int(os.getenv("VOL_WIN", "21"))
    RP_WIN = int(os.getenv("RP_WIN", "63"))
    EWMA_HALFLIFE = int(os.getenv("EWMA_HALFLIFE", "30"))
    MIN_LEV = float(os.getenv("MIN_LEV", "0.5"))
    MAX_LEV = float(os.getenv("MAX_LEV", "1.5"))
    WINSOR = float(os.getenv("WINSOR", "0.01"))

    # 1) per-asset vol-gated, tiny-direction PnL streams
    frames = []
    for p in sorted(DATA.glob("*.csv")):
        sym = p.stem
        try:
            df = load_series(p)
        except Exception:
            continue
        if len(df) < max(200, RP_WIN + MOM_WIN + VOL_WIN):
            continue

        px = pd.to_numeric(df["VALUE"], errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
        if px.empty: 
            continue

        ret = daily_returns(px)
        vol = realized_vol(ret, VOL_WIN)

        mom = px.pct_change(MOM_WIN)
        direction = np.sign(mom).fillna(0.0)  # {-1,0,1}

        gate = pd.Series(1.0, index=ret.index)
        gate[vol > CALM_TH] = STRESS_CUT

        pos = (direction * gate).clip(-1.0, 1.0)
        pnl = (pos * ret).rename(sym)

        frames.append(pd.DataFrame({"DATE": df["DATE"].iloc[:len(pnl)], sym: pnl.values}))

    if not frames:
        raise SystemExit("No assets available for vol overlay.")

    wide = frames[0]
    for f in frames[1:]:
        wide = pd.merge(wide, f, on="DATE", how="outer")
    wide = wide.sort_values("DATE").fillna(0.0).set_index("DATE")

    # 2) daily equal-risk weights (risk parity via 1/vol over RP_WIN)
    rolling_vols = wide.rolling(RP_WIN).std()
    inv = 1.0 / rolling_vols.replace(0, np.nan)
    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w = inv.div(inv.abs().sum(axis=1), axis=0).fillna(0.0)

    # 3) portfolio returns (winsorize tails)
    port_ret = (wide * w).sum(axis=1)
    if WINSOR > 0:
        port_ret = winsorize(port_ret, q=WINSOR)

    # 4) EWMA vol estimate, lagged (no look-ahead), clamp leverage
    ewma_vol = port_ret.ewm(halflife=EWMA_HALFLIFE, adjust=False).std()
    lag_vol = ewma_vol.shift(1)  # use yesterday’s vol
    lag_vol = lag_vol.replace(0, np.nan)

    scale = (TARGET_VOL / lag_vol).replace([np.inf,-np.inf], np.nan).fillna(1.0)
    scale = scale.clip(lower=MIN_LEV, upper=MAX_LEV)

    port_ret = (port_ret * scale).fillna(0.0)

    # 5) stats & save
    eq = (1 + port_ret).cumprod()
    sharpe = ann_sharpe(port_ret)
    hit = (port_ret > 0).mean()
    dd = float((eq / eq.cummax() - 1.0).min())

    out = pd.DataFrame({"DATE": port_ret.index, "ret": port_ret.values, "eq": eq.values})
    out.to_csv(RUNS / "vol_overlay.csv", index=False)
    (RUNS / "vol_overlay_summary.json").write_text(json.dumps({
        "sharpe": float(sharpe),
        "hit": float(hit),
        "maxdd": dd,
        "calm_threshold": CALM_TH,
        "stress_cut": STRESS_CUT,
        "target_vol": TARGET_VOL,
        "mom_win": MOM_WIN,
        "vol_win": VOL_WIN,
        "riskparity_win": RP_WIN,
        "ewma_halflife": EWMA_HALFLIFE,
        "min_lev": MIN_LEV,
        "max_lev": MAX_LEV,
        "winsor": WINSOR
    }, indent=2))

    print("\nVOLATILITY OVERLAY PORTFOLIO (v3 robust)")
    print("========================================")
    print(f"Sharpe: {sharpe:.3f}")
    print(f"Hit:    {hit:.3f}")
    print(f"MaxDD:  {dd:.3f}")
    print(f"(Assets used: {wide.shape[1]})")
    print(f"Saved: {RUNS/'vol_overlay.csv'} and vol_overlay_summary.json")
