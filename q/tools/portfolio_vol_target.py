#!/usr/bin/env python3
# tools/portfolio_vol_target.py
# Creates a volatility-targeted version of your Main portfolio.
# Target is annualized vol; we scale daily returns by target / realized_vol.
# Realized vol = rolling stdev over window (default 20d). Scale bounded for safety.
#
# Outputs:
#   runs_plus/portfolio_plus_vt.csv
#   runs_plus/final_portfolio_vt_summary.json

from pathlib import Path
import pandas as pd, numpy as np, json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(parents=True, exist_ok=True)

# ---- knobs ----
TARGET_ANN   = 0.12   # 10% annualized vol target (try 0.08–0.12 later)
ROLL_DAYS    = 20     # lookback for realized vol
SCALE_MIN    = 0.50   # don’t cut risk by more than half in one step
SCALE_MAX    = 2.00   # don’t lever more than 2x in one step
IS_FRAC      = 0.75   # same split as your report
RET_COL_CAND = ["ret","ret_plus","ret_net","daily_ret","port_ret","portfolio_ret","return"]
EQ_COL_CAND  = ["eq","eq_net","equity","equity_curve","equity_index","portfolio_eq"]
# ---------------

DAYS = 252.0
TARGET_D = TARGET_ANN / np.sqrt(DAYS)

def _ann_sharpe(r):
    s = pd.Series(r).replace([np.inf,-np.inf], np.nan).dropna()
    if s.empty: return 0.0
    sd = s.std()
    if not np.isfinite(sd) or sd==0: return 0.0
    return float((s.mean()/sd)*np.sqrt(DAYS))
def _maxdd(r):
    s = pd.Series(r).fillna(0.0)
    eq = (1.0 + s).cumprod()
    dd = eq/eq.cummax() - 1.0
    return float(dd.min())
def _hit(r):
    s = pd.Series(r).dropna()
    return float((s>0).mean()) if not s.empty else 0.0

def _split(vals, frac):
    n = len(vals); k = int(n*frac)
    return vals[:k], vals[k:]

def _load_port():
    p = RUNS/"portfolio_plus.csv"
    if not p.exists():
        raise SystemExit("Missing runs_plus/portfolio_plus.csv (build your main portfolio first).")
    df = pd.read_csv(p)
    # date sort
    dcol = None
    for c in df.columns:
        if str(c).lower() in ("date","timestamp"): dcol = c; break
    if dcol:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol]).sort_values(dcol)
    # returns
    for c in RET_COL_CAND:
        if c in df.columns:
            r = pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
            return df[[dcol]].rename(columns={dcol:"DATE"}) if dcol else None, r.values
    # or from equity
    for c in EQ_COL_CAND:
        if c in df.columns:
            eq = pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan)
            r = eq.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
            return df[[dcol]].rename(columns={dcol:"DATE"}) if dcol else None, r.values
    raise SystemExit("portfolio_plus.csv has no recognizable returns/equity columns.")

if __name__ == "__main__":
    dates, r = _load_port()
    s = pd.Series(r)
    # realized daily vol
    rv = s.rolling(ROLL_DAYS, min_periods=max(5, ROLL_DAYS//3)).std().replace(0,np.nan)
    scale = (TARGET_D / rv).clip(lower=SCALE_MIN, upper=SCALE_MAX).fillna(1.0)
    r_vt = (s * scale).astype(float).values

    # outputs
    out = pd.DataFrame({
        "DATE": dates["DATE"] if dates is not None else range(len(r_vt)),
        "ret_vt": r_vt,
        "eq_vt": (1.0 + pd.Series(r_vt)).cumprod()
    })
    out.to_csv(RUNS/"portfolio_plus_vt.csv", index=False)

    r_is, r_oos = _split(r_vt, IS_FRAC)
    summary = {
        "in_sample":  {"sharpe": _ann_sharpe(r_is),  "hit": _hit(r_is),  "maxdd": _maxdd(r_is)},
        "out_sample": {"sharpe": _ann_sharpe(r_oos), "hit": _hit(r_oos), "maxdd": _maxdd(r_oos)},
        "target_ann": TARGET_ANN, "roll_days": ROLL_DAYS,
        "scale_bounds": [SCALE_MIN, SCALE_MAX]
    }
    (RUNS/"final_portfolio_vt_summary.json").write_text(json.dumps(summary, indent=2))
    print("✅ Wrote portfolio_plus_vt.csv and final_portfolio_vt_summary.json")
    print("OOS Sharpe=%.3f  OOS MaxDD=%.3f" % (summary["out_sample"]["sharpe"], summary["out_sample"]["maxdd"]))
