#!/usr/bin/env python3
# tools/make_overlay_alpha.py
# Builds a scalar overlay alpha in [ALPHA_MIN, ALPHA_MAX] per day using:
#  - Breadth = % of portfolio assets with positive 20d return
#  - Calmness = inverse of realized daily vol on the main portfolio (20d)
# The two are z-scored and combined, then squashed to [ALPHA_MIN, ALPHA_MAX].
#
# Writes: runs_plus/overlay_alpha.csv  (DATE, overlay_alpha)

from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"
DATA = ROOT/"data"

# ---- knobs ----
RET_WINDOW   = 14          # days for breadth & realized vol
ALPHA_MIN    = 0.75        # floor scale
ALPHA_MAX    = 1.3        # cap scale
BREADTH_W    = 0.6         # weight on breadth vs calmness
CALM_W       = 0.4
DATE_COL     = "Date"
PRICE_COLS   = ["Close","Adj Close","Adjusted Close","close","Price","Last"]
# ---------------

def _zscore(x):
    s = pd.Series(x)
    m, sd = s.mean(), s.std()
    if not np.isfinite(sd) or sd == 0: return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - m) / sd

def _load_portfolio_returns():
    p = RUNS/"portfolio_plus.csv"
    if not p.exists():
        raise SystemExit("Missing runs_plus/portfolio_plus.csv (build main first).")
    df = pd.read_csv(p)
    dcol = None
    for c in df.columns:
        if str(c).lower() in ("date","timestamp"):
            dcol = c; break
    if dcol:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol]).sort_values(dcol)
    # returns
    for c in ["ret","ret_plus","ret_net","daily_ret","port_ret","portfolio_ret","return"]:
        if c in df.columns:
            r = pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
            return df[dcol].values if dcol else None, r.values
    # or from equity
    for c in ["eq","eq_net","equity","equity_curve","equity_index","portfolio_eq"]:
        if c in df.columns:
            eq = pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan)
            r = eq.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
            return df[dcol].values if dcol else None, r.values
    raise SystemExit("portfolio_plus.csv has no recognizable returns/equity columns.")

def _load_asset_panel(asset_list):
    panel = None
    for a in asset_list:
        p = DATA/f"{a}.csv"
        if not p.exists(): 
            continue
        df = pd.read_csv(p)
        dcol = DATE_COL if DATE_COL in df.columns else next((c for c in df.columns if str(c).lower() in ("date","timestamp")), None)
        if dcol is None: 
            continue
        if any(c in df.columns for c in PRICE_COLS):
            pxcol = next(c for c in PRICE_COLS if c in df.columns)
        else:
            nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not nums: 
                continue
            pxcol = nums[-1]
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol]).sort_values(dcol)
        ret = pd.to_numeric(df[pxcol], errors="coerce").pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
        p1 = pd.DataFrame({"DATE": df[dcol], a: ret.values})
        panel = p1 if panel is None else panel.merge(p1, on="DATE", how="outer")
    if panel is not None:
        panel = panel.sort_values("DATE").fillna(0.0).reset_index(drop=True)
    return panel

if __name__ == "__main__":
    # list of current portfolio assets
    wpath = RUNS/"portfolio_weights.csv"
    if not wpath.exists():
        raise SystemExit("Missing runs_plus/portfolio_weights.csv (build main first).")
    w = pd.read_csv(wpath)
    assets = [a for a in w["asset"].tolist() if isinstance(a,str) and a.strip()]

    # asset panel for breadth
    panel = _load_asset_panel(assets)
    if panel is None or panel.empty:
        raise SystemExit("Could not build asset return panel for breadth.")
    # breadth: % assets with positive RET_WINDOW total return
    R = panel.set_index("DATE")[assets]
    roll_sum = (1.0 + R).rolling(RET_WINDOW, min_periods=max(5,RET_WINDOW//3)).apply(lambda x: float(np.prod(1.0+x)-1.0), raw=False)
    breadth = (roll_sum > 0.0).mean(axis=1)  # fraction of assets up over window

    # calmness: inverse of realized vol of main portfolio returns
    dates, port_r = _load_portfolio_returns()
    s = pd.Series(port_r, index=pd.to_datetime(dates))
    rv = s.rolling(RET_WINDOW, min_periods=max(5,RET_WINDOW//3)).std()
    calm = -rv  # lower vol → higher calm (more positive)

    # align indexes
    df = pd.concat([breadth, calm], axis=1).dropna()
    df.columns = ["breadth","calm"]
    zb = _zscore(df["breadth"])
    zc = _zscore(df["calm"])
    z = BREADTH_W*zb + CALM_W*zc

    # squash to [ALPHA_MIN, ALPHA_MAX] using tanh
    z_clip = np.tanh(z.values)  # -1..1
    scale = (ALPHA_MIN + ALPHA_MAX)/2.0
    half  = (ALPHA_MAX - ALPHA_MIN)/2.0
    alpha = scale + half * z_clip

    out = pd.DataFrame({"DATE": df.index, "overlay_alpha": alpha})
    out.to_csv(RUNS/"overlay_alpha.csv", index=False)
    print("✅ Wrote runs_plus/overlay_alpha.csv")
