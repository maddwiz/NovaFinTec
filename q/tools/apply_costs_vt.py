#!/usr/bin/env python3
# tools/apply_costs_vt.py
# Applies the same costs model to Vol-Target returns and writes a *_costs.json summary.

from pathlib import Path
import pandas as pd, numpy as np, json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"

# ---- knobs (match your main costs) ----
MGMT_FEE_BPS = 100.0   # 1%/yr
SLIP_BPS     = 2.0     # 2 bps * |ret| proxy
DAYS = 252.0
IS_FRAC = 0.75
# ---------------------------------------

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
def _hit(r): s = pd.Series(r).dropna(); return float((s>0).mean()) if not s.empty else 0.0

def _split(r, frac=IS_FRAC):
    n=len(r); k=int(n*frac); return r[:k], r[k:]

def _load_vt():
    p = RUNS/"portfolio_plus_vt.csv"
    if not p.exists(): raise SystemExit("Missing runs_plus/portfolio_plus_vt.csv")
    df = pd.read_csv(p)
    col = "ret_vt" if "ret_vt" in df.columns else None
    if col is None:
        # fall back to pct-change of eq_vt if needed
        if "eq_vt" in df.columns:
            r = pd.to_numeric(df["eq_vt"], errors="coerce").pct_change().fillna(0.0).clip(-0.5,0.5).values
            return r
        raise SystemExit("portfolio_plus_vt.csv lacks ret_vt/eq_vt")
    r = pd.to_numeric(df[col], errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5).values
    return r

def _apply_costs(r):
    fee = (MGMT_FEE_BPS/1e4)/DAYS
    slip = (SLIP_BPS/1e4) * np.abs(r)
    return r - fee - slip

if __name__ == "__main__":
    r = _load_vt()
    r_net = _apply_costs(r)
    r_is, r_oos = _split(r_net)
    out = {
        "in_sample":  {"sharpe": _ann_sharpe(r_is), "hit": _hit(r_is), "maxdd": _maxdd(r_is)},
        "out_sample": {"sharpe": _ann_sharpe(r_oos),"hit": _hit(r_oos),"maxdd": _maxdd(r_oos)},
        "note": f"Costs: {MGMT_FEE_BPS}bps/yr mgmt + {SLIP_BPS}bps·|ret| proxy"
    }
    (RUNS/"final_portfolio_vt_summary_costs.json").write_text(json.dumps(out, indent=2))
    print("✅ Wrote final_portfolio_vt_summary_costs.json")
