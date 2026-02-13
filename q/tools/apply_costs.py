#!/usr/bin/env python3
# tools/apply_costs.py
# Applies simple trading costs to portfolio_plus.csv and final_* summaries.
# Cost model:
#   - mgmt_fee: annualized bps (e.g., 100 = 1%/yr) applied evenly to daily returns
#   - slip_bps: per-trade bps * daily turnover proxy (uses |ret| as proxy if turnover not available)
# Writes:
#   runs_plus/portfolio_plus_costs.csv
#   runs_plus/final_portfolio_summary_costs.json
#   runs_plus/final_portfolio_regime_summary_costs.json
#   runs_plus/final_portfolio_regime_dna_summary_costs.json

from pathlib import Path
import pandas as pd, numpy as np, json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"

# ---- knobs ----
MGMT_FEE_BPS = 100.0     # 1% per year
SLIP_BPS     = 2.0       # 2 bps per (proxy) turnover unit
DAYS = 252.0
# ---------------

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
def _split(r, frac=0.75):
    n = len(r); k = int(n*frac); return r[:k], r[k:]

def _load_ret_csv(path, ret_col_candidates):
    p = RUNS/path
    if not p.exists(): return None
    df = pd.read_csv(p)
    date_col = None
    for c in df.columns:
        if c.lower() in ("date","timestamp"): date_col = c; break
    if date_col is None and "DATE" in df.columns: date_col = "DATE"
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)
    for c in ret_col_candidates:
        if c in df.columns:
            r = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(-0.5,0.5)
            return df[[date_col]].rename(columns={date_col:"DATE"}) if date_col else None, r.values
    return None

def _apply_costs(ret):
    # daily management fee:
    fee = (MGMT_FEE_BPS/1e4)/DAYS   # e.g. 1%/yr -> ~0.0000397 per day
    # turnover proxy: use |ret| as a cheap proxy (you can replace later with true turnover)
    slip = (SLIP_BPS/1e4) * np.abs(ret)
    net = ret - fee - slip
    return net

def _summarize(ret):
    r_is, r_oos = _split(ret, 0.75)
    return {
        "in_sample":  {"sharpe": _ann_sharpe(r_is),  "hit": _hit(r_is),  "maxdd": _maxdd(r_is)},
        "out_sample": {"sharpe": _ann_sharpe(r_oos), "hit": _hit(r_oos), "maxdd": _maxdd(r_oos)},
        "note": f"Costs: fee={MGMT_FEE_BPS}bps/yr, slippage={SLIP_BPS}bps·|ret| proxy"
    }

if __name__ == "__main__":
    RUNS.mkdir(parents=True, exist_ok=True)

    # Base portfolio
    base = _load_ret_csv("portfolio_plus.csv", ["ret","return","port_ret","daily_ret"])
    if base is None: raise SystemExit("Missing portfolio_plus.csv / ret col")
    dates, r = base
    r_net = _apply_costs(r)
    out = pd.DataFrame({"DATE": dates["DATE"] if dates is not None else range(len(r_net)),
                        "ret_net_costs": r_net,
                        "eq_net_costs": (1.0 + pd.Series(r_net)).cumprod()})
    out.to_csv(RUNS/"portfolio_plus_costs.csv", index=False)
    (RUNS/"final_portfolio_summary_costs.json").write_text(json.dumps(_summarize(r_net), indent=2))

    # Regime
    reg = _load_ret_csv("final_portfolio_regime.csv", ["ret_governed","ret"])
    if reg is not None:
        _, r = reg
        r_net = _apply_costs(r)
        (RUNS/"final_portfolio_regime_summary_costs.json").write_text(json.dumps(_summarize(r_net), indent=2))

    # Regime+DNA
    dnap = _load_ret_csv("final_portfolio_regime_dna.csv", ["ret_governed_dna","ret"])
    if dnap is not None:
        _, r = dnap
        r_net = _apply_costs(r)
        (RUNS/"final_portfolio_regime_dna_summary_costs.json").write_text(json.dumps(_summarize(r_net), indent=2))

    print("✅ Costs applied and summaries written.")
