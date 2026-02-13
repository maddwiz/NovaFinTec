#!/usr/bin/env python3
# tools/robust_time_splits.py
# Re-compute IS/OOS metrics at multiple split points for:
#   - Main:    runs_plus/portfolio_plus.csv (ret or eq)
#   - Regime:  runs_plus/final_portfolio_regime.csv (ret_governed or ret)
#   - DNA:     runs_plus/final_portfolio_regime_dna.csv (ret_governed_dna or ret)
# Writes: runs_plus/robust_time_splits.csv and prints a compact table.

from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

# Try a few IS/OOS boundaries
SPLITS = [0.65, 0.70, 0.75, 0.80]

def _ann_sharpe(r):
    s = pd.Series(r).replace([np.inf,-np.inf], np.nan).dropna()
    if s.empty: return np.nan
    sd = s.std()
    if not np.isfinite(sd) or sd==0: return np.nan
    return float((s.mean()/sd) * np.sqrt(252.0))
def _maxdd(r):
    s = pd.Series(r).fillna(0.0)
    eq = (1.0 + s).cumprod()
    dd = eq/eq.cummax() - 1.0
    return float(dd.min())
def _hit(r):
    s = pd.Series(r).dropna()
    return float((s>0).mean()) if not s.empty else np.nan

def _load_series(rel, ret_candidates, eq_candidates):
    p = RUNS / rel
    if not p.exists(): return None
    df = pd.read_csv(p)
    # date & sort
    dcol = None
    for c in df.columns:
        if c.lower() in ("date","timestamp"): dcol = c; break
    if dcol and dcol in df.columns:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol]).sort_values(dcol)
    # returns if present
    for c in ret_candidates:
        if c in df.columns:
            r = pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
            return r.values
    # else derive from equity
    for c in eq_candidates:
        if c in df.columns:
            eq = pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan)
            r = eq.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
            return r.values
    return None

def _metrics_at_split(r, frac):
    n = len(r); k = max(30, min(n-30, int(n*frac)))
    r_is, r_oos = r[:k], r[k:]
    return dict(
        is_sharpe=_ann_sharpe(r_is), oos_sharpe=_ann_sharpe(r_oos),
        is_hit=_hit(r_is), oos_hit=_hit(r_oos),
        is_maxdd=_maxdd(r_is), oos_maxdd=_maxdd(r_oos),
        n_is=len(r_is), n_oos=len(r_oos)
    )

if __name__ == "__main__":
    main = _load_series("portfolio_plus.csv",
                        ["ret","ret_net","ret_plus","return","daily_ret","port_ret","portfolio_ret"],
                        ["eq","eq_net","equity","equity_curve","equity_index","portfolio_eq"])
    reg  = _load_series("final_portfolio_regime.csv",
                        ["ret_governed","ret"], ["eq","equity","equity_curve"])
    dna  = _load_series("final_portfolio_regime_dna.csv",
                        ["ret_governed_dna","ret"], ["eq","equity","equity_curve"])

    rows = []
    for f in SPLITS:
        row = {"split": f}
        if main is not None:
            m = _metrics_at_split(main, f)
            row.update({f"main_{k}": v for k,v in m.items()})
        if reg is not None:
            m = _metrics_at_split(reg, f)
            row.update({f"reg_{k}": v for k,v in m.items()})
        if dna is not None:
            m = _metrics_at_split(dna, f)
            row.update({f"dna_{k}": v for k,v in m.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    outp = RUNS / "robust_time_splits.csv"
    df.to_csv(outp, index=False)

    # compact console view
    show_cols = ["split",
                 "main_oos_sharpe","reg_oos_sharpe","dna_oos_sharpe",
                 "main_oos_maxdd","reg_oos_maxdd","dna_oos_maxdd"]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    print("Saved:", outp)
