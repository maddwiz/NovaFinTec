#!/usr/bin/env python3
"""
WF Terminal Summary (prints a clean table).
"""

import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
np.set_printoptions(suppress=True, linewidth=120)

def maybe_load_series(path: Path):
    if path.exists():
        try:
            return np.loadtxt(path, delimiter=",").ravel()
        except Exception:
            try:
                return np.loadtxt(path, delimiter=",", skiprows=1).ravel()
            except Exception:
                return None
    return None

def maybe_load_matrix(path: Path):
    if path.exists():
        try:
            a = np.loadtxt(path, delimiter=",")
        except Exception:
            try:
                a = np.loadtxt(path, delimiter=",", skiprows=1)
            except Exception:
                return None
        if a.ndim == 1: a = a.reshape(-1,1)
        return a
    return None

def sharpe(r):
    r = np.asarray(r,float)
    if r.size==0: return float("nan")
    mu=np.nanmean(r); sd=np.nanstd(r)+1e-12
    return float((mu/sd)*np.sqrt(252.0))

def hit_rate(r):
    r = np.asarray(r,float)
    if r.size==0: return float("nan")
    pos=(r>0).sum(); tot=len(r)
    return float(pos/tot)

def max_drawdown(cum):
    peak=-1e18; mdd=0.0
    for x in cum:
        peak=max(peak,x)
        mdd=min(mdd,x-peak)
    return float(mdd)

def fmt(x,digs=3):
    if x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x))): return "—"
    return f"{x:.{digs}f}"

def line(w=72,ch="─"): return ch*w
def row(cols,widths): return "│ " + " │ ".join(f"{c:<{w}}" for c,w in zip(cols,widths)) + " │"

if __name__=="__main__":
    # returns
    r = maybe_load_series(RUNS/"wf_oos_returns.csv") or maybe_load_series(RUNS/"daily_returns.csv")
    if r is None:
        print("(!) No returns found in runs_plus/. Run WF first."); sys.exit(0)

    s=sharpe(r); hr=hit_rate(r); cum=np.cumsum(r); mdd=max_drawdown(cum)

    # weights
    W=None
    for path in [ROOT/"portfolio_weights.csv", RUNS/"portfolio_weights.csv"]:
        mat=maybe_load_matrix(path)
        if mat is not None: W=mat; break
    N=int(W.shape[1]) if W is not None else None

    print("\n"+line())
    print("│ Walk-Forward Summary".ljust(72)+"│")
    print(line())
    widths=[26,43]
    print(row(["Metric","Value"],widths))
    print(line())
    print(row(["Assets (N)", str(N) if N else "—"],widths))
    print(row(["Sharpe (annualized)", fmt(s,3)],widths))
    print(row(["Hit rate", fmt(hr,3)],widths))
    print(row(["Max drawdown", fmt(mdd,3)],widths))
    print(line()+"\n")

    # tail returns
    tail=r[-5:] if r.size>=5 else r
    if tail.size>0:
        print("Last few daily returns:")
        for i,v in enumerate(tail,1):
            print(f"  t-{len(tail)-i:>2}: {v:+.5f}")
        print()
