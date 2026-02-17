#!/usr/bin/env python3
"""
WF Terminal Summary (prints a clean table).
"""

import json, os, sys
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

def pick_returns_series(runs_dir: Path, source_pref: str = "auto"):
    pref = str(source_pref or "auto").strip().lower()
    candidates = {
        "wf_oos_returns": runs_dir / "wf_oos_returns.csv",
        "daily_returns": runs_dir / "daily_returns.csv",
    }

    def _load(label: str):
        p = candidates[label]
        if not p.exists():
            return None, None
        arr = maybe_load_series(p)
        if arr is None or np.asarray(arr, float).size == 0:
            return None, None
        return np.asarray(arr, float).ravel(), p

    if pref in {"wf", "wf_oos", "wf_oos_returns"}:
        arr, _ = _load("wf_oos_returns")
        return arr, "wf_oos_returns"
    if pref in {"daily", "daily_returns"}:
        arr, _ = _load("daily_returns")
        return arr, "daily_returns"

    loaded = []
    tie_pref = {"wf_oos_returns": 0, "daily_returns": 1}
    for label in ["wf_oos_returns", "daily_returns"]:
        arr, path = _load(label)
        if arr is not None and path is not None:
            try:
                mtime = float(path.stat().st_mtime)
            except Exception:
                mtime = -1.0
            loaded.append((mtime, int(tie_pref.get(label, 0)), label, arr))
    if not loaded:
        return None, None
    loaded.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, label, arr = loaded[0]
    return arr, label

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
    source_pref = str(os.getenv("Q_PRINT_RESULTS_SOURCE", "auto")).strip().lower()
    r, returns_source = pick_returns_series(RUNS, source_pref=source_pref)
    if r is None:
        print("(!) No returns found in runs_plus/. Run WF first."); sys.exit(0)

    s=sharpe(r); hr=hit_rate(r); cum=np.cumsum(r); mdd=max_drawdown(cum)

    # weights
    W=None
    for path in [RUNS/"portfolio_weights_final.csv", ROOT/"portfolio_weights.csv", RUNS/"portfolio_weights.csv"]:
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
    print(row(["Returns source", str(returns_source or "—")],widths))
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
