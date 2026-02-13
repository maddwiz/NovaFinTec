#!/usr/bin/env python3
# tools/blend_osc_with_main.py  (v2: safe dd + flexible columns)
from pathlib import Path
import pandas as pd, numpy as np, json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

def smart_load(path: Path):
    df = pd.read_csv(path)
    lowers={c.lower():c for c in df.columns}
    dcol = lowers.get("date") or lowers.get("timestamp") or lowers.get("time") or df.columns[0]
    df[dcol]=pd.to_datetime(df[dcol], errors="coerce")
    df=df.dropna(subset=[dcol]).sort_values(dcol)

    # return column candidates
    cands = ["ret_net","ret","ret_plus","ret_gross","return","pnl","pnl_plus","daily_ret","portfolio_ret","port_ret"]
    r=None
    for c in cands:
        if c in df.columns:
            s=pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum()>5: r=s; break
    if r is None:
        for c in ["eq","equity","equity_curve","portfolio_eq","equity_index","port_equity"]:
            if c in df.columns:
                eq=pd.to_numeric(df[c], errors="coerce")
                if eq.notna().sum()>5: r=eq.pct_change(); break
    if r is None: raise SystemExit(f"No returns in {path.name}")

    r = r.replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5, 0.5)
    return pd.DataFrame({"DATE": df[dcol], "ret": r})

def ann_sharpe(r):
    r=pd.Series(r).replace([np.inf,-np.inf], np.nan).dropna()
    s=r.std()
    if s==0 or np.isnan(s): return 0.0
    return float((r.mean()/s)*(252**0.5))

if __name__=="__main__":
    main_p = RUNS/"portfolio_plus.csv"
    osc_p  = RUNS/"osc_portfolio.csv"
    if not main_p.exists(): raise SystemExit("Missing runs_plus/portfolio_plus.csv")
    if not osc_p.exists():  raise SystemExit("Missing runs_plus/osc_portfolio.csv")

    a=smart_load(main_p)
    b=smart_load(osc_p)
    df=pd.merge(a,b,on="DATE",how="inner",suffixes=("_main","_osc")).sort_values("DATE")
    if df.empty: raise SystemExit("No overlapping dates between main and oscillator")

    best={"alpha":None,"sharpe":-9,"hit":0,"maxdd":0}; rows=[]
    for i in range(21):
        alpha=i/20.0
        r=alpha*df["ret_main"] + (1-alpha)*df["ret_osc"]
        r=r.replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
        eq=(1+r).cumprod()
        peak = pd.concat([eq, pd.Series(1.0, index=eq.index)], axis=1).max(axis=1).cummax()
        dd_series = (eq/peak) - 1.0
        sh=ann_sharpe(r)
        hit=float((r>0).mean())
        dd=float(np.nan_to_num(dd_series.min(), nan=0.0, posinf=0.0, neginf=-1.0))
        rows.append((alpha,sh,hit,dd))
        if sh>best["sharpe"]:
            best={"alpha":float(alpha),"sharpe":float(sh),"hit":hit,"maxdd":dd}

    out=pd.DataFrame({"DATE":df["DATE"],
                      "ret":best["alpha"]*df["ret_main"]+(1-best["alpha"])*df["ret_osc"],
                      "alpha":best["alpha"]})
    out["ret"]=out["ret"].replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
    out["eq"]=(1+out["ret"]).cumprod()
    out.to_csv(RUNS/"blend_osc_main.csv", index=False)
    (RUNS/"blend_osc_main_summary.json").write_text(json.dumps(best, indent=2))

    print(" alpha   Sharpe    Hit   MaxDD")
    for a_,sh,h,dd in rows:
        print(f" {a_:>4.2f}   {sh:>6.3f}  {h:>5.3f}  {dd:>6.3f}")
    print("\nBEST:",best)
    print("Saved: runs_plus/blend_osc_main.csv and blend_osc_main_summary.json")
