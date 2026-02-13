#!/usr/bin/env python3
"""
build_portfolio_plus.py v6 (auto-tuned + hive caps + hive vol targeting)

Priority:
  1) ENV: CAP_PER, COST_BPS, LOOKBACK, HIVE_CAP, HIVE_VOL_LOOKBACK, HIVE_VOL_PENALTY
  2) runs_plus/portfolio_tuning_best.json (may include HIVE_CAP)
  3) defaults

Logic:
  - base weights ∝ inverse asset vol
  - per-asset CAP_PER -> renorm
  - per-hive vol targeting: scale each hive by inverse hive vol^penalty
  - per-hive HIVE_CAP -> renorm
  - compute pnl - turnover costs
"""

from pathlib import Path
import os, json, math
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

def _float(env, key, default): 
    try: return float(os.environ.get(key, default))
    except: return float(default)

def _int(env, key, default):
    try: return int(float(os.environ.get(key, default)))
    except: return int(default)

def load_tuned_defaults():
    d={"CAP_PER":0.10,"COST_BPS":2.0,"LOOKBACK":63,"HIVE_CAP":0.40}
    p=RUNS/"portfolio_tuning_best.json"
    if p.exists():
        try:
            j=json.loads(p.read_text())
            d["CAP_PER"]=float(j.get("CAP_PER", d["CAP_PER"]))
            d["COST_BPS"]=float(j.get("COST_BPS", d["COST_BPS"]))
            d["LOOKBACK"]=int(j.get("LOOKBACK", d["LOOKBACK"]))
            d["HIVE_CAP"]=float(j.get("HIVE_CAP", d["HIVE_CAP"]))
        except: pass
    return d

def pick(envkey, tuned, default):
    v=os.environ.get(envkey, "")
    if str(v).strip()=="":
        return tuned if tuned is not None else default
    try:
        return int(float(v)) if envkey=="LOOKBACK" else float(v)
    except:
        return tuned if tuned is not None else default

def safe_mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)

def load_oos_plus():
    out={}
    for d in RUNS.iterdir():
        if not d.is_dir(): continue
        p=d/"oos_plus.csv"
        if not p.exists(): continue
        df=pd.read_csv(p, parse_dates=["date"])
        need={"date","ret","pos_plus","pnl_plus"}
        if not need.issubset(df.columns): continue
        sym=d.name
        df=df[["date","ret","pos_plus","pnl_plus"]].rename(columns={"ret":f"{sym}_ret","pos_plus":f"{sym}_pos","pnl_plus":f"{sym}_pnl"})
        out[sym]=df
    return out

def load_hives():
    p=RUNS/"hive.json"
    if not p.exists(): return {}
    try:
        j=json.loads(p.read_text())
        return j.get("hives",{})
    except:
        return {}

def max_drawdown(eq):
    peak=eq.cummax(); dd=eq/peak-1.0
    return float(dd.min())

def sharpe(pnl):
    s=pnl.dropna()
    return float((s.mean()/(s.std()+1e-9))*np.sqrt(252)) if len(s) else float("nan")

if __name__=="__main__":
    tuned=load_tuned_defaults()
    CAP_PER  = pick("CAP_PER",  tuned.get("CAP_PER"),  0.10)
    COST_BPS = pick("COST_BPS", tuned.get("COST_BPS"), 2.0)
    LOOKBACK = int(pick("LOOKBACK", tuned.get("LOOKBACK"), 63))
    HIVE_CAP = pick("HIVE_CAP", tuned.get("HIVE_CAP"), 0.40)
    HIVE_VOL_LOOKBACK = int(os.environ.get("HIVE_VOL_LOOKBACK", 63))  # hive vol window
    HIVE_VOL_PENALTY  = float(os.environ.get("HIVE_VOL_PENALTY", 1.0)) # 0=off, 1=inv vol, 0.5=sqrt

    data=load_oos_plus()
    if not data: raise SystemExit("No oos_plus.csv files. Run: python tools/walk_forward_plus.py")

    merged=None
    symbols=sorted(data.keys())
    for sym in symbols:
        df=data[sym]
        merged=df if merged is None else merged.merge(df, on="date", how="outer")
    merged=merged.sort_values("date").ffill()

    pnl_cols=[f"{s}_pnl" for s in symbols if f"{s}_pnl" in merged.columns]
    pos_cols=[f"{s}_pos" for s in symbols if f"{s}_pos" in merged.columns]
    if not pnl_cols: raise SystemExit("No *_pnl columns after merge.")
    pnl_df=merged[pnl_cols].fillna(0.0)

    # base asset inverse-vol
    roll_vol=pnl_df.rolling(LOOKBACK).std()*np.sqrt(252)
    inv_vol=1.0/(roll_vol+1e-6)
    w=inv_vol.copy()
    w=w.div(w.sum(axis=1).replace(0,np.nan), axis=0).fillna(0.0)
    w=w.clip(upper=CAP_PER)
    w=w.div(w.sum(axis=1).replace(0,np.nan), axis=0).fillna(0.0)

    # hive maps + hive vol targeting
    hive_map=load_hives()  # {"HIVE1":["VIX","VIX9D",...], ...}
    if hive_map and HIVE_VOL_PENALTY>0:
        # compute hive pnl by summing member pnl equally weighted
        mems_by_hive={h:[m for m in mems if f"{m}_pnl" in pnl_df.columns] for h,mems in hive_map.items()}
        hive_pnl={}
        for h, mems in mems_by_hive.items():
            if not mems: continue
            hive_pnl[h]=pnl_df[[f"{m}_pnl" for m in mems]].mean(axis=1)
        hive_vol={}
        for h, series in hive_pnl.items():
            hv=series.rolling(HIVE_VOL_LOOKBACK).std()*np.sqrt(252)
            hive_vol[h]=hv
        # per-date scaling for each symbol based on its hive vol
        w_sym_cols=[c[:-4] for c in w.columns] if w.columns[0].endswith("_pnl")==False else [c for c in symbols]
        w.columns=[c[:-4] if c.endswith("_pnl") else c for c in w.columns]  # ensure raw symbols
        for h, mems in mems_by_hive.items():
            if not mems: continue
            hv = hive_vol.get(h, None)
            if hv is None: continue
            # scale members by inv hive vol^penalty
            scale = (1.0 / (hv + 1e-6)) ** HIVE_VOL_PENALTY
            scale = scale.clip(lower=0.0).fillna(method="ffill").fillna(0.0)
            for m in mems:
                if m in w.columns:
                    w[m] = (w[m].values * scale.values)
        # renormalize
        w = w.div(w.sum(axis=1).replace(0,np.nan), axis=0).fillna(0.0)

    # enforce HIVE_CAP
    if hive_map:
        for idx in w.index:
            row=w.loc[idx].copy()
            for h, mems in hive_map.items():
                cols=[m for m in mems if m in row.index]
                if not cols: continue
                s=float(row[cols].sum())
                if s>HIVE_CAP+1e-12 and s>0:
                    row[cols]=row[cols]*(HIVE_CAP/s)
            s=float(row.sum())
            if s>0: row=row/s
            w.loc[idx]=row

    # align w to pnl_df order
    w=w[[c[:-4] for c in pnl_cols]]

    # portfolio pnl and costs
    port_pnl_raw=(w.values * pnl_df.values).sum(axis=1)
    w_prev=w.shift(1).fillna(0.0)
    d_w=(w - w_prev).abs().sum(axis=1)
    avg_abs_pos=merged[pos_cols].abs().mean(axis=1).fillna(0.0) if pos_cols else 0.0
    costs=(float(COST_BPS)/10000.0)*d_w*avg_abs_pos
    port_pnl=pd.Series(port_pnl_raw, index=merged.index) - costs

    # equity & metrics
    port_eq=(1.0+port_pnl).cumprod()
    port_hit=float((port_pnl>0).mean())
    port_sharpe=sharpe(port_pnl)
    port_mdd=max_drawdown(port_eq)

    out_df=pd.DataFrame({"date": merged["date"], "port_ret": port_pnl, "port_equity": port_eq})
    safe_mkdir(RUNS)
    out_df.to_csv(RUNS/"portfolio_plus.csv", index=False)
    (RUNS/"portfolio_summary.json").write_text(json.dumps({
        "hit": port_hit, "sharpe": port_sharpe, "maxDD": port_mdd,
        "lookback": LOOKBACK, "cap_per_asset": CAP_PER, "cost_bps": COST_BPS,
        "hive_cap": HIVE_CAP, "hive_vol_lookback": HIVE_VOL_LOOKBACK, "hive_vol_penalty": HIVE_VOL_PENALTY
    }, indent=2))
    print("✅ Wrote runs_plus/portfolio_plus.csv and portfolio_summary.json")
    print(f"PORTFOLIO  Hit={port_hit:.3f}  Sharpe={port_sharpe:.3f}  MaxDD={port_mdd:.3f}")
    print(f"(Params) LOOKBACK={LOOKBACK}  CAP_PER={CAP_PER}  HIVE_CAP={HIVE_CAP}  COST_BPS={COST_BPS}  HIVE_VOL_LOOKBACK={HIVE_VOL_LOOKBACK}  HIVE_VOL_PENALTY={HIVE_VOL_PENALTY}")
