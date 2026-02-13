#!/usr/bin/env python3
# tools/sweep_cluster_caps.py
# Sweeps CLUSTER_CAP and CORR_THRESH around safe ranges, builds cluster-capped portfolios,
# compares OOS Sharpe vs Main. Writes runs_plus/sweep_cluster_caps.csv (best at top).

from pathlib import Path
import json, pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"
DATA = ROOT/"data"

# ---- grids (small, safe) ----
CAP_PER      = 0.10               # keep same per-asset cap
CLUSTER_CAPS = [0.16, 0.18, 0.20, 0.22, 0.24]
CORR_THRS    = [0.55, 0.60, 0.65, 0.70]
DATE_COL     = "Date"
RET_COL      = "Close"
IS_FRAC      = 0.75
# --------------------------------

def _ann_sharpe(r):
    s = pd.Series(r).replace([np.inf,-np.inf], np.nan).dropna()
    if s.empty: return 0.0
    sd = s.std()
    return 0.0 if not np.isfinite(sd) or sd==0 else float((s.mean()/sd)*np.sqrt(252))

def _maxdd(r):
    s = pd.Series(r).fillna(0.0)
    eq = (1.0 + s).cumprod()
    dd = eq/eq.cummax() - 1.0
    return float(dd.min())

def _split(r, frac):
    n = len(r); k = int(n*frac)
    return r[:k], r[k:]

def _load_close_returns(asset):
    p = DATA / f"{asset}.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    dcol = DATE_COL if DATE_COL in df.columns else next((c for c in df.columns if str(c).lower() in ("date","timestamp")), None)
    if dcol is None: return None
    pxcol = RET_COL if RET_COL in df.columns else next((c for c in ["Adj Close","Adjusted Close","close","adj_close","Close*","Price","Last"] if c in df.columns), None)
    if pxcol is None:
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not nums: return None
        pxcol = nums[-1]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol)
    px = pd.to_numeric(df[pxcol], errors="coerce")
    ret = px.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
    return pd.DataFrame({"Date": df[dcol], asset: ret.values})

def _clusters(corr, assets, thr):
    n = len(assets)
    g = {i:set() for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr[i,j]) >= thr:
                g[i].add(j); g[j].add(i)
    seen=set(); out=[]
    for i in range(n):
        if i in seen: continue
        stack=[i]; comp=[]
        while stack:
            u=stack.pop()
            if u in seen: continue
            seen.add(u); comp.append(u)
            for v in g[u]:
                if v not in seen: stack.append(v)
        out.append([assets[k] for k in comp])
    return out

if __name__ == "__main__":
    # main baseline
    main = json.loads((RUNS/"final_portfolio_summary.json").read_text())
    main_oos = float(main["out_sample"]["sharpe"])

    # assets per current portfolio
    w = pd.read_csv(RUNS/"portfolio_weights.csv")
    assets = [a for a in w["asset"].tolist() if isinstance(a,str) and a.strip()]
    # build returns panel
    panel=None; ok=[]
    for a in assets:
        r = _load_close_returns(a)
        if r is None or r.empty: continue
        panel = r if panel is None else panel.merge(r, on="Date", how="outer")
        ok.append(a)
    if panel is None:
        raise SystemExit("No asset returns loaded.")
    panel = panel.sort_values("Date").fillna(0.0)
    assets = ok
    R = panel[assets].values
    corr = np.corrcoef(R.T) if len(assets) >= 2 else np.eye(len(assets))

    rows=[]
    for cc in CLUSTER_CAPS:
        for thr in CORR_THRS:
            # clusters
            cl = _clusters(corr, assets, thr)
            # start equal → per-asset cap
            n=len(assets); wgt=np.ones(n)/max(1,n)
            wgt = np.minimum(wgt, CAP_PER); wgt = wgt/wgt.sum()
            s = pd.Series(wgt, index=assets)

            # apply cluster cap
            for group in cl:
                tot = s[group].sum()
                if tot > cc:
                    s[group] = s[group]*(cc/tot)
            s = s/s.sum()

            port = (panel[assets]*s.values).sum(axis=1).values
            is_, oos = _split(port, IS_FRAC)
            rows.append(dict(
                cluster_cap=cc, corr_thresh=thr,
                oos_sharpe=_ann_sharpe(oos),
                oos_maxdd=_maxdd(oos),
                main_oos=main_oos,
                n_assets=len(assets),
                clusters=len(cl)
            ))

    df = pd.DataFrame(rows).sort_values(["oos_sharpe"], ascending=False)
    out = RUNS/"sweep_cluster_caps.csv"
    df.to_csv(out, index=False)
    print(df.head(12).to_string(index=False))
    print("Saved:", out)
