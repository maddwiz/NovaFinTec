#!/usr/bin/env python3
# tools/portfolio_cluster_caps.py
# Build a portfolio using your CURRENT selected assets (from runs_plus/portfolio_weights.csv),
# but enforce correlation CLUSTER_CAP on group totals (plus per-asset CAP_PER).
# Outputs:
#   runs_plus/portfolio_plus_cluster.csv
#   runs_plus/final_portfolio_cluster_summary.json
#   runs_plus/portfolio_weights_cluster.csv

from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs_plus"
RUNS.mkdir(parents=True, exist_ok=True)

# ---- knobs ----
CAP_PER      = 0.10   # max 10% per asset (same spirit as your main)
CLUSTER_CAP  = 0.20   # max 20% per correlation cluster
CORR_THRESH  = 0.60   # assets with |corr| >= 0.60 are connected in same cluster
DATE_COL     = "Date"
RET_COL      = "Close"
IS_FRAC      = 0.75   # same IS/OOS split as your main summary
# ---------------

def _ann_sharpe(r):
    s = pd.Series(r).replace([np.inf,-np.inf], np.nan).dropna()
    if s.empty: return 0.0
    sd = s.std()
    if not np.isfinite(sd) or sd == 0: return 0.0
    return float((s.mean()/sd)*np.sqrt(252.0))
def _maxdd(r):
    s = pd.Series(r).fillna(0.0)
    eq = (1.0 + s).cumprod()
    dd = eq/eq.cummax() - 1.0
    return float(dd.min())
def _hit(r):
    s = pd.Series(r).dropna()
    return float((s>0).mean()) if not s.empty else 0.0
def _split_is_oos(vals, frac=IS_FRAC):
    n = len(vals); k = int(n * frac)
    return vals[:k], vals[k:]

def _load_close_returns(asset):
    p = DATA / f"{asset}.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None

    # date column
    if DATE_COL not in df.columns:
        for c in df.columns:
            if str(c).lower() in ("date","timestamp"):
                df[DATE_COL] = df[c]; break
    if DATE_COL not in df.columns:
        return None

    # pick price column
    if RET_COL not in df.columns:
        for c in ["Adj Close","Adjusted Close","close","adj_close","Close*","Price","Last"]:
            if c in df.columns:
                df[RET_COL] = df[c]; break
    if RET_COL not in df.columns:
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if nums:
            df[RET_COL] = df[nums[-1]]
        else:
            return None

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)
    px = pd.to_numeric(df[RET_COL], errors="coerce")
    ret = px.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
    return pd.DataFrame({"Date": df[DATE_COL], asset: ret.values})

def _connected_components(corr, assets, thresh=CORR_THRESH):
    # Build undirected graph by |corr| >= thresh; return list of clusters (lists of asset names)
    n = len(assets)
    g = {i:set() for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr[i,j]) >= thresh:
                g[i].add(j); g[j].add(i)
    seen = set()
    clusters = []
    for i in range(n):
        if i in seen: continue
        stack = [i]; comp = []
        while stack:
            u = stack.pop()
            if u in seen: continue
            seen.add(u); comp.append(u)
            for v in g[u]:
                if v not in seen:
                    stack.append(v)
        clusters.append([assets[k] for k in comp])
    return clusters

if __name__ == "__main__":
    # 1) use your current chosen assets from weights file (keeps Top-K etc.)
    wpath = RUNS / "portfolio_weights.csv"
    if not wpath.exists():
        raise SystemExit("Missing runs_plus/portfolio_weights.csv. Build main portfolio first.")
    wdf = pd.read_csv(wpath)
    assets = [a for a in wdf["asset"].tolist() if isinstance(a, str) and a.strip()]
    if not assets:
        raise SystemExit("No assets found in portfolio_weights.csv.")

    # 2) load returns panel
    panel = None
    ok = []
    for a in assets:
        r = _load_close_returns(a)
        if r is None or r.empty: 
            continue
        panel = r if panel is None else panel.merge(r, on="Date", how="outer")
        ok.append(a)
    if panel is None or not ok:
        raise SystemExit("Could not load returns for any assets in portfolio_weights.csv.")
    panel = panel.sort_values("Date").fillna(0.0).reset_index(drop=True)
    assets = ok
    R = panel[assets].values

    # 3) correlation + clusters
    if R.shape[0] < 10:
        raise SystemExit("Not enough rows to compute correlations.")
    corr = np.corrcoef(R.T)
    clusters = _connected_components(corr, assets, CORR_THRESH)

    # 4) start from equal weights → per-asset cap → renorm
    n = len(assets)
    w = np.ones(n) / max(n, 1)
    w = np.minimum(w, CAP_PER)
    w = w / w.sum()
    weights = pd.Series(w, index=assets)

    # 5) enforce cluster caps: if cluster sum > CLUSTER_CAP, scale that cluster down
    for cl in clusters:
        s = weights[cl].sum()
        if s > CLUSTER_CAP:
            scale = CLUSTER_CAP / s
            weights.loc[cl] = weights.loc[cl] * scale
    # renormalize to 1
    weights = weights / weights.sum()

    # 6) portfolio returns + summary
    port_ret = (panel[assets] * weights.values).sum(axis=1)
    eq = (1.0 + port_ret).cumprod()
    r = port_ret.values
    r_is, r_oos = _split_is_oos(r, IS_FRAC)
    summary = {
        "in_sample":  {"sharpe": _ann_sharpe(r_is),  "hit": _hit(r_is),  "maxdd": _maxdd(r_is)},
        "out_sample": {"sharpe": _ann_sharpe(r_oos), "hit": _hit(r_oos), "maxdd": _maxdd(r_oos)},
        "n_assets": int(len(assets)),
        "cap_per": CAP_PER,
        "cluster_cap": CLUSTER_CAP,
        "corr_thresh": CORR_THRESH,
        "note": "Cluster-capped portfolio built from current selection; per-asset cap then per-cluster cap."
    }

    # 7) save outputs
    pd.DataFrame({"Date": panel["Date"], "ret": port_ret, "eq": eq}).to_csv(RUNS/"portfolio_plus_cluster.csv", index=False)
    (RUNS/"final_portfolio_cluster_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({"asset": assets, "weight": weights.values}).to_csv(RUNS/"portfolio_weights_cluster.csv", index=False)

    print("CLUSTERS:", clusters)
    print("OOS Sharpe=%.3f Hit=%.2f MaxDD=%.3f" % (summary["out_sample"]["sharpe"], summary["out_sample"]["hit"], summary["out_sample"]["maxdd"]))
    print("Wrote: portfolio_plus_cluster.csv, portfolio_weights_cluster.csv, final_portfolio_cluster_summary.json")
