#!/usr/bin/env python3
# tools/portfolio_from_runs_plus.py
# Auto-build a portfolio from WF results:
# - reads runs_plus/walk_forward_table.csv (asset, sharpe, hit, maxDD)
# - picks TOP_K assets by Sharpe (fallback: at least MIN_KEEP assets)
# - loads data/<ASSET>.csv, uses Close to compute daily returns
# - equal weight, cap per-asset at CAP_PER, normalize, build portfolio
# - writes runs_plus/portfolio_plus.csv and runs_plus/final_portfolio_summary.json
#
# NOTE: This replaces any old whitelist behavior that limited you to 2 assets.

from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs_plus"
RUNS.mkdir(parents=True, exist_ok=True)

# ---- knobs you can safely change ----
TOP_K    = 34     # how many best Sharpe assets to include
MIN_KEEP = 10     # if fewer than this have Sharpe>0, include the top MIN_KEEP anyway
CAP_PER  = 0.08   # max 10% per asset
RET_COL  = "Close"
DATE_COL = "Date"
# -------------------------------------

def _ann_sharpe(r):
    s = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty: return 0.0
    sd = s.std()
    if not np.isfinite(sd) or sd == 0: return 0.0
    return float((s.mean() / sd) * np.sqrt(252.0))

def _maxdd(r):
    s = pd.Series(r).fillna(0.0)
    eq = (1.0 + s).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())

def _hit(r):
    s = pd.Series(r).dropna()
    return float((s > 0).mean()) if not s.empty else 0.0

def _split_is_oos(vals, frac=0.75):
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
    if DATE_COL not in df.columns:
        # try common alternates
        for c in df.columns:
            if c.lower() in ("date", "timestamp"):
                df[DATE_COL] = df[c]; break
    if DATE_COL not in df.columns:
        return None
    if RET_COL not in df.columns:
        # try to find any price-like column
        for c in ["Adj Close","Adjusted Close","close","adj_close","Close*","Price","Last"]:
            if c in df.columns:
                df[RET_COL] = df[c]; break
    if RET_COL not in df.columns:
        # last resort: right-most numeric
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if nums:
            df[RET_COL] = df[nums[-1]]
        else:
            return None

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)
    px = pd.to_numeric(df[RET_COL], errors="coerce")
    ret = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.5, 0.5)
    out = pd.DataFrame({"Date": df[DATE_COL], asset: ret.values})
    return out

if __name__ == "__main__":
    # 1) read WF table
    wfp = RUNS / "walk_forward_table.csv"
    if not wfp.exists():
        raise SystemExit("Missing runs_plus/walk_forward_table.csv. Run tools/walk_forward_batch.py first.")
    wf = pd.read_csv(wfp)
    # normalize column names
    cols = {c.lower(): c for c in wf.columns}
    asset_col  = cols.get("asset") or "asset"
    sharpe_col = cols.get("sharpe") or "sharpe"

    if asset_col not in wf.columns or sharpe_col not in wf.columns:
        raise SystemExit("walk_forward_table.csv must have 'asset' and 'sharpe' columns.")

    wf = wf[[asset_col, sharpe_col]].rename(columns={asset_col: "asset", sharpe_col: "sharpe"})
    wf["sharpe"] = pd.to_numeric(wf["sharpe"], errors="coerce")
    wf = wf.dropna(subset=["asset"]).reset_index(drop=True)

    # 2) pick assets
    pos = wf[wf["sharpe"] > 0].sort_values("sharpe", ascending=False)
    if len(pos) >= MIN_KEEP:
        chosen = pos.head(TOP_K)["asset"].tolist()
    else:
        chosen = wf.sort_values("sharpe", ascending=False).head(max(TOP_K, MIN_KEEP))["asset"].tolist()

    if not chosen:
        raise SystemExit("No assets selected from WF table.")

    # 3) load each asset's returns from data/*.csv
    panel = None
    ok_assets = []
    for a in chosen:
        df = _load_close_returns(a)
        if df is None or df.empty:
            continue
        panel = df if panel is None else panel.merge(df, on="Date", how="outer")
        ok_assets.append(a)

    if panel is None or not ok_assets:
        raise SystemExit("Could not build any return series from data/*.csv")

    panel = panel.sort_values("Date").fillna(0.0).reset_index(drop=True)

    # 4) weights: equal → cap → renorm
    n = len(ok_assets)
    w = np.ones(n) / max(n, 1)
    w = np.minimum(w, CAP_PER)
    w = w / w.sum()

    weights = pd.Series(w, index=ok_assets)

    # 5) portfolio returns
    rets = panel[ok_assets]
    port_ret = (rets * weights.values).sum(axis=1)
    eq = (1.0 + port_ret).cumprod()

    # 6) metrics
    r = port_ret.values
    r_is, r_oos = _split_is_oos(r, 0.75)
    summary = {
        "in_sample":  {"sharpe": _ann_sharpe(r_is),  "hit": _hit(r_is),  "maxdd": _maxdd(r_is)},
        "out_sample": {"sharpe": _ann_sharpe(r_oos), "hit": _hit(r_oos), "maxdd": _maxdd(r_oos)},
        "n_assets": int(n),
        "cap_per": CAP_PER,
        "note": "Auto portfolio from WF table, equal-weight capped."
    }

    # 7) save outputs
    out = pd.DataFrame({"Date": panel["Date"], "ret": port_ret, "eq": eq})
    out.to_csv(RUNS / "portfolio_plus.csv", index=False)
    (RUNS / "final_portfolio_summary.json").write_text(json.dumps(summary, indent=2))
    # also write per-asset weights table
    pd.DataFrame({"asset": ok_assets, "weight": weights.values}).to_csv(RUNS / "portfolio_weights.csv", index=False)

    print(f"PORTFOLIO n={n} cap={CAP_PER:.2f}")
    print("IS  Sharpe=%.3f Hit=%.2f MaxDD=%.2f" % (summary["in_sample"]["sharpe"], summary["in_sample"]["hit"], summary["in_sample"]["maxdd"]))
    print("OOS Sharpe=%.3f Hit=%.2f MaxDD=%.2f" % (summary["out_sample"]["sharpe"], summary["out_sample"]["hit"], summary["out_sample"]["maxdd"]))
    print("Wrote: runs_plus/portfolio_plus.csv, final_portfolio_summary.json, portfolio_weights.csv")
