#!/usr/bin/env python3
# qmods/cross_overlay.py
# Cross-Domain Dream Overlays:
# - Reads finance dream latents from runs_plus/dreams_latents.csv (DATE, ASSET, L1..)
# - Reads external latents from data_cross/*.csv (DATE, DOMAIN, X1..)
# - Builds a daily 1D latent index for Finance and each external DOMAIN
# - Computes rolling correlations (default 126d) between Finance and each DOMAIN
# - Writes CSV + JSON summaries for report card

from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
CROSS_DIR = ROOT / "data_cross"

ROLL_WIN = 126  # ~6 months

def _safe_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf], np.nan)

def _load_finance_latents():
    p = RUNS / "dreams_latents.csv"
    if not p.exists():
        return pd.DataFrame(columns=["DATE","FINANCE_IDX"])
    df = pd.read_csv(p)
    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or lowers.get("timestamp") or df.columns[0]
    acol = lowers.get("asset") or "ASSET"
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).rename(columns={dcol:"DATE", acol:"ASSET"})
    lat_cols = [c for c in df.columns if c not in ["DATE","ASSET"] and pd.api.types.is_numeric_dtype(df[c])]
    if not lat_cols:
        return pd.DataFrame(columns=["DATE","FINANCE_IDX"])
    # mean of z-scored latents across dims and assets per day -> 1D index
    daily = []
    for dt, g in df.groupby("DATE"):
        Zs = []
        for c in lat_cols:
            s = _safe_num(g[c])
            mu, sd = float(np.nanmean(s)), float(np.nanstd(s))
            Zs.append((s - (mu if np.isfinite(mu) else 0.0)) / (sd if (np.isfinite(sd) and sd!=0) else 1.0))
        Z = np.nanmean(np.vstack([z.values for z in Zs]), axis=0)
        daily.append({"DATE": pd.to_datetime(dt).normalize(), "FINANCE_IDX": float(np.nanmean(Z))})
    out = pd.DataFrame(daily).sort_values("DATE")
    return out

def _load_external_domains():
    """Load external latent files from data_cross/*.csv
       Expected columns: DATE, DOMAIN, plus any numeric latent columns (X1..).
    """
    rows = []
    if not CROSS_DIR.exists():
        return pd.DataFrame(columns=["DATE","DOMAIN","VAL"])
    for p in CROSS_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(p)
            lowers = {c.lower(): c for c in df.columns}
            dcol = lowers.get("date") or lowers.get("timestamp") or df.columns[0]
            dom  = lowers.get("domain")
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
            df = df.dropna(subset=[dcol]).rename(columns={dcol:"DATE"})
            if dom is None or dom not in df.columns:
                # infer domain name from filename
                domain_name = p.stem.upper()
                df["DOMAIN"] = domain_name
            else:
                df["DOMAIN"] = df[dom].astype(str).str.upper()
            num_cols = [c for c in df.columns if c not in ["DATE","DOMAIN"] and pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols: 
                continue
            # compress row to mean of z-scored numeric cols
            for dt, g in df.groupby("DATE"):
                Zs = []
                for c in num_cols:
                    s = _safe_num(g[c])
                    mu, sd = float(np.nanmean(s)), float(np.nanstd(s))
                    Zs.append((s - (mu if np.isfinite(mu) else 0.0)) / (sd if (np.isfinite(sd) and sd!=0) else 1.0))
                Z = float(np.nanmean(np.hstack([z.values for z in Zs])))
                doms = list(map(str, g["DOMAIN"].unique()))
                # if multiple domain labels in one file, output each
                for dname in doms:
                    rows.append({"DATE": pd.to_datetime(dt).normalize(), "DOMAIN": dname, "VAL": Z})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["DATE","DOMAIN","VAL"])
    out = pd.DataFrame(rows).sort_values(["DOMAIN","DATE"])
    # average per day/domain if duplicates
    out = out.groupby(["DATE","DOMAIN"], as_index=False)["VAL"].mean()
    return out

def _rolling_corr(a: pd.Series, b: pd.Series, win=ROLL_WIN):
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if df.empty: return pd.Series([], dtype=float)
    return df["a"].rolling(win, min_periods=max(20, win//4)).corr(df["b"])

def run_cross_overlay():
    RUNS.mkdir(parents=True, exist_ok=True)
    fin = _load_finance_latents()
    ext = _load_external_domains()
    # save raw 1D series
    fin.to_csv(RUNS/"cross_finance_idx.csv", index=False)
    ext.to_csv(RUNS/"cross_external_idx.csv", index=False)

    if fin.empty or ext.empty:
        # write empty results but valid shells
        pd.DataFrame(columns=["DATE","DOMAIN","corr"]).to_csv(RUNS/"cross_overlay.csv", index=False)
        (RUNS/"cross_overlay_summary.json").write_text(json.dumps({
            "domains": [], "rows": 0, "win": ROLL_WIN
        }, indent=2))
        return fin, ext, pd.DataFrame(), {"domains": [], "rows": 0, "win": ROLL_WIN}

    # align dates
    fin = fin.dropna().sort_values("DATE")
    out_rows = []
    for dom, g in ext.groupby("DOMAIN"):
        g = g.dropna().sort_values("DATE")
        df = fin.merge(g, on="DATE", how="inner")
        if df.empty:
            continue
        corr = _rolling_corr(df["FINANCE_IDX"], df["VAL"], win=ROLL_WIN)
        df2 = pd.DataFrame({"DATE": df["DATE"], "DOMAIN": dom, "corr": corr})
        out_rows.append(df2)
    res = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=["DATE","DOMAIN","corr"])
    res = res.sort_values(["DOMAIN","DATE"]).dropna()
    res.to_csv(RUNS/"cross_overlay.csv", index=False)

    # summarize strongest recent links (last 252d)
    recent = res[res["DATE"] >= (res["DATE"].max() - pd.Timedelta(days=252))] if not res.empty else res
    tops = []
    if not recent.empty:
        for dom, gg in recent.groupby("DOMAIN"):
            tops.append({"DOMAIN": dom, "corr_median": float(gg["corr"].median()), "corr_last": float(gg["corr"].iloc[-1])})
        tops = sorted(tops, key=lambda x: x["corr_median"], reverse=True)[:10]

    meta = {"domains": sorted(list(res["DOMAIN"].unique())) if not res.empty else [],
            "rows": int(len(res)),
            "win": ROLL_WIN,
            "top_recent": tops}
    (RUNS/"cross_overlay_summary.json").write_text(json.dumps(meta, indent=2))
    return fin, ext, res, meta

if __name__ == "__main__":
    fin, ext, res, meta = run_cross_overlay()
    print("Finance idx rows:", len(fin), "| External rows:", len(ext), "| Corr rows:", len(res))
    print("Domains:", ", ".join(meta.get("domains", [])) or "(none)")
