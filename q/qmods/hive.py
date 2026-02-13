#!/usr/bin/env python3
# qmods/hive.py
# Hive / Ecosystem layer (read-only version):
# - Reads an optional mapping data_hives/hive_map.csv with columns: asset,hive
# - Reads runs_plus/symbolic_signal.csv and runs_plus/reflexive_signal.csv
# - Builds a combined per-asset daily "meta_signal" = mean(sym_signal, reflex_signal)
# - Aggregates by HIVE -> daily hive_signal (mean across assets in hive)
# - Writes:
#     runs_plus/hive_signals.csv        (DATE, HIVE, hive_signal, sym, reflex, n_assets)
#     runs_plus/hive_summary.json       (hive list, counts, dates, top recent leaders)
#     runs_plus/hive_assets.csv         (asset -> hive mapping actually used)
#
# No changes to portfolio results yet; this is an observational card.

from pathlib import Path
import pandas as pd
import numpy as np
import json
import re

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
HIVES_DIR = ROOT / "data_hives"
MAP_CSV = HIVES_DIR / "hive_map.csv"

FX_RE = re.compile(r"^[A-Z]{6}$")

RATES_SYMS = {
    "TLT", "IEF", "SHY", "VGSH", "BND", "AGG", "GOVT", "LQD", "HYG", "MBB",
    "ZB", "ZN", "ZF", "ZT", "UB", "TY", "FV", "TU",
    "^TNX", "^IRX", "^FVX",
}
COMMOD_SYMS = {
    "GLD", "SLV", "USO", "UNG", "DBC", "DBA", "XLE", "XOP", "GDX", "XME",
    "CL", "NG", "GC", "SI", "HG", "PL", "PA", "XAUUSD", "XAGUSD",
}
CRYPTO_SYMS = {
    "BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "XRPUSD", "DOGEUSD",
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "IBIT", "FBTC", "BITO",
}

def _infer_hive_from_symbol(sym: str) -> str:
    s = str(sym or "").upper().replace("-", "").replace("_", "").replace("/", "")
    if not s:
        return "EQ"
    if s in CRYPTO_SYMS:
        return "CRYPTO"
    if s in RATES_SYMS:
        return "RATES"
    if s in COMMOD_SYMS:
        return "COMMOD"
    if FX_RE.match(s):
        c1, c2 = s[:3], s[3:]
        ccy = {"USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK"}
        if c1 in ccy and c2 in ccy:
            return "FX"
    return "EQ"

def _read_mapping():
    """Return DataFrame with columns ASSET,HIVE (uppercased)."""
    if not MAP_CSV.exists():
        # fallback: use cluster map if available
        cm = RUNS / "cluster_map.csv"
        if cm.exists():
            try:
                df = pd.read_csv(cm)
                lowers = {c.lower(): c for c in df.columns}
                a = lowers.get("asset")
                c = lowers.get("cluster")
                if a and c:
                    out = df[[a, c]].copy()
                    out.columns = ["ASSET", "HIVE"]
                    out["ASSET"] = out["ASSET"].astype(str).str.upper()
                    out["HIVE"] = out["HIVE"].astype(str).str.upper()
                    return out
            except Exception:
                pass
        return pd.DataFrame(columns=["ASSET","HIVE"])
    df = pd.read_csv(MAP_CSV)
    lowers = {c.lower(): c for c in df.columns}
    a = lowers.get("asset") or list(df.columns)[0]
    h = lowers.get("hive") or (list(df.columns)[1] if len(df.columns) > 1 else None)
    if h is None:
        # single column -> every row is a hive label? Fallback: everything to that hive
        name = str(df.iloc[0,0]).strip().upper() if len(df) else "ALL"
        return pd.DataFrame(columns=["ASSET","HIVE"]), name
    out = df[[a,h]].copy()
    out.columns = ["ASSET","HIVE"]
    out["ASSET"] = out["ASSET"].astype(str).str.upper()
    out["HIVE"] = out["HIVE"].astype(str).str.upper()
    return out

def _load_signal(path, sig_col, rename_to):
    if not path.exists():
        return pd.DataFrame(columns=["DATE","ASSET",rename_to])
    df = pd.read_csv(path, parse_dates=["DATE"])
    lowers = {c.lower(): c for c in df.columns}
    a = lowers.get("asset") or "ASSET"
    if sig_col not in df.columns:
        return pd.DataFrame(columns=["DATE","ASSET",rename_to])
    out = df[["DATE", a, sig_col]].copy()
    out.columns = ["DATE","ASSET",rename_to]
    out["ASSET"] = out["ASSET"].astype(str).str.upper()
    return out

def run_hive():
    RUNS.mkdir(parents=True, exist_ok=True)
    HIVES_DIR.mkdir(parents=True, exist_ok=True)

    # Load signals
    sym = _load_signal(RUNS/"symbolic_signal.csv", "sym_signal", "sym")
    ref = _load_signal(RUNS/"reflexive_signal.csv", "reflex_signal", "reflex")

    # Merge to a per-asset meta signal
    if sym.empty and ref.empty:
        # Still write shells
        pd.DataFrame(columns=["DATE","HIVE","hive_signal","sym","reflex","n_assets"]).to_csv(RUNS/"hive_signals.csv", index=False)
        summary = {"hives": [], "date_min": None, "date_max": None, "rows": 0, "top_recent": [], "notes":"no symbolic/reflexive data"}
        (RUNS/"hive_summary.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame(columns=["ASSET","HIVE"]).to_csv(RUNS/"hive_assets.csv", index=False)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), summary

    if sym.empty:
        aset = ref.copy(); aset["sym"] = np.nan
    elif ref.empty:
        aset = sym.copy(); aset["reflex"] = np.nan
    else:
        aset = pd.merge(sym, ref, on=["DATE","ASSET"], how="outer").sort_values(["DATE","ASSET"])

    # Combined per-asset signal: mean of available parts, bounded
    aset["meta_signal"] = aset[["sym","reflex"]].mean(axis=1)
    aset["meta_signal"] = aset["meta_signal"].clip(-1.0, 1.0)

    # Load mapping
    mapping = _read_mapping()
    if isinstance(mapping, tuple):
        # odd single-column case returned (empty, name)
        mapping = mapping[0]
    if mapping.empty:
        # If no mapping, infer hives from symbol names.
        aset["HIVE"] = aset["ASSET"].astype(str).map(_infer_hive_from_symbol).fillna("EQ").astype(str).str.upper()
    else:
        aset["ASSET_u"] = aset["ASSET"].astype(str).str.upper()
        m = mapping.copy()
        aset = pd.merge(aset, m, left_on="ASSET_u", right_on="ASSET", how="left")
        aset["HIVE"] = aset["HIVE"].fillna(aset["ASSET_u"].map(_infer_hive_from_symbol)).astype(str).str.upper()
        aset.drop(columns=["ASSET_y","ASSET_u"], errors="ignore", inplace=True)
        aset.rename(columns={"ASSET_x":"ASSET"}, inplace=True)

    # Aggregate by HIVE per day
    grp = aset.groupby(["DATE","HIVE"], as_index=False).agg(
        hive_signal=("meta_signal","mean"),
        sym=("sym","mean"),
        reflex=("reflex","mean"),
        n_assets=("ASSET","nunique"),
    ).sort_values(["DATE","HIVE"])

    # Hive health (rolling sharpe proxy) + ecosystem weights
    health_frames = []
    for h, g in grp.groupby("HIVE"):
        gg = g.sort_values("DATE").copy()
        mu = gg["hive_signal"].rolling(63, min_periods=20).mean()
        sd = gg["hive_signal"].rolling(63, min_periods=20).std(ddof=1).replace(0, np.nan)
        sharpe_proxy = (mu / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gg["hive_health"] = np.tanh(sharpe_proxy / 2.0)
        gg["hive_stability"] = (1.0 - gg["hive_signal"].rolling(21, min_periods=7).std(ddof=1).fillna(0.0)).clip(0.0, 1.0)
        health_frames.append(gg)
    grp = pd.concat(health_frames, ignore_index=True).sort_values(["DATE", "HIVE"])

    # Cross-hive arbitration weights (softmax over health)
    pivot_h = grp.pivot(index="DATE", columns="HIVE", values="hive_health").fillna(0.0)
    x = np.exp(2.5 * pivot_h.values)  # higher temperature => stronger routing
    x = x / (x.sum(axis=1, keepdims=True) + 1e-12)
    w_hive = pd.DataFrame(x, index=pivot_h.index, columns=pivot_h.columns)
    w_hive = w_hive.reset_index()
    w_hive.to_csv(RUNS/"weights_cross_hive.csv", index=False)

    grp.to_csv(RUNS/"hive_signals.csv", index=False)

    # Summary
    if grp.empty:
        summary = {"hives": [], "date_min": None, "date_max": None, "rows": 0, "top_recent": [], "notes":"no hive rows"}
    else:
        date_min = str(grp["DATE"].min().date())
        date_max = str(grp["DATE"].max().date())
        # rank hives on last 252d median signal
        recent = grp[grp["DATE"] >= (grp["DATE"].max() - pd.Timedelta(days=252))]
        top_recent = []
        if not recent.empty:
            for h, g in recent.groupby("HIVE"):
                last_date = g["DATE"].max()
                try:
                    suggested = float(
                        w_hive.loc[w_hive["DATE"] == last_date, h].iloc[0]
                    )
                except Exception:
                    suggested = 0.0
                top_recent.append({
                    "HIVE": h,
                    "median_signal": float(g["hive_signal"].median()),
                    "last_signal": float(g["hive_signal"].iloc[-1]),
                    "avg_members": float(g["n_assets"].mean()),
                    "median_health": float(g["hive_health"].median()),
                    "suggested_weight": suggested,
                    "atrophy_flag": bool(float(g["hive_health"].median()) < -0.15),
                })
            top_recent = sorted(top_recent, key=lambda x: x["median_signal"], reverse=True)[:10]
        summary = {
            "hives": sorted(list(grp["HIVE"].unique())),
            "rows": int(len(grp)),
            "date_min": date_min,
            "date_max": date_max,
            "top_recent": top_recent,
            "cross_hive_weights_file": "runs_plus/weights_cross_hive.csv",
        }

    # Save back the mapping actually used
    used_map = (mapping if not mapping.empty else pd.DataFrame(columns=["ASSET","HIVE"]))
    used_map.to_csv(RUNS/"hive_assets.csv", index=False)

    (RUNS/"hive_summary.json").write_text(json.dumps(summary, indent=2))
    return mapping, aset, grp, summary

if __name__ == "__main__":
    mapping, aset, grp, summary = run_hive()
    print("Hive rows:", len(grp), "| Hives:", len(summary.get("hives", [])))
    print("Span:", summary.get("date_min"), "→", summary.get("date_max"))
    print("Top recent:", ", ".join([t["HIVE"] for t in summary.get("top_recent", [])]) or "(none)")
