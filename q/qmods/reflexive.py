#!/usr/bin/env python3
# qmods/reflexive.py
# Reflexive Feedback: take Dream latents (if available), compress to a 1D daily signal per asset,
# normalize, and emit a bounded "reflex_signal" that Q can use as an add-on sleeve.
#
# Looks for (any of the following, first match wins):
#   runs_plus/dreams_latents.csv   columns: DATE, ASSET, L1, L2, ... LN
#   runs_plus/dreams.csv           columns: DATE, ASSET, latent (stringified), or any numeric cols
#
# If nothing is found, it still writes empty-but-valid outputs so the pipeline and report won’t break.
#
# Outputs:
#   runs_plus/reflexive_events.csv   (per-row latent + compression preview)
#   runs_plus/reflexive_signal.csv   (DATE, ASSET, reflex_signal, reflex_raw)
#   runs_plus/reflexive_summary.json (counts, span)

from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
PORTF = RUNS / "portfolio_plus.csv"

def _safe_num(s):
    x = pd.to_numeric(s, errors="coerce")
    return x.replace([np.inf,-np.inf], np.nan)

def _load_latents():
    # Preferred: dreams_latents.csv
    p1 = RUNS / "dreams_latents.csv"
    if p1.exists():
        df = pd.read_csv(p1)
        lowers = {c.lower(): c for c in df.columns}
        dcol = lowers.get("date") or lowers.get("timestamp") or df.columns[0]
        acol = lowers.get("asset") or "ASSET"
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol]).rename(columns={dcol:"DATE", acol:"ASSET"})
        # numeric latent columns only
        lat_cols = [c for c in df.columns if c not in ["DATE","ASSET"]]
        lat_cols = [c for c in lat_cols if pd.api.types.is_numeric_dtype(df[c])]
        if lat_cols:
            return df[["DATE","ASSET"] + lat_cols].sort_values(["DATE","ASSET"])
    # Fallback: dreams.csv
    p2 = RUNS / "dreams.csv"
    if p2.exists():
        df = pd.read_csv(p2)
        lowers = {c.lower(): c for c in df.columns}
        dcol = lowers.get("date") or lowers.get("timestamp") or df.columns[0]
        acol = lowers.get("asset") or "ASSET"
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol]).rename(columns={dcol:"DATE", acol:"ASSET"})
        # try to find numeric columns to act as latents
        lat_cols = [c for c in df.columns if c not in ["DATE","ASSET","text","headline","title"]]
        lat_cols = [c for c in lat_cols if pd.api.types.is_numeric_dtype(df[c])]
        if lat_cols:
            return df[["DATE","ASSET"] + lat_cols].sort_values(["DATE","ASSET"])
    # Nothing found
    return pd.DataFrame(columns=["DATE","ASSET"])

def _compress_latents_to_1d(df):
    """Row-wise mean of z-scored latents per asset/day (robust, no sklearn)."""
    if df.empty:
        return pd.DataFrame(columns=["DATE","ASSET","reflex_raw"])
    lat_cols = [c for c in df.columns if c not in ["DATE","ASSET"]]
    if not lat_cols:
        return pd.DataFrame(columns=["DATE","ASSET","reflex_raw"])
    out = []
    for (dt, asset), g in df.groupby(["DATE","ASSET"]):
        sub = g[lat_cols]
        # z-score each column using expanding stats for stability
        z = []
        # use simple per-column z with nan-safe fallback
        for c in lat_cols:
            s = _safe_num(sub[c])
            mu = np.nanmean(s.values)
            sd = np.nanstd(s.values)
            if not np.isfinite(sd) or sd == 0: 
                zc = (s.values - (mu if np.isfinite(mu) else 0.0))
            else:
                zc = (s.values - mu)/sd
            z.append(zc)
        Z = np.vstack(z)
        # 1D score = mean across latent dimensions, then mean across rows
        raw = float(np.nanmean(Z))
        out.append({"DATE": pd.to_datetime(dt).normalize(), "ASSET": str(asset), "reflex_raw": raw})
    return pd.DataFrame(out)

def _rolling_tanh_norm(s, win=63):
    s = pd.Series(pd.to_numeric(s, errors="coerce")).replace([np.inf,-np.inf], np.nan)
    m = s.rolling(win, min_periods=max(5, win//3)).mean()
    v = s.rolling(win, min_periods=max(5, win//3)).std()
    z = (s - m) / v.replace(0, np.nan)
    z = z.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return np.tanh(z)


def _load_market_feedback():
    if not PORTF.exists():
        return pd.DataFrame(columns=["DATE", "feedback"])
    df = pd.read_csv(PORTF)
    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or lowers.get("timestamp")
    if dcol is None or dcol not in df.columns:
        return pd.DataFrame(columns=["DATE", "feedback"])
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).rename(columns={dcol: "DATE"})
    ret_col = None
    for c in ["ret_net", "ret", "ret_plus", "daily_ret", "portfolio_ret", "port_ret"]:
        if c in df.columns:
            ret_col = c
            break
    if ret_col is None:
        return pd.DataFrame(columns=["DATE", "feedback"])
    r = pd.to_numeric(df[ret_col], errors="coerce").fillna(0.0).clip(-0.5, 0.5)
    fb = np.tanh((r.rolling(5, min_periods=2).mean() / (r.rolling(20, min_periods=5).std().replace(0, np.nan))).fillna(0.0))
    return pd.DataFrame({"DATE": df["DATE"].dt.normalize(), "feedback": fb})

def run_reflexive():
    RUNS.mkdir(parents=True, exist_ok=True)
    lat = _load_latents()
    if lat.empty:
        # write empty shells so the rest of the pipeline is happy
        pd.DataFrame(columns=["DATE","ASSET","reflex_raw"]).to_csv(RUNS/"reflexive_events.csv", index=False)
        pd.DataFrame(columns=["DATE","ASSET","reflex_signal","reflex_raw"]).to_csv(RUNS/"reflexive_signal.csv", index=False)
        info = {"rows":0,"assets":[],"date_min":None,"date_max":None}
        (RUNS/"reflexive_summary.json").write_text(json.dumps(info, indent=2))
        return lat, pd.DataFrame(), info

    # compress per (DATE, ASSET)
    ev = _compress_latents_to_1d(lat)
    ev = ev.sort_values(["DATE","ASSET"])
    ev.to_csv(RUNS/"reflexive_events.csv", index=False)

    # rolling standardization → bounded signal
    market_fb = _load_market_feedback()
    sigs = []
    for asset, g in ev.groupby("ASSET"):
        g = g.sort_values("DATE").copy()
        g["reflex_base"] = _rolling_tanh_norm(g["reflex_raw"])
        if not market_fb.empty:
            g = g.merge(market_fb, on="DATE", how="left")
            g["feedback"] = pd.to_numeric(g["feedback"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
        else:
            g["feedback"] = 0.0
        # Reflexive loop: latent state + market feedback interaction
        g["reflex_signal"] = np.tanh(g["reflex_base"] + 0.35 * g["feedback"] * np.abs(g["reflex_base"]))
        g["reflex_confidence"] = np.clip(np.abs(g["reflex_signal"]), 0.0, 1.0)
        sigs.append(g[["DATE","ASSET","reflex_signal","reflex_raw","reflex_base","feedback","reflex_confidence"]])
    sig = pd.concat(sigs, ignore_index=True).sort_values(["DATE","ASSET"])
    sig.to_csv(RUNS/"reflexive_signal.csv", index=False)

    info = {
        "rows": int(len(ev)),
        "assets": sorted(list(map(str, ev["ASSET"].unique()))),
        "date_min": str(ev["DATE"].min().date()),
        "date_max": str(ev["DATE"].max().date())
    }
    (RUNS/"reflexive_summary.json").write_text(json.dumps(info, indent=2))
    return ev, sig, info

if __name__ == "__main__":
    ev, sig, info = run_reflexive()
    print("Reflexive rows:", len(ev), "| days×assets:", len(sig))
    print("Span:", info.get("date_min"), "→", info.get("date_max"))
    print("Assets:", ", ".join(info.get("assets", [])) or "(none)")
