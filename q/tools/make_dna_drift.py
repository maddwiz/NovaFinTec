#!/usr/bin/env python3
# tools/make_dna_drift.py
# Standalone DNA drift builder (no qmods import).
# It reads portfolio_plus + optional symbolic/reflexive, builds features,
# and computes a drift score over time. Writes:
#   runs_plus/dna_drift.csv
#   runs_plus/dna_summary.json

from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
DATA = ROOT / "data"

PORT = RUNS / "portfolio_plus.csv"
SYMP = RUNS / "symbolic_signal.csv"
RFLX = RUNS / "reflexive_signal.csv"

def _safe_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf], np.nan)

def _load_portfolio():
    if not PORT.exists():
        raise SystemExit("Missing runs_plus/portfolio_plus.csv")
    df = pd.read_csv(PORT)
    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or lowers.get("timestamp") or df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).rename(columns={dcol: "DATE"}).sort_values("DATE")
    # prefer returns
    for c in ["ret_net","ret","ret_plus","ret_gross","return","pnl","pnl_plus","daily_ret","portfolio_ret","port_ret"]:
        if c in df.columns:
            r = _safe_num(df[c]).fillna(0.0).clip(-0.5,0.5)
            return pd.DataFrame({"DATE": df["DATE"], "ret": r})
    # fallback equity -> returns
    for c in ["eq_net","eq","equity","equity_curve","portfolio_eq","equity_index","port_equity"]:
        if c in df.columns:
            eq = _safe_num(df[c])
            r = eq.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
            return pd.DataFrame({"DATE": df["DATE"], "ret": r})
    raise SystemExit("Could not find returns/equity in portfolio_plus.csv")

def _load_signal(path, want_col):
    if not path.exists():
        return pd.DataFrame(columns=["DATE", want_col])
    df = pd.read_csv(path)
    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or "DATE"
    if dcol not in df.columns:
        return pd.DataFrame(columns=["DATE", want_col])
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).rename(columns={dcol:"DATE"})
    col = want_col if want_col in df.columns else None
    if col is None:
        # try to guess
        for c in df.columns:
            lc = c.lower()
            if want_col.startswith("sym") and ("sym" in lc or "symbolic" in lc):
                col = c; break
            if want_col.startswith("reflex") and "reflex" in lc:
                col = c; break
    if col is None:
        return pd.DataFrame(columns=["DATE", want_col])
    if "ASSET" in df.columns and df["ASSET"].nunique() > 1:
        g = df.groupby("DATE", as_index=False)[col].mean()
        return g.rename(columns={col: want_col})
    return df[["DATE", col]].rename(columns={col: want_col})

def _trend_strength(price_series: pd.Series, win=63) -> pd.Series:
    s = pd.Series(price_series.values, index=price_series.index)
    def _rho(x):
        y = pd.Series(x)
        z = pd.Series(np.arange(len(y)))
        if y.std(ddof=0) == 0 or z.std(ddof=0) == 0:
            return 0.0
        return float(np.corrcoef(y, z)[0,1])
    out = s.rolling(win, min_periods=max(5, win//3)).apply(lambda x: _rho(x), raw=True)
    return out.fillna(0.0).abs()

def _choppiness(ret_series: pd.Series, win=21) -> pd.Series:
    sgn = np.sign(pd.Series(ret_series).fillna(0.0))
    def _fliprate(x):
        y = pd.Series(x)
        flips = (np.sign(y).diff() != 0).sum()
        denom = max(len(y) - 1, 1)
        return float(flips) / float(denom)
    return sgn.rolling(win, min_periods=max(5, win//3)).apply(lambda x: _fliprate(x), raw=True).fillna(0.0)

def _cosine_dist(a, b):
    # a,b are 1D numpy arrays
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    return 1.0 - np.clip(cos, -1.0, 1.0)

if __name__ == "__main__":
    RUNS.mkdir(parents=True, exist_ok=True)

    main = _load_portfolio()                      # DATE, ret
    sym  = _load_signal(SYMP, "sym_signal")       # DATE, sym_signal
    rfx  = _load_signal(RFLX, "reflexive_signal") # DATE, reflexive_signal

    df = main.merge(sym, on="DATE", how="left").merge(rfx, on="DATE", how="left").sort_values("DATE")
    df["sym_signal"] = df["sym_signal"].astype(float).clip(-1,1).fillna(0.0)
    df["reflexive_signal"] = df["reflexive_signal"].astype(float).clip(-1,1).fillna(0.0)

    r  = df["ret"].fillna(0.0)
    eq = (1.0 + r).cumprod()
    vol20   = r.rolling(20, min_periods=10).std() * np.sqrt(252.0)
    trend63 = _trend_strength(eq, 63)
    chop21  = _choppiness(r, 21)

    # Build feature matrix (z-scored windowed)
    feats = pd.DataFrame({
        "DATE": df["DATE"],
        "vol20": vol20.fillna(0.0),
        "trend63": trend63.fillna(0.0),
        "chop21": chop21.fillna(0.0),
        "sym": df["sym_signal"],
        "reflex": df["reflexive_signal"],
    }).reset_index(drop=True)

    # rolling baseline (median) over 126d
    W = 126
    cols = ["vol20","trend63","chop21","sym","reflex"]
    X = feats[cols].astype(float)

    # zscore per column (robust clip)
    Xz = (X - X.rolling(W, min_periods=20).median()) / (X.rolling(W, min_periods=20).std().replace(0, np.nan))
    Xz = Xz.fillna(0.0).clip(-5,5)

    # drift = cosine distance to rolling median vector
    med = Xz.rolling(W, min_periods=20).median().fillna(0.0)
    drift = []
    for i in range(len(Xz)):
        a = Xz.iloc[i].values
        b = med.iloc[i].values
        drift.append(_cosine_dist(a, b))
    feats["dna_drift"] = np.array(drift)

    # save series
    out = feats[["DATE","dna_drift","vol20","trend63","chop21","sym","reflex"]].copy()
    out.to_csv(RUNS/"dna_drift.csv", index=False)

    # Compatibility JSON expected by WF+ tooling: per-symbol date->drift map.
    drift_series = out[["DATE", "dna_drift"]].dropna().copy()
    drift_series["DATE"] = pd.to_datetime(drift_series["DATE"], errors="coerce")
    drift_series = drift_series.dropna(subset=["DATE"])
    date_map = {
        d.strftime("%Y-%m-%d"): float(v)
        for d, v in zip(drift_series["DATE"], drift_series["dna_drift"])
    }
    symbols = sorted(p.stem.replace("_prices", "") for p in DATA.glob("*.csv"))
    drift_map = {str(sym).upper(): dict(date_map) for sym in symbols}
    (RUNS / "dna_drift.json").write_text(json.dumps({"dna_drift": drift_map}, indent=2))

    # Optional preview plot for report card.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not drift_series.empty:
            plt.figure(figsize=(8, 3))
            plt.plot(drift_series["DATE"], drift_series["dna_drift"])
            plt.title("DNA Drift (market-level)")
            plt.tight_layout()
            plt.savefig(RUNS / "dna_drift.png", dpi=140)
            plt.close()
    except Exception:
        pass

    # summary
    last_row = out.dropna().iloc[-1] if len(out.dropna()) else None
    summary = {
        "last": {
            "date": (str(last_row["DATE"].date()) if last_row is not None else None),
            "drift": (float(last_row["dna_drift"]) if last_row is not None else None)
        },
        "window_days": W,
        "note": "dna_drift is 1 - cosine similarity to rolling median of z-scored features."
    }
    (RUNS/"dna_summary.json").write_text(json.dumps(summary, indent=2))

    print("✅ Wrote runs_plus/dna_drift.csv")
    print("✅ Wrote runs_plus/dna_drift.json")
    print("✅ Wrote runs_plus/dna_drift.png")
    print("✅ Wrote runs_plus/dna_summary.json")
