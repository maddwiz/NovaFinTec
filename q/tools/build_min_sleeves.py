#!/usr/bin/env python3
# tools/build_min_sleeves.py
# Build minimal sleeve streams so the Regime governor has inputs.
# Writes in runs_plus/:
#   sleeve_vol.csv        (DATE, ret)
#   sleeve_osc.csv        (DATE, ret)
#   symbolic_signal.csv   (DATE, sym_signal)
#   reflexive_signal.csv  (DATE, reflexive_signal)

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(parents=True, exist_ok=True)

P_PORT = RUNS / "portfolio_plus.csv"

# ==== KNOBS (tiny nudges) ====
VOL_SCALE = 0.0014    # was 0.0015
OSC_SCALE = 0.0022    # was 0.0012
SYM_DEFAULT = 0.0
RFX_DEFAULT = 0.0
# =============================

def _safe_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf], np.nan)

def _load_main():
    if not P_PORT.exists():
        raise SystemExit("Missing runs_plus/portfolio_plus.csv (run portfolio_from_runs_plus.py first).")
    df = pd.read_csv(P_PORT)
    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or lowers.get("timestamp") or list(df.columns)[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).rename(columns={dcol:"DATE"}).sort_values("DATE")
    for c in ["ret","ret_net","ret_plus","return","daily_ret","port_ret","portfolio_ret","pnl","pnl_plus"]:
        if c in df.columns:
            r = _safe_num(df[c]).fillna(0.0).clip(-0.5,0.5)
            return pd.DataFrame({"DATE": df["DATE"], "ret_main": r})
    for c in ["eq","eq_net","equity","equity_curve","equity_index","portfolio_eq","port_equity"]:
        if c in df.columns:
            eq = _safe_num(df[c])
            r = eq.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
            return pd.DataFrame({"DATE": df["DATE"], "ret_main": r})
    raise SystemExit("portfolio_plus.csv has no returns/equity columns I recognize.")

def _zscore(x, w):
    s = pd.Series(x)
    m = s.rolling(w, min_periods=max(5, w//3)).mean()
    v = s.rolling(w, min_periods=max(5, w//3)).std()
    z = (s - m) / v.replace(0, np.nan)
    return z.fillna(0.0).clip(-5,5)

if __name__ == "__main__":
    df = _load_main()  # DATE, ret_main
    r  = df["ret_main"].values

    # VOL sleeve: low realized vol -> +, high vol -> -
    vol20 = pd.Series(r).rolling(20, min_periods=10).std()
    vol100= pd.Series(r).rolling(100, min_periods=30).std()
    vol_rel = (vol20 / vol100.replace(0,np.nan)).replace([np.inf,-np.inf], np.nan).fillna(1.0)
    sig_vol = (1.0 - vol_rel).clip(-1.0, 1.0).fillna(0.0)
    ret_vol = (sig_vol * VOL_SCALE).clip(-0.05, 0.05)
    pd.DataFrame({"DATE": df["DATE"], "ret": ret_vol}).to_csv(RUNS/"sleeve_vol.csv", index=False)

    # OSC sleeve: fade short-term extremes
    z10 = _zscore(r, 10)
    sig_osc = (-z10).clip(-1.0, 1.0)
    ret_osc = (sig_osc * OSC_SCALE).clip(-0.05, 0.05)
    pd.DataFrame({"DATE": df["DATE"], "ret": ret_osc}).to_csv(RUNS/"sleeve_osc.csv", index=False)

    # Symbolic & Reflexive placeholders (kept tiny at zero for now)
    pd.DataFrame({"DATE": df["DATE"], "sym_signal": SYM_DEFAULT}).to_csv(RUNS/"symbolic_signal.csv", index=False)
    pd.DataFrame({"DATE": df["DATE"], "reflexive_signal": RFX_DEFAULT}).to_csv(RUNS/"reflexive_signal.csv", index=False)

    print("✅ Wrote: sleeve_vol.csv, sleeve_osc.csv, symbolic_signal.csv, reflexive_signal.csv")
