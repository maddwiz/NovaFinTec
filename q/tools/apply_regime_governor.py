#!/usr/bin/env python3
# tools/apply_regime_governor.py
#
# Builds a GOVERNED portfolio alongside your existing Main-only final.
# Uses regime-suggested weights and available sleeve return streams.
#
# Inputs we try to read (robust, optional except portfolio_plus + regime_weights):
#   runs_plus/portfolio_plus.csv              -> Main returns
#   runs_plus/regime_weights.csv             -> DATE, w_main, w_vol, w_osc, w_sym, w_reflex
#   runs_plus/sleeve_vol.csv                 -> Vol sleeve daily returns (any 'ret' column)
#   runs_plus/vol_overlay.csv / volatility_overlay.csv / vol_overlay.csv  (fallback names)
#   runs_plus/sleeve_osc.csv                 -> Oscillator sleeve returns (any 'ret' column)
#   runs_plus/osc_overlay.csv / oscillator_overlay.csv  (fallback names)
#   runs_plus/symbolic_signal.csv            -> sym_signal (we map signal -> small return)
#   runs_plus/reflexive_signal.csv           -> reflexive_signal (signal -> small return)
#
# Outputs:
#   runs_plus/final_portfolio_regime.csv
#   runs_plus/final_portfolio_regime_summary.json
#
# Notes:
# - If a sleeve return series is missing, we set that sleeve to 0 and re-allocate that weight to Main for that day.
# - Symbolic/Reflexive are signals, not PnL; we map them to tiny returns with conservative scaling caps.
# - Nothing here overwrites your existing final files.

from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

# Conservative daily scale for mapping bounded signals -> pseudo-returns
SCALE_SYM    = 0.0005   # 15 bps *at most* when signal=1.0
SCALE_REFLEX = 0.0003   # 10 bps *at most* when signal=1.0

def _safe_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf], np.nan)

def _load_main_returns():
    p = RUNS / "portfolio_plus.csv"
    if not p.exists():
        raise SystemExit("Missing runs_plus/portfolio_plus.csv")
    df = pd.read_csv(p)
    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or lowers.get("timestamp") or list(df.columns)[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).rename(columns={dcol:"DATE"}).sort_values("DATE")
    # try return columns
    for c in ["ret_net","ret","ret_plus","ret_gross","return","pnl","pnl_plus","daily_ret","portfolio_ret","port_ret"]:
        if c in df.columns:
            r = _safe_num(df[c]).fillna(0.0).clip(-0.5,0.5)
            return pd.DataFrame({"DATE": df["DATE"], "ret_main": r})
    # fallback: equity -> returns
    for c in ["eq_net","eq","equity","equity_curve","portfolio_eq","equity_index","port_equity"]:
        if c in df.columns:
            eq = _safe_num(df[c])
            r = eq.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
            return pd.DataFrame({"DATE": df["DATE"], "ret_main": r})
    raise SystemExit("Could not find returns or equity in portfolio_plus.csv")

def _find_first_existing(paths):
    for name in paths:
        p = RUNS / name
        if p.exists():
            return p
    return None

def _load_ret_stream(candidates, colnames=("ret","ret_net","overlay_ret","pnl","return","daily_ret","port_ret")):
    p = _find_first_existing(candidates)
    if not p:
        return pd.DataFrame(columns=["DATE","ret"])
    df = pd.read_csv(p)
    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or lowers.get("timestamp") or list(df.columns)[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).rename(columns={dcol:"DATE"})
    # pick a return-like column
    for c in colnames:
        if c in df.columns:
            r = _safe_num(df[c]).fillna(0.0).clip(-0.5,0.5)
            return pd.DataFrame({"DATE": df["DATE"], "ret": r}).sort_values("DATE")
    # last resort: try to compute from equity if present
    for c in ["eq_net","eq","equity","equity_curve","equity_index"]:
        if c in df.columns:
            eq = _safe_num(df[c])
            r = eq.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
            return pd.DataFrame({"DATE": df["DATE"], "ret": r}).sort_values("DATE")
    return pd.DataFrame(columns=["DATE","ret"])

def _load_signal(pathname, sig_col, scale):
    p = RUNS / pathname
    if not p.exists():
        return pd.DataFrame(columns=["DATE","ret"])
    df = pd.read_csv(p, parse_dates=["DATE"])
    # If multiple assets, average
    if "ASSET" in df.columns and df["ASSET"].nunique() > 1:
        g = df.groupby("DATE", as_index=False)[sig_col].mean()
        sig = g[sig_col].clip(-1.0, 1.0)
        ret = (sig * scale).clip(-0.05, 0.05)  # hard daily cap
        return pd.DataFrame({"DATE": g["DATE"], "ret": ret}).sort_values("DATE")
    # Single stream
    if sig_col not in df.columns:
        return pd.DataFrame(columns=["DATE","ret"])
    sig = df[sig_col].clip(-1.0, 1.0)
    ret = (sig * scale).clip(-0.05, 0.05)
    return pd.DataFrame({"DATE": df["DATE"], "ret": ret}).sort_values("DATE")

def _ann_sharpe(r):
    s = pd.Series(r).replace([np.inf,-np.inf], np.nan).dropna()
    if s.empty: return 0.0
    sd = s.std()
    if not np.isfinite(sd) or sd == 0: return 0.0
    return float((s.mean()/sd) * np.sqrt(252.0))

def _maxdd(r):
    s = pd.Series(r).fillna(0.0)
    eq = (1.0 + s).cumprod()
    peak = eq.cummax()
    dd = eq/peak - 1.0
    return float(dd.min())

def _hit(r):
    s = pd.Series(r).dropna()
    if s.empty: return 0.0
    return float((s > 0).mean())

def _split_is_oos(r, frac=0.75):
    n = len(r)
    k = int(n*frac)
    return r[:k], r[k:]

if __name__ == "__main__":
    RUNS.mkdir(parents=True, exist_ok=True)

    main = _load_main_returns()                                 # DATE, ret_main
    wts  = pd.read_csv(RUNS/"regime_weights.csv", parse_dates=["DATE"])  # DATE, regime, w_*

    # Optional sleeves
    vol = _load_ret_stream([
        "sleeve_vol.csv", "vol_overlay.csv", "volatility_overlay.csv", "vol_overlay_plus.csv"
    ])  # DATE, ret
    osc = _load_ret_stream([
        "sleeve_osc.csv", "osc_overlay.csv", "oscillator_overlay.csv", "osc_overlay_plus.csv"
    ])
    sym = _load_signal("symbolic_signal.csv", "sym_signal", SCALE_SYM)
    reflex = _load_signal("reflexive_signal.csv", "reflexive_signal", SCALE_REFLEX)

    # Merge everything by DATE
    df = main.copy()
    for (name, d) in [("ret_vol", vol), ("ret_osc", osc), ("ret_sym", sym), ("ret_reflex", reflex)]:
        if not d.empty:
            df = df.merge(d.rename(columns={"ret": name}), on="DATE", how="left")
        else:
            df[name] = 0.0
    df = df.sort_values("DATE").reset_index(drop=True)

    # Attach weights (forward-fill to cover every date)
    w = wts[["DATE","w_main","w_vol","w_osc","w_sym","w_reflex"]].copy()
    w = w.sort_values("DATE").reset_index(drop=True)
    df = df.merge(w, on="DATE", how="left")
    # Forward-fill weights; if still NaN at start, use conservative defaults
    df[["w_main","w_vol","w_osc","w_sym","w_reflex"]] = df[["w_main","w_vol","w_osc","w_sym","w_reflex"]].ffill()
    df[["w_main","w_vol","w_osc","w_sym","w_reflex"]] = df[["w_main","w_vol","w_osc","w_sym","w_reflex"]].fillna(value={"w_main":1.0,"w_vol":0.0,"w_osc":0.0,"w_sym":0.0,"w_reflex":0.0})

    # If a sleeve stream is effectively zero (missing), reallocate its weight to Main for that day
    present = pd.DataFrame({
        "vol_ok":    (df["ret_vol"].abs() > 0).astype(float),
        "osc_ok":    (df["ret_osc"].abs() > 0).astype(float),
        "sym_ok":    (df["ret_sym"].abs() > 0).astype(float),
        "reflex_ok": (df["ret_reflex"].abs() > 0).astype(float),
    })
    # if ok==0, move its weight to main
    df["w_vol_eff"]    = df["w_vol"]    * present["vol_ok"]
    df["w_osc_eff"]    = df["w_osc"]    * present["osc_ok"]
    df["w_sym_eff"]    = df["w_sym"]    * present["sym_ok"]
    df["w_reflex_eff"] = df["w_reflex"] * present["reflex_ok"]
    others = df["w_vol_eff"] + df["w_osc_eff"] + df["w_sym_eff"] + df["w_reflex_eff"]
    df["w_main_eff"] = 1.0 - others
    # keep in [0,1]
    for c in ["w_main_eff","w_vol_eff","w_osc_eff","w_sym_eff","w_reflex_eff"]:
        df[c] = df[c].clip(lower=0.0, upper=1.0)

    # Build governed return
    df["ret_governed"] = (
        df["w_main_eff"]  * df["ret_main"] +
        df["w_vol_eff"]   * df["ret_vol"] +
        df["w_osc_eff"]   * df["ret_osc"] +
        df["w_sym_eff"]   * df["ret_sym"] +
        df["w_reflex_eff"]* df["ret_reflex"]
    ).fillna(0.0)

    out = df[[
        "DATE","ret_governed","ret_main",
        "w_main_eff","w_vol_eff","w_osc_eff","w_sym_eff","w_reflex_eff"
    ]].copy()
    out = out.sort_values("DATE").reset_index(drop=True)
    out["eq_governed"] = (1.0 + out["ret_governed"]).cumprod()
    out["eq_main"]     = (1.0 + out["ret_main"]).cumprod()

    # Metrics
    r = out["ret_governed"]
    r_is, r_oos = _split_is_oos(r.values, 0.75)
    m = {
        "in_sample":  {"sharpe": _ann_sharpe(r_is),  "hit": _hit(r_is),  "maxdd": _maxdd(r_is)},
        "out_sample": {"sharpe": _ann_sharpe(r_oos), "hit": _hit(r_oos), "maxdd": _maxdd(r_oos)},
        "weights_note": "Effective weights reallocate missing sleeves back to Main.",
        "scales": {"SCALE_SYM": SCALE_SYM, "SCALE_REFLEX": SCALE_REFLEX}
    }

    # Save governed
    out_csv = RUNS / "final_portfolio_regime.csv"
    out_json= RUNS / "final_portfolio_regime_summary.json"
    out.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(m, indent=2))

    print("✅ Wrote", out_csv.name, "and", out_json.name)
    print("IS Sharpe:", f"{m['in_sample']['sharpe']:.3f}",
          "| OOS Sharpe:", f"{m['out_sample']['sharpe']:.3f}",
          "| Note:", m["weights_note"])
