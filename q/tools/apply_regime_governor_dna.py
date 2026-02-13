#!/usr/bin/env python3
# tools/apply_regime_governor_dna.py
# Builds final_portfolio_regime_dna.* using regime_weights_dna.csv

from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

SCALE_SYM    = 0.0005
SCALE_REFLEX = 0.0003

def _safe_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf], np.nan)

def _load_main():
    p = RUNS / "portfolio_plus.csv"
    if not p.exists(): raise SystemExit("Missing runs_plus/portfolio_plus.csv")
    df = pd.read_csv(p)
    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or lowers.get("timestamp") or list(df.columns)[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).rename(columns={dcol:"DATE"}).sort_values("DATE")
    for c in ["ret_net","ret","ret_plus","ret_gross","return","pnl","pnl_plus","daily_ret","portfolio_ret","port_ret"]:
        if c in df.columns:
            return pd.DataFrame({"DATE": df["DATE"], "ret_main": _safe_num(df[c]).fillna(0.0).clip(-0.5,0.5)})
    for c in ["eq_net","eq","equity","equity_curve","portfolio_eq","equity_index","port_equity"]:
        if c in df.columns:
            eq = _safe_num(df[c])
            r = eq.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
            return pd.DataFrame({"DATE": df["DATE"], "ret_main": r})
    raise SystemExit("No returns/equity in portfolio_plus.csv")

def _load_ret(name_candidates):
    for name in name_candidates:
        p = RUNS / name
        if p.exists():
            df = pd.read_csv(p)
            lowers = {c.lower(): c for c in df.columns}
            dcol = lowers.get("date") or lowers.get("timestamp") or list(df.columns)[0]
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
            df = df.dropna(subset=[dcol]).rename(columns={dcol:"DATE"})
            for c in ["ret","ret_net","overlay_ret","pnl","return","daily_ret","port_ret"]:
                if c in df.columns:
                    return pd.DataFrame({"DATE": df["DATE"], "ret": _safe_num(df[c]).fillna(0.0).clip(-0.5,0.5)}).sort_values("DATE")
            for c in ["eq_net","eq","equity","equity_curve","equity_index"]:
                if c in df.columns:
                    eq = _safe_num(df[c])
                    r = eq.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
                    return pd.DataFrame({"DATE": df["DATE"], "ret": r}).sort_values("DATE")
    return pd.DataFrame(columns=["DATE","ret"])

def _load_signal(pathname, sig_col, scale):
    p = RUNS / pathname
    if not p.exists(): return pd.DataFrame(columns=["DATE","ret"])
    df = pd.read_csv(p, parse_dates=["DATE"])
    if "ASSET" in df.columns and df["ASSET"].nunique() > 1:
        g = df.groupby("DATE", as_index=False)[sig_col].mean()
        sig = g[sig_col].clip(-1.0, 1.0)
        ret = (sig * scale).clip(-0.05, 0.05)
        return pd.DataFrame({"DATE": g["DATE"], "ret": ret}).sort_values("DATE")
    if sig_col not in df.columns: return pd.DataFrame(columns=["DATE","ret"])
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
    dd = eq/eq.cummax() - 1.0
    return float(dd.min())
def _hit(r):
    s = pd.Series(r).dropna()
    return float((s > 0).mean()) if not s.empty else 0.0
def _split(r, frac=0.75):
    n = len(r); k = int(n*frac); return r[:k], r[k:]

if __name__ == "__main__":
    main = _load_main()
    wts  = pd.read_csv(RUNS/"regime_weights_dna.csv", parse_dates=["DATE"])

    vol = _load_ret(["sleeve_vol.csv","vol_overlay.csv","volatility_overlay.csv","vol_overlay_plus.csv"])
    osc = _load_ret(["sleeve_osc.csv","osc_overlay.csv","oscillator_overlay.csv","osc_overlay_plus.csv"])
    sym = _load_signal("symbolic_signal.csv","sym_signal",SCALE_SYM)
    rfx = _load_signal("reflexive_signal.csv","reflexive_signal",SCALE_REFLEX)

    df = main.copy()
    for (nm, d) in [("ret_vol",vol),("ret_osc",osc),("ret_sym",sym),("ret_reflex",rfx)]:
        df = df.merge(d.rename(columns={"ret":nm}), on="DATE", how="left") if not d.empty else df.assign(**{nm:0.0})
    w = wts.rename(columns={
        "w_main_dna":"w_main","w_vol_dna":"w_vol","w_osc_dna":"w_osc","w_sym_dna":"w_sym","w_reflex_dna":"w_reflex"
    })[["DATE","w_main","w_vol","w_osc","w_sym","w_reflex"]].sort_values("DATE")
    df = df.merge(w, on="DATE", how="left")
    df[["w_main","w_vol","w_osc","w_sym","w_reflex"]] = df[["w_main","w_vol","w_osc","w_sym","w_reflex"]].ffill().fillna({"w_main":1.0,"w_vol":0.0,"w_osc":0.0,"w_sym":0.0,"w_reflex":0.0})

    # governed DNA returns
    df["ret_governed_dna"] = (
        df["w_main"]*df["ret_main"] + df["w_vol"]*df["ret_vol"] + df["w_osc"]*df["ret_osc"] +
        df["w_sym"]*df["ret_sym"] + df["w_reflex"]*df["ret_reflex"]
    ).fillna(0.0)

    out = df[["DATE","ret_main","ret_governed_dna"]].copy().sort_values("DATE")
    out["eq_main"] = (1.0 + out["ret_main"]).cumprod()
    out["eq_governed_dna"] = (1.0 + out["ret_governed_dna"]).cumprod()

    r = out["ret_governed_dna"].values
    r_is, r_oos = _split(r, 0.75)
    m = {
        "in_sample":  {"sharpe": _ann_sharpe(r_is),  "hit": _hit(r_is),  "maxdd": _maxdd(r_is)},
        "out_sample": {"sharpe": _ann_sharpe(r_oos), "hit": _hit(r_oos), "maxdd": _maxdd(r_oos)},
        "note": "DNA drift attenuates add-on sleeves; weight reallocated to Main."
    }

    out.to_csv(RUNS/"final_portfolio_regime_dna.csv", index=False)
    (RUNS/"final_portfolio_regime_dna_summary.json").write_text(json.dumps(m, indent=2))

    print("✅ Wrote final_portfolio_regime_dna.csv and final_portfolio_regime_dna_summary.json")
    print("IS Sharpe: %.3f | OOS Sharpe: %.3f" % (m["in_sample"]["sharpe"], m["out_sample"]["sharpe"]))
