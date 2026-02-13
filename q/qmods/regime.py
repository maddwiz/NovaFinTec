#!/usr/bin/env python3
# qmods/regime.py
# Adaptive Regime Switcher (observer-only, robust to missing cols)

from pathlib import Path
import pandas as pd
import numpy as np
import json
import re

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

MAIN_P = RUNS / "portfolio_plus.csv"
SYMP   = RUNS / "symbolic_signal.csv"
REFLP  = RUNS / "reflexive_signal.csv"

def _safe(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf], np.nan)

def _load_main():
    if not MAIN_P.exists():
        raise SystemExit("Missing runs_plus/portfolio_plus.csv")
    df = pd.read_csv(MAIN_P)
    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or lowers.get("timestamp") or df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).rename(columns={dcol:"DATE"}).sort_values("DATE")

    # prefer returns if present
    for c in ["ret_net","ret","ret_plus","ret_gross","return","pnl","pnl_plus","daily_ret","portfolio_ret","port_ret"]:
        if c in df.columns:
            r = _safe(df[c]).fillna(0.0).clip(-0.5,0.5)
            return pd.DataFrame({"DATE": df["DATE"], "ret": r})

    # fallback: equity → returns
    for c in ["eq_net","eq","equity","equity_curve","portfolio_eq","equity_index","port_equity"]:
        if c in df.columns:
            eq = _safe(df[c])
            r = eq.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
            return pd.DataFrame({"DATE": df["DATE"], "ret": r})

    raise SystemExit("Could not find returns/equity in portfolio_plus.csv")

def _auto_pick_signal_col(df, want):
    """
    If 'want' not found, try to find a close match:
      - exact
      - case-insensitive
      - contains 'reflex' or 'sym' for the respective wants
    Returns column name or None.
    """
    cols = list(df.columns)
    if want in cols: return want
    low = {c.lower(): c for c in cols}
    if want.lower() in low: return low[want.lower()]

    if "reflex" in want.lower():
        # find anything that looks like reflex/reflexive
        for c in cols:
            if re.search(r"reflex", c, re.IGNORECASE):
                return c
    if "sym" in want.lower():
        for c in cols:
            if re.search(r"\bsym\b|symbolic", c, re.IGNORECASE):
                return c
    return None

def _load_signal(path, sig_col):
    """
    Load a DATE + signal column robustly.
    If ASSET exists with >1 asset, average per DATE.
    If the requested column isn't present, try to auto-detect; if still not found,
    return empty frame with DATE and the requested name to avoid KeyErrors.
    """
    if not path.exists():
        return pd.DataFrame(columns=["DATE", sig_col])

    df = pd.read_csv(path)
    # DATE handling
    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or "DATE"
    if dcol not in df.columns:
        # nothing we can do
        return pd.DataFrame(columns=["DATE", sig_col])
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).rename(columns={dcol:"DATE"})

    # pick a usable signal column
    use_col = _auto_pick_signal_col(df, sig_col)
    if use_col is None:
        return pd.DataFrame(columns=["DATE", sig_col])

    # average across assets if present
    if "ASSET" in df.columns and df["ASSET"].nunique() > 1:
        g = df.groupby("DATE", as_index=False)[use_col].mean()
        g = g.rename(columns={use_col: sig_col})
        return g[["DATE", sig_col]].sort_values("DATE").reset_index(drop=True)

    g = df[["DATE", use_col]].rename(columns={use_col: sig_col}).sort_values("DATE").reset_index(drop=True)
    return g

def _trend_strength(price_series: pd.Series, win=63) -> pd.Series:
    s = pd.Series(price_series.values, index=price_series.index)
    t = pd.Series(np.arange(len(s)), index=s.index)
    def _rho(x):
        y = pd.Series(x)
        z = pd.Series(np.arange(len(y)))
        if y.std(ddof=0) == 0 or z.std(ddof=0) == 0:
            return 0.0
        return float(np.corrcoef(y, z)[0,1])
    out = s.rolling(win, min_periods=max(5, win//3)).apply(lambda x: _rho(x), raw=True)
    return out.abs().fillna(0.0)

def _choppiness(ret_series: pd.Series, win=21) -> pd.Series:
    sgn = np.sign(pd.Series(ret_series).fillna(0.0))
    def _fliprate(x):
        y = pd.Series(x)
        flips = (np.sign(y).diff() != 0).sum()
        denom = max(len(y) - 1, 1)
        return float(flips) / float(denom)
    return sgn.rolling(win, min_periods=max(5, win//3)).apply(lambda x: _fliprate(x), raw=True).fillna(0.0)

def _classify(vol: pd.Series, trend: pd.Series, chop: pd.Series,
              sym: pd.Series|None=None, reflex: pd.Series|None=None) -> pd.Series:
    v = pd.Series(vol).copy()
    t = pd.Series(trend).copy()
    c = pd.Series(chop).copy()
    s = pd.Series(sym).reindex(v.index) if sym is not None else pd.Series([np.nan]*len(v), index=v.index)
    rf= pd.Series(reflex).reindex(v.index) if reflex is not None else pd.Series([np.nan]*len(v), index=v.index)

    if v.notna().sum() > 20:
        v60, v85, v95 = np.nanpercentile(v.dropna(), [60,85,95])
    else:
        v60 = v85 = v95 = np.nan

    regs = []
    for i in v.index:
        vi = v.loc[i]; ti = float(t.loc[i]) if pd.notna(t.loc[i]) else 0.0; ci = float(c.loc[i]) if pd.notna(c.loc[i]) else 0.0
        si = float(s.loc[i]) if pd.notna(s.loc[i]) else 0.0
        rfi= float(rf.loc[i]) if pd.notna(rf.loc[i]) else 0.0

        if (pd.notna(vi) and pd.notna(v95) and vi >= v95) or (si - 0.5*rfi < -0.6):
            regs.append("CRISIS")
        elif (pd.notna(vi) and pd.notna(v85) and vi >= v85) and ti > 0.2:
            regs.append("HIGHVOL_TREND")
        elif (pd.notna(vi) and pd.notna(v60) and vi <= v60) and ti > 0.2 and ci < 0.4:
            regs.append("CALM_TREND")
        else:
            regs.append("CALM_CHOP")
    return pd.Series(regs, index=v.index)

def _suggest_weights(reg: str) -> dict:
    mapping = {
        "CALM_TREND":     dict(w_main=0.70, w_vol=0.15, w_osc=0.05, w_sym=0.05, w_reflex=0.05),
        "CALM_CHOP":      dict(w_main=0.70, w_vol=0.05, w_osc=0.20, w_sym=0.03, w_reflex=0.02),
        "HIGHVOL_TREND":  dict(w_main=0.65, w_vol=0.25, w_osc=0.05, w_sym=0.03, w_reflex=0.02),
        "CRISIS":         dict(w_main=0.80, w_vol=0.15, w_osc=0.00, w_sym=0.03, w_reflex=0.02),
    }
    base = mapping.get(reg, mapping["CALM_CHOP"])
    rest = 1.0 - base["w_main"]
    others = base["w_vol"] + base["w_osc"] + base["w_sym"] + base["w_reflex"]
    if others > rest and others > 0:
        scale = rest / others
        for k in ["w_vol","w_osc","w_sym","w_reflex"]:
            base[k] *= scale
    base["w_main"] = 1.0 - (base["w_vol"] + base["w_osc"] + base["w_sym"] + base["w_reflex"])
    return base

def run_regime():
    RUNS.mkdir(parents=True, exist_ok=True)

    main = _load_main()
    sym = _load_signal(SYMP, "sym_signal")
    reflex = _load_signal(REFLP, "reflexive_signal")

    df = main.copy()
    if not sym.empty:    df = df.merge(sym, on="DATE", how="left")
    if not reflex.empty: df = df.merge(reflex, on="DATE", how="left")
    df = df.sort_values("DATE").reset_index(drop=True)

    r = pd.Series(df["ret"]).fillna(0.0)
    eq = (1.0 + r).cumprod()

    vol20   = r.rolling(20, min_periods=10).std() * np.sqrt(252.0)
    trend63 = _trend_strength(eq, 63)
    chop21  = _choppiness(r, 21)
    sym_s   = pd.Series(df["sym_signal"]) if "sym_signal" in df.columns else pd.Series([np.nan]*len(df))
    ref_s   = pd.Series(df["reflexive_signal"]) if "reflexive_signal" in df.columns else pd.Series([np.nan]*len(df))

    regime = _classify(vol20, trend63, chop21, sym_s, ref_s)

    feat = pd.DataFrame({
        "DATE": df["DATE"],
        "vol20_ann": vol20,
        "trend63": trend63,
        "chop21": chop21,
        "sym": sym_s,
        "reflex": ref_s,
        "regime": regime
    })
    feat.to_csv(RUNS/"regime_series.csv", index=False)

    rows = []
    for i in range(len(feat)):
        reg = regime.iloc[i]
        w = _suggest_weights(reg)
        rows.append({"DATE": feat["DATE"].iloc[i], "regime": reg, **w})
    wdf = pd.DataFrame(rows)
    wdf.to_csv(RUNS/"regime_weights.csv", index=False)

    recent = feat.tail(252)
    counts = recent["regime"].value_counts().to_dict()
    cur = wdf.iloc[-1].to_dict() if len(wdf) else {}
    summary = {
        "recent_counts": counts,
        "current": {
            "date": str(cur.get("DATE","")),
            "regime": cur.get("regime",""),
            "suggested_weights": {k: float(cur[k]) for k in ["w_main","w_vol","w_osc","w_sym","w_reflex"] if k in cur}
        }
    }
    (RUNS/"regime_summary.json").write_text(json.dumps(summary, indent=2))
    return feat, wdf, summary

if __name__ == "__main__":
    feat, wdf, summary = run_regime()
    print("Regime rows:", len(feat), "| weights rows:", len(wdf))
    print("Recent regime mix:", summary.get("recent_counts", {}))
    cur = summary.get("current", {})
    print("Current:", cur.get("date"), cur.get("regime"), cur.get("suggested_weights"))
