#!/usr/bin/env python3
# tools/run_osc_portfolio.py  (v7: safe returns, no -inf dd)
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs_plus"
RUNS.mkdir(parents=True, exist_ok=True)

def load_series(p: Path):
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date", df.columns[0])
    vcol = cols.get("value", df.columns[-1])
    s = pd.to_numeric(df[vcol], errors="coerce")
    d = pd.to_datetime(df[dcol], errors="coerce")
    out = pd.DataFrame({"DATE": d, "VALUE": s}).dropna().sort_values("DATE")
    return out

def to_returns_and_level(val: pd.Series):
    v = pd.to_numeric(val, errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
    if v.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    looks_like_returns = (v.abs().median() < 0.05 and abs(v.mean()) < 0.02)
    if looks_like_returns:
        ret = v.fillna(0.0)
        lvl = (1.0 + ret).cumprod()
    else:
        ret = v.pct_change().fillna(0.0)
        lvl = v
    return ret, lvl

def ann_sharpe(r):
    r = pd.Series(r).replace([np.inf,-np.inf], np.nan).dropna()
    s = r.std()
    if s == 0 or np.isnan(s): 
        return 0.0
    return float((r.mean()/s)*np.sqrt(252))

if __name__ == "__main__":
    frames=[]
    for p in sorted(DATA.glob("*.csv")):
        sym = p.stem
        try:
            df = load_series(p)
        except Exception:
            continue
        if len(df) < 120:
            continue

        ret, level = to_returns_and_level(df["VALUE"])
        if ret.empty or level.empty:
            continue

        # Momentum sign (21d) + daily sign fallback
        mom = level.pct_change(21)
        pos1 = np.sign(mom).fillna(0.0)
        pos2 = np.sign(ret).fillna(0.0)
        pos = (0.7*pos1 + 0.3*pos2)
        pos = pd.Series(pos, index=ret.index).ewm(span=5, adjust=False).mean().clip(-1,1).fillna(0.0)
        if float(pos.abs().sum()) == 0.0:
            pos = pos2.copy()

        # SAFE: clip returns so equity can never hit zero
        ret = pd.Series(ret).replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(lower=-0.5, upper=0.5)
        pos = pd.Series(pos).replace([np.inf,-np.inf], 0.0).fillna(0.0).clip(-1,1)

        frames.append(pd.DataFrame({
            "DATE": df["DATE"].iloc[:len(ret)],
            "ret": ret.values,
            "pos": pos.values,
            "asset": sym
        }))

    if not frames:
        raise SystemExit("No assets produced oscillator signals (check data/*.csv).")

    all_df = pd.concat(frames, ignore_index=True)
    R = all_df.pivot(index="DATE", columns="asset", values="ret").sort_index().fillna(0.0)
    P = all_df.pivot(index="DATE", columns="asset", values="pos").reindex(R.index).fillna(0.0)

    # Simple daily equal-weight (robust)
    n_assets = R.shape[1]
    if n_assets < 2:
        raise SystemExit("Need >=2 assets for oscillator portfolio.")
    W = pd.DataFrame(1.0/n_assets, index=R.index, columns=R.columns)

    port = (P * R * W).sum(axis=1)
    port = pd.Series(port).replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5, 0.5)

    # Equity and safe drawdown
    eq = (1.0 + port).cumprod()
    peak = pd.concat([eq, pd.Series(1.0, index=eq.index)], axis=1).max(axis=1).cummax()
    dd_series = (eq / peak) - 1.0
    dd = float(np.nan_to_num(dd_series.min(), nan=0.0, posinf=0.0, neginf=-1.0))

    sharpe = ann_sharpe(port)
    hit = float((port > 0).mean())

    out = pd.DataFrame({"DATE": R.index, "ret": port.values, "eq": eq.values})
    out.to_csv(RUNS / "osc_portfolio.csv", index=False)

    RUNS.joinpath("osc_portfolio_summary.json").write_text(
        pd.Series({"sharpe": float(sharpe), "hit": float(hit), "maxdd": dd, "assets_used": int(n_assets)}, dtype=object).to_json(indent=2)
    )

    print("\nOSCILLATOR PORTFOLIO (v7 safe)")
    print("================================")
    print(f"Sharpe: {sharpe:.3f}")
    print(f"Hit:    {hit:.3f}")
    print(f"MaxDD:  {dd:.3f}")
    print(f"(Assets used: {n_assets})")
    print(f"Saved: {RUNS/'osc_portfolio.csv'} and osc_portfolio_summary.json")
