#!/usr/bin/env python3
# tools/run_osc_portfolio_costed.py
# v1: oscillator with per-turnover cost, position-change cap, and safe stats.

from pathlib import Path
import pandas as pd, numpy as np, json, os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True, parents=True)

# ====== Params via ENV (you can change without editing file) ======
COST_BPS   = float(os.getenv("OSC_COST_BPS", "2.0")) / 10000.0  # 2 bps per 1.0 turnover
MAX_DPOS   = float(os.getenv("OSC_MAX_DPOS", "0.20"))           # max daily change in position (abs)
SMOOTH_SPAN= int(os.getenv("OSC_SMOOTH_SPAN", "5"))             # ewm smoothing for position
MOM_WIN    = int(os.getenv("OSC_MOM_WIN", "21"))                # momentum window (days)
CAP_RET    = float(os.getenv("OSC_CAP_RET", "0.50"))            # cap per-day abs return
MIN_SERIES = int(os.getenv("OSC_MIN_SERIES", "120"))            # min rows required

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
    used=0
    for p in sorted(DATA.glob("*.csv")):
        sym = p.stem
        try:
            df = load_series(p)
        except Exception:
            continue
        if len(df) < MIN_SERIES:
            continue

        ret, level = to_returns_and_level(df["VALUE"])
        if ret.empty or level.empty:
            continue

        # --- SIGNAL: momentum sign on level with fallback to daily sign ---
        mom = level.pct_change(MOM_WIN)
        pos1 = np.sign(mom).fillna(0.0)
        pos2 = np.sign(ret).fillna(0.0)
        raw_pos = (0.7*pos1 + 0.3*pos2)

        # Smooth and clip positions
        pos = pd.Series(raw_pos, index=ret.index).ewm(span=SMOOTH_SPAN, adjust=False).mean().clip(-1,1).fillna(0.0)

        # Cap daily position change (transaction-slowing)
        dpos = pos.diff().fillna(0.0)
        dpos = dpos.clip(lower=-MAX_DPOS, upper=MAX_DPOS)
        pos  = (dpos.cumsum() + pos.iloc[0]).clip(-1,1)

        # Safety caps (avoid crazy math)
        ret = pd.Series(ret).replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-CAP_RET, CAP_RET)
        pos = pd.Series(pos).replace([np.inf,-np.inf], 0.0).fillna(0.0).clip(-1,1)

        frames.append(pd.DataFrame({
            "DATE": df["DATE"].iloc[:len(ret)],
            "ret": ret.values,
            "pos": pos.values,
            "asset": sym
        }))
        used += 1

    if not frames:
        raise SystemExit("No assets produced oscillator signals (check data/*.csv).")

    all_df = pd.concat(frames, ignore_index=True)
    R = all_df.pivot(index="DATE", columns="asset", values="ret").sort_index().fillna(0.0)
    P = all_df.pivot(index="DATE", columns="asset", values="pos").reindex(R.index).fillna(0.0)

    # Equal-weight (robust)
    n_assets = R.shape[1]
    if n_assets < 2:
        raise SystemExit("Need >=2 assets for oscillator portfolio.")
    W = pd.DataFrame(1.0/n_assets, index=R.index, columns=R.columns)

    # Gross and turnover
    exposure = P * W
    gross_ret = (exposure * R).sum(axis=1).fillna(0.0)

    # Daily turnover: sum abs change in exposure (across assets)
    turnover = exposure.diff().abs().sum(axis=1).fillna(0.0)

    # Costs in decimal (COST_BPS * turnover)
    cost = COST_BPS * turnover
    net_ret = gross_ret - cost

    # Safe equity / dd
    eq_g = (1.0 + gross_ret).cumprod()
    eq_n = (1.0 + net_ret).cumprod()
    peak_g = pd.concat([eq_g, pd.Series(1.0, index=eq_g.index)], axis=1).max(axis=1).cummax()
    peak_n = pd.concat([eq_n, pd.Series(1.0, index=eq_n.index)], axis=1).max(axis=1).cummax()
    dd_g = float((eq_g/peak_g - 1.0).min())
    dd_n = float((eq_n/peak_n - 1.0).min())

    sharp_g = ann_sharpe(gross_ret)
    sharp_n = ann_sharpe(net_ret)
    hit_g = float((gross_ret > 0).mean())
    hit_n = float((net_ret  > 0).mean())

    # Save series
    out = pd.DataFrame({
        "DATE": R.index, 
        "ret_gross": gross_ret.values,
        "ret_net":   net_ret.values,
        "turnover":  turnover.values,
        "eq_gross":  eq_g.values,
        "eq_net":    eq_n.values
    })
    out.to_csv(RUNS/"osc_portfolio_costed.csv", index=False)

    # Save summary
    RUNS.joinpath("osc_portfolio_costed_summary.json").write_text(json.dumps({
        "gross": {"sharpe": sharp_g, "hit": hit_g, "maxdd": dd_g},
        "net":   {"sharpe": sharp_n, "hit": hit_n, "maxdd": dd_n},
        "params": {
            "cost_bps": COST_BPS*10000.0, "max_dpos": MAX_DPOS,
            "smooth_span": SMOOTH_SPAN, "mom_win": MOM_WIN,
            "cap_ret": CAP_RET, "assets_used": n_assets
        }
    }, indent=2))

    print("\nOSCILLATOR (COSTED)")
    print("===================")
    print(f"Gross  Sharpe: {sharp_g:.3f} | Hit: {hit_g:.3f} | MaxDD: {dd_g:.3f}")
    print(f" Net   Sharpe: {sharp_n:.3f} | Hit: {hit_n:.3f} | MaxDD: {dd_n:.3f}")
    print(f"(Assets used: {n_assets} | Cost bps: {COST_BPS*10000:.1f} | Max dPos: {MAX_DPOS:.2f})")
    print(f"Saved: {RUNS/'osc_portfolio_costed.csv'} and osc_portfolio_costed_summary.json")
