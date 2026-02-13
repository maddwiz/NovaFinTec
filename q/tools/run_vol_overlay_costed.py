#!/usr/bin/env python3
"""
run_vol_overlay_costed.py  (v4.1: universe controls)

Conservative, realistic overlay with costs. Now you can choose the universe:

Env options (set BEFORE running):
  USE_FREEZE=1         -> use runs_plus/universe_freeze.json (default)
  USE_FREEZE=0         -> ignore freeze, use all data/*.csv
  TOPN_FROM_WF=31      -> ignore both, take top-N by sharpe from runs_plus/walk_forward_table_plus.csv

Other params:
  CALM_TH=0.30     # annualized realized vol threshold for "calm"
  STRESS_CUT=0.7   # exposure multiplier in stress
  MOM_WIN=21       # momentum lookback (days)
  VOL_WIN=21       # realized vol window
  COST_BPS=5.0     # cost per 1.0 notional turnover (bps)
"""

import os, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True, parents=True)

def load_series(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "DATE" not in df.columns or "VALUE" not in df.columns:
        cols = {c.lower(): c for c in df.columns}
        dcol = cols.get("date", list(df.columns)[0])
        vcol = cols.get("value", list(df.columns)[-1])
        df = pd.DataFrame({"DATE": df[dcol], "VALUE": df[vcol]})
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna().sort_values("DATE")
    return df

def daily_returns(price: pd.Series):
    return price.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0)

def realized_vol(r, window=21):
    return r.rolling(window).std() * np.sqrt(252)

def pick_universe():
    use_freeze = int(os.getenv("USE_FREEZE", "1"))
    topn = os.getenv("TOPN_FROM_WF", "").strip()
    if topn:
        try:
            n = int(topn)
        except Exception:
            n = 0
        table = RUNS / "walk_forward_table_plus.csv"
        if not table.exists():
            raise SystemExit("TOPN_FROM_WF set but runs_plus/walk_forward_table_plus.csv not found.")
        df = pd.read_csv(table)
        if "asset" not in df.columns or "sharpe" not in df.columns:
            raise SystemExit("walk_forward_table_plus.csv missing 'asset' or 'sharpe'.")
        dff = df.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False)
        uni = dff["asset"].astype(str).head(n).tolist()
        return set(uni), f"topN_from_WF:{n}"
    if use_freeze:
        freeze = RUNS / "universe_freeze.json"
        if freeze.exists():
            try:
                uni = set(json.loads(freeze.read_text()).get("universe", []))
                return uni, "freeze"
            except Exception:
                pass
    # fallback: all data/*.csv
    all_syms = sorted([p.stem for p in DATA.glob("*.csv")])
    return set(all_syms), "all_data"

if __name__ == "__main__":
    CALM_TH   = float(os.getenv("CALM_TH", "0.30"))
    STRESS_CUT= float(os.getenv("STRESS_CUT", "0.7"))
    MOM_WIN   = int(os.getenv("MOM_WIN", "21"))
    VOL_WIN   = int(os.getenv("VOL_WIN", "21"))
    COST_BPS  = float(os.getenv("COST_BPS", "5.0")) / 10000.0  # bps -> dec

    universe, src = pick_universe()

    # Build per-asset returns and positions
    pnl_frames, pos_frames = [], []
    for p in sorted(DATA.glob("*.csv")):
        sym = p.stem
        if sym not in universe:
            continue
        try:
            df = load_series(p)
        except Exception:
            continue
        if len(df) < max(150, MOM_WIN + VOL_WIN + 5):
            continue
        px = pd.to_numeric(df["VALUE"], errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
        if px.empty: 
            continue

        ret = daily_returns(px)
        vol = realized_vol(ret, VOL_WIN)

        mom = px.pct_change(MOM_WIN)
        direction = np.sign(mom).fillna(0.0)   # {-1,0,1}

        gate = pd.Series(1.0, index=ret.index)
        gate[vol > CALM_TH] = STRESS_CUT

        pos = (direction * gate).clip(-1.0, 1.0)   # position in [-1,1]
        pnl = (pos * ret).rename(sym)

        pnl_frames.append(pd.DataFrame({"DATE": df["DATE"].iloc[:len(pnl)], sym: pnl.values}))
        pos_frames.append(pd.DataFrame({"DATE": df["DATE"].iloc[:len(pos)],  sym: pos.values}))

    if not pnl_frames:
        raise SystemExit("No assets available for overlay (check data/ or universe selection).")

    # Merge wide
    wide = pnl_frames[0]
    for f in pnl_frames[1:]:
        wide = pd.merge(wide, f, on="DATE", how="outer")
    wide = wide.sort_values("DATE").fillna(0.0).set_index("DATE")

    wide_pos = pos_frames[0]
    for f in pos_frames[1:]:
        wide_pos = pd.merge(wide_pos, f, on="DATE", how="outer")
    wide_pos = wide_pos.sort_values("DATE").fillna(0.0).set_index("DATE")

    # Equal-weight across available assets each day
    assets = list(wide.columns)
    if len(assets) < 2:
        raise SystemExit("Need at least 2 assets for an overlay.")
    w = pd.DataFrame(1.0 / len(assets), index=wide.index, columns=assets)

    # Portfolio returns (no vol-targeting, scale=1)
    port_ret_gross = (wide * w).sum(axis=1)

    # Turnover & costs: exposure = w * pos (scale=1)
    exposure = w * wide_pos[assets]
    turnover = exposure.diff().abs().sum(axis=1).fillna(0.0)
    cost = COST_BPS * turnover

    port_ret_net = port_ret_gross - cost

    # Stats
    def ann_sharpe(r):
        r = pd.Series(r).dropna()
        s = r.std()
        if s == 0 or np.isnan(s): return 0.0
        return float((r.mean() / s) * np.sqrt(252.0))

    eq_g = (1 + port_ret_gross).cumprod()
    eq_n = (1 + port_ret_net).cumprod()

    sharpe_g = ann_sharpe(port_ret_gross)
    sharpe_n = ann_sharpe(port_ret_net)
    hit_g = float((port_ret_gross > 0).mean())
    hit_n = float((port_ret_net > 0).mean())
    dd_g  = float((eq_g / eq_g.cummax() - 1.0).min())
    dd_n  = float((eq_n / eq_n.cummax() - 1.0).min())

    (RUNS / "vol_overlay_costed.csv").write_text(
        pd.DataFrame({
            "DATE": port_ret_net.index,
            "ret_gross": port_ret_gross.values,
            "ret_net": port_ret_net.values,
            "eq_gross": eq_g.values,
            "eq_net": eq_n.values,
            "turnover": turnover.values
        }).to_csv(index=False)
    )

    RUNS.joinpath("vol_overlay_costed_summary.json").write_text(json.dumps({
        "gross": {"sharpe": sharpe_g, "hit": hit_g, "maxdd": dd_g},
        "net":   {"sharpe": sharpe_n, "hit": hit_n, "maxdd": dd_n},
        "params": {
            "calm_th": CALM_TH, "stress_cut": STRESS_CUT,
            "mom_win": MOM_WIN, "vol_win": VOL_WIN,
            "cost_bps": COST_BPS * 10000.0,
            "universe_source": src,
            "universe_size": len(assets)
        }
    }, indent=2))

    print("\nVOLATILITY OVERLAY (COSTED, v4.1)")
    print("==================================")
    print(f"Gross Sharpe: {sharpe_g:.3f} | Hit: {hit_g:.3f} | MaxDD: {dd_g:.3f}")
    print(f" Net  Sharpe: {sharpe_n:.3f} | Hit: {hit_n:.3f} | MaxDD: {dd_n:.3f}")
    print(f"(Universe source: {src} | size: {len(assets)})")
    print(f"Saved: {RUNS/'vol_overlay_costed.csv'} and vol_overlay_costed_summary.json")
