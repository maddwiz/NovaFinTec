import json, pathlib
import numpy as np, pandas as pd
from qmods.meta_council import momentum_signal, meanrev_signal, carry_signal

ASSETS = ["IWM","RSP","LQD_TR","HYG_TR"]
DATA = pathlib.Path("data")
RUNS = pathlib.Path("runs_cv")
COST_BPS = 1.0

def eval_strat(close: pd.Series, pos: np.ndarray, cost_bps=1.0):
    ret = close.pct_change().to_numpy()
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    pos = np.nan_to_num(np.asarray(pos, float), nan=0.0)
    pos_lag = np.roll(pos, 1); pos_lag[0] = 0.0
    cost = (cost_bps/10000.0)
    turnover = np.abs(np.diff(np.r_[0.0, pos]))
    strat = pos_lag*ret - turnover*cost
    eq = np.cumprod(1.0 + strat)
    s = strat.std(ddof=1)
    sharpe = float(strat.mean()/s*np.sqrt(252)) if s > 1e-12 else 0.0
    hit = float(np.mean(np.sign(pos_lag)==np.sign(ret))) if ret.size else 0.0
    mdd = float((eq/np.maximum.accumulate(eq)-1.0).min())
    return pd.Series(strat, index=close.index), hit, sharpe, mdd

per_asset = {}
for a in ASSETS:
    sfile = RUNS/a/"summary.json"
    dfile = DATA/f"{a}.csv"
    if not sfile.exists() or not dfile.exists(): continue
    m = json.loads(sfile.read_text())
    w = m.get("weights", {"mom":0.4,"mr":0.3,"carry":0.3})
    df = pd.read_csv(dfile, parse_dates=["Date"], index_col="Date")
    close = df["Close"].astype(float)
    mom = momentum_signal(close)
    mr  = meanrev_signal(close)
    car = carry_signal(close)
    meta = w["mom"]*mom + w["mr"]*mr + w["carry"]*car
    pos  = np.tanh(meta)
    strat, hit, sh, dd = eval_strat(close, pos, cost_bps=COST_BPS)
    per_asset[a] = {"strat": strat, "hit": hit, "sharpe": sh, "mdd": dd}

# align and equal-weight
rets = [per_asset[a]["strat"] for a in per_asset]
if not rets:
    print("No assets found."); raise SystemExit
panel = pd.concat(rets, axis=1).fillna(0.0)
panel.columns = list(per_asset.keys())
port_ret = panel.mean(axis=1)
eq = (1.0 + port_ret).cumprod()
mdd = (eq/eq.cummax() - 1.0).min()
s = port_ret.std(ddof=1)
sh = float(port_ret.mean()/s*np.sqrt(252)) if s>1e-12 else 0.0
hit = float((np.sign(port_ret.shift(1)).eq(np.sign(port_ret))).mean())

print("PER-ASSET")
print("asset    hit     sharpe   maxDD")
for a, v in per_asset.items():
    print(f"{a:7s} {v['hit']:.3f}  {v['sharpe']:.3f}  {v['mdd']:.4f}")

print("\nPORTFOLIO (equal-weight)")
print(f"hit {hit:.3f}  sharpe {sh:.3f}  maxDD {mdd:.4f}")
