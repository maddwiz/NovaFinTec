import json, pathlib
import numpy as np, pandas as pd
from qmods.io import load_close
from qmods.meta_council import momentum_signal, meanrev_signal, carry_signal

RUNS = pathlib.Path("runs_plus")
DATA = pathlib.Path("data")
COST_BPS = 1.0

def eval_strat(close: pd.Series, pos: np.ndarray, cost_bps=1.0):
    r = np.diff(np.log(np.maximum(close.values, 1e-12)))
    ret = np.r_[0.0, r]
    ret = np.clip(ret, -0.20, 0.20)

    pos = np.nan_to_num(np.asarray(pos, float), nan=0.0)
    pos_lag = np.roll(pos, 1); pos_lag[0] = 0.0
    cost = (cost_bps/10000.0)
    turnover = np.abs(np.diff(np.r_[0.0, pos]))

    strat = pos_lag*ret - turnover*cost
    strat_s = pd.Series(strat, index=close.index)
    eq_s = (1.0 + strat_s).cumprod()
    s = strat_s.std(ddof=1)
    sharpe = float(strat_s.mean()/s*np.sqrt(252)) if s>1e-12 else 0.0
    hit = float((np.sign(strat_s.shift(1)) == np.sign(strat_s)).mean())
    mdd = float((eq_s/eq_s.cummax() - 1.0).min())
    return strat_s, hit, sharpe, mdd, eq_s

assets = []
for p in RUNS.glob("*/summary.json"):
    a = p.parent.name
    if (DATA/f"{a}.csv").exists():
        assets.append(a)
assets = sorted(set(assets))

per_asset = {}
for a in assets:
    sfile = RUNS/a/"summary.json"
    dfile = DATA/f"{a}.csv"
    if not sfile.exists() or not dfile.exists(): continue

    m = json.loads(sfile.read_text())
    w = m.get("weights", {"mom":0.4,"mr":0.3,"carry":0.3})

    close = load_close(dfile)
    mom = momentum_signal(close); mr = meanrev_signal(close); car = carry_signal(close)
    meta = w["mom"]*mom + w["mr"]*mr + w["carry"]*car
    pos  = np.tanh(meta)

    strat, hit, sh, dd, eq = eval_strat(close, pos, cost_bps=COST_BPS)
    per_asset[a] = {"strat": strat, "hit": hit, "sharpe": sh, "mdd": dd, "eq": eq}

if not per_asset:
    print("No assets for portfolio."); raise SystemExit

panel = pd.concat([v["strat"] for v in per_asset.values()], axis=1).fillna(0.0)
panel.columns = list(per_asset.keys())

port_ret = panel.mean(axis=1)
port_eq  = (1.0 + port_ret).cumprod()
port_mdd = float((port_eq/port_eq.cummax() - 1.0).min())
s = port_ret.std(ddof=1)
port_sh = float(port_ret.mean()/s*np.sqrt(252)) if s>1e-12 else 0.0
port_hit= float((np.sign(port_ret.shift(1))==np.sign(port_ret)).mean())

summary = {"hit": port_hit, "sharpe": port_sh, "max_dd": port_mdd}
(RUNS/"portfolio.json").write_text(json.dumps(summary, indent=2))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(9,4))
plt.plot(port_eq.index, port_eq.values, label="Equal-weight portfolio")
plt.legend(); plt.title("Portfolio equity (auto)"); plt.tight_layout()
plt.savefig(RUNS/"portfolio.png", dpi=120)
plt.close()

print("PORTFOLIO", summary)
