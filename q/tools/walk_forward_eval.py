# tools/walk_forward_eval.py — simple expanding-window walk-forward
import argparse, json
import numpy as np, pandas as pd
from pathlib import Path
from qmods.io import load_close
from qmods.meta_council import momentum_signal, meanrev_signal, carry_signal

def safe_equity(close, pos, cost_bps):
    r = np.diff(np.log(np.maximum(close.values, 1e-12)))
    ret = np.r_[0.0, np.clip(r, -0.20, 0.20)]
    pos = np.nan_to_num(np.asarray(pos,float), nan=0.0)
    pos_lag = np.roll(pos,1); pos_lag[0]=0.0
    cost = cost_bps/10000.0
    turnover = np.abs(np.diff(np.r_[0.0,pos]))
    strat = pos_lag*ret - turnover*cost
    s = pd.Series(strat, index=close.index)
    eq = (1+s).cumprod()
    hit = float((np.sign(s.shift(1))==np.sign(s)).mean())
    sd = s.std(ddof=1)
    sh = float(s.mean()/sd*np.sqrt(252)) if sd>1e-12 else 0.0
    mdd = float((eq/eq.cummax()-1.0).min())
    return s, hit, sh, mdd, eq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--asset", required=True)  # e.g., SPY.csv
    ap.add_argument("--cost_bps", type=float, default=1.0)
    ap.add_argument("--train_min", type=int, default=504)   # ~2y
    ap.add_argument("--step", type=int, default=21)         # ~1m steps
    args = ap.parse_args()

    close = load_close(Path(args.data)/args.asset)
    mom = momentum_signal(close)
    mr  = meanrev_signal(close)
    car = carry_signal(close)

    # CV grid for weights (same as pipeline)
    grid = [(0.8,0.2,0.0),(0.6,0.3,0.1),(0.4,0.4,0.2),(0.3,0.5,0.2),(0.2,0.6,0.2)]

    pos_all = np.zeros(len(close))
    for t in range(args.train_min, len(close), args.step):
        best=None; best_sh=-9e9
        for wm,wr,wc in grid:
            meta = wm*mom[:t] + wr*mr[:t] + wc*car[:t]
            pos  = np.tanh(meta)
            strat,_,sh,_,_ = safe_equity(close.iloc[:t], pos, args.cost_bps)
            if sh>best_sh:
                best_sh=sh; best=(wm,wr,wc)
        wm,wr,wc = best
        meta_f = wm*mom[:t+args.step] + wr*mr[:t+args.step] + wc*car[:t+args.step]
        pos_all[:t+args.step] = np.tanh(meta_f)

    strat, hit, sh, mdd, eq = safe_equity(close, pos_all, args.cost_bps)
    print(f"{args.asset}  walk-forward  hit={hit:.3f}  sharpe={sh:.3f}  maxDD={mdd:.3f}")
    (Path("runs_plus")/Path(args.asset).stem/"wf_metrics.json").write_text(
        json.dumps({"hit":hit,"sharpe":sh,"max_dd":mdd},indent=2)
    )

if __name__ == "__main__":
    main()
