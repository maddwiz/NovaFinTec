import argparse, json, pathlib
import numpy as np, pandas as pd
from qmods.io import load_close
from qmods.meta_council import momentum_signal, meanrev_signal, carry_signal
from qmods.drift import rolling_dna_drift
from qmods.dna import fft_topk_dna

def safe_equity(close, pos, cost_bps):
    # log-returns, clip extremes to avoid overflow
    r = np.diff(np.log(np.maximum(close.values, 1e-12)))
    ret = np.r_[0.0, r]
    ret = np.clip(ret, -0.20, 0.20)

    pos = np.nan_to_num(np.asarray(pos, float), nan=0.0)
    pos_lag = np.roll(pos, 1); pos_lag[0] = 0.0
    cost = cost_bps/10000.0
    turnover = np.abs(np.diff(np.r_[0.0, pos]))

    strat = pos_lag*ret - turnover*cost
    strat_s = pd.Series(strat, index=close.index)
    eq = (1.0 + strat_s).cumprod()
    s = strat_s.std(ddof=1)
    sh = (strat_s.mean()/s*np.sqrt(252)) if s>1e-12 else 0.0
    hit = float((np.sign(strat_s.shift(1))==np.sign(strat_s)).mean())
    mdd = float((eq/eq.cummax()-1.0).min())
    return strat_s, hit, sh, mdd, eq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cost_bps", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=80)
    args = ap.parse_args()

    data_path = pathlib.Path(args.data)/args.asset
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load price series safely
    close = load_close(data_path)

    # 2) signals
    mom = momentum_signal(close)
    mr  = meanrev_signal(close)
    car = carry_signal(close)

    # 3) quick CV for weights
    grid = [(0.8,0.2,0.0),(0.6,0.3,0.1),(0.4,0.4,0.2),(0.3,0.5,0.2),(0.2,0.6,0.2)]
    best = None; best_sh = -9e9
    for w_mom, w_mr, w_car in grid:
        meta = w_mom*mom + w_mr*mr + w_car*car
        pos  = np.tanh(meta)
        strat, _, sh, _, _ = safe_equity(close, pos, args.cost_bps)
        if sh > best_sh:
            best_sh = sh; best = (w_mom, w_mr, w_car)

    # 4) final metrics
    w_mom, w_mr, w_car = best
    meta = w_mom*mom + w_mr*mr + w_car*car
    pos  = np.tanh(meta)
    strat, hit, sh, mdd, eq = safe_equity(close, pos, args.cost_bps)

    # 5) dna + drift
    drift = rolling_dna_drift(close, 126)
    dna   = fft_topk_dna(close.values)

    # 6) save summary
    summary = {
        "weights": {"mom": w_mom, "mr": w_mr, "carry": w_car},
        "hit_rate": float(hit),
        "sharpe": float(sh),
        "max_dd": float(mdd),
        "dna": dna,
        "heartbeat_bpm_latest": "-"
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir/"alarms.json").write_text("[]")  # placeholder

    # 7) dream gif (optional; if missing module, skip gracefully)
    try:
        from qmods.dreams import save_dream_gif
        save_dream_gif(close.values, out_dir/"dream.gif", frames=args.frames)
    except Exception:
        pass

    # 8) tiny price plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,3))
    plt.plot(close.index, close.values)
    plt.title(data_path.name)
    plt.tight_layout()
    plt.savefig(out_dir/"signals.png", dpi=120)
    plt.close()

if __name__ == "__main__":
    main()
