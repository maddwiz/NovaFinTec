import argparse, pathlib, json
import numpy as np
import pandas as pd
from qmods.meta_council import momentum_signal, meanrev_signal, carry_signal
from qmods.council_train import train_council
from qmods.dna import fft_topk_dna
from qmods.heartbeat import heartbeat_bpm
from qmods.drift import rolling_dna_drift
from qmods.dreams import save_dream_png, save_dream_video
from qmods.log import append_growth_log

def _safe_sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(252)) if s > 1e-12 else 0.0

def _max_drawdown_from_equity(eq: np.ndarray) -> float:
    eq = np.asarray(eq, float)
    eq = np.nan_to_num(eq, nan=1.0, posinf=1.0, neginf=1.0)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq / np.where(peak == 0, 1.0, peak) - 1.0
    return float(dd.min())

def eval_strategy(close: pd.Series, pos: np.ndarray, cost_bps: float = 1.0):
    # returns & position
    ret = close.pct_change().to_numpy()
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)

    pos = np.asarray(pos, float)
    pos = np.nan_to_num(pos, nan=0.0)
    pos_lag = np.roll(pos, 1)
    pos_lag[0] = 0.0  # no position before first return

    # transaction costs (per unit change in position)
    cost = (cost_bps / 10000.0)
    turnover = np.abs(np.diff(np.r_[0.0, pos])).astype(float)
    tcost = turnover * cost

    # strategy returns after cost
    strat = pos_lag * ret - tcost
    strat = np.nan_to_num(strat, nan=0.0)

    # equity curve starts at 1.0
    eq = np.cumprod(1.0 + strat)

    # metrics
    hit = float(np.mean(np.sign(pos_lag) == np.sign(ret))) if ret.size else 0.0
    sh = _safe_sharpe(strat)
    mdd = _max_drawdown_from_equity(eq)
    return hit, sh, mdd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cost_bps", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()

    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pathlib.Path(args.data)/args.asset, parse_dates=["Date"], index_col="Date")
    close = df["Close"].astype(float)

    # === base signals ===
    mom = momentum_signal(close)
    mr  = meanrev_signal(close)
    car = carry_signal(close)

    # === CV weights (time-ordered, no leakage) ===
    w = train_council(close, cost_bps=args.cost_bps, kfold=5, min_hist=252*3)
    meta = w[0]*mom + w[1]*mr + w[2]*car
    pos  = np.tanh(meta)  # continuous sizing in [-1, 1]

    # === evaluate after costs ===
    hit, sh, mdd = eval_strategy(close, pos, cost_bps=args.cost_bps)

    # === organism bits ===
    dna = fft_topk_dna(close.values)
    drift = rolling_dna_drift(close, 126)
    bpm = heartbeat_bpm(close)

    # === dreams (PNG + GIF/MP4) ===
    save_dream_png(close.values, outdir/"dream.png")
    save_dream_video(close.values, outdir, frames=args.frames, step=5, fps=12)

    # === save summary ===
    result = {
        "asset": args.asset,
        "weights": {"mom": float(w[0]), "mr": float(w[1]), "carry": float(w[2])},
        "hit_rate": float(hit),
        "sharpe": float(sh),
        "max_dd": float(mdd),
        "dna": dna,
        "dna_drift_pct": float(drift.ffill().iloc[-1]) if drift.notna().any() else None,
        "heartbeat_bpm_latest": float(bpm.ffill().iloc[-1]) if bpm.notna().any() else None,
    }
    (outdir/"summary.json").write_text(json.dumps(result, indent=2))
    append_growth_log(result, pathlib.Path("GROWTH_LOG.md"))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
