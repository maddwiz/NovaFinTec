import numpy as np, pandas as pd
from .meta_council import momentum_signal, meanrev_signal, carry_signal

def _sharpe(ret: np.ndarray) -> float:
    s = np.nanstd(ret, ddof=1)
    return float(np.nanmean(ret)/s*np.sqrt(252)) if s > 1e-12 else 0.0

def train_council(close: pd.Series,
                  cost_bps: float = 1.0,
                  kfold: int = 5,
                  min_hist: int = 252*3):
    """
    Time-ordered k-fold: split the series into k folds in time; for each fold,
    train on all *prior* data only (no peeking).
    Grid search weights in {0..1} step .1, sum to 1.
    Objective: maximize Sharpe after costs on validation fold.
    """
    n = len(close)
    if n < min_hist:
        return (0.4, 0.3, 0.3)  # fallback

    # build base signals once
    mom = momentum_signal(close)         # numpy array
    mr  = meanrev_signal(close)
    car = carry_signal(close)
    ret = close.pct_change().values
    cost = (cost_bps/10000.0)

    # fold boundaries
    idx = np.arange(n)
    folds = np.array_split(idx[min_hist:], kfold)

    best_w = (0.4, 0.3, 0.3)
    best_score = -1e9

    grid = np.linspace(0, 1, 11)  # 0.0..1.0 step .1
    weights = []
    for a in grid:
        for b in grid:
            c = 1.0 - a - b
            if c < 0 or c > 1: continue
            weights.append((a, b, c))

    for fold in folds:
        val_start, val_end = fold[0], fold[-1] + 1
        train_end = val_start  # use strictly prior history
        # use only non-empty train
        if train_end - 1 <= 0: continue

        r_val_best = -1e9
        w_val_best = best_w

        for w in weights:
            s_train = w[0]*mom[:train_end] + w[1]*mr[:train_end] + w[2]*car[:train_end]
            s_val   = w[0]*mom[val_start:val_end] + w[1]*mr[val_start:val_end] + w[2]*car[val_start:val_end]
            # strategy = lagged position * returns - turn cost
            pos_train = np.tanh(s_train)
            pos_val   = np.tanh(s_val)
            strat_val = np.nan_to_num(np.roll(pos_val, 1))*ret[val_start:val_end] - np.abs(np.diff(np.r_[0, pos_val]))*cost
            sh_val = _sharpe(strat_val)
            if sh_val > r_val_best:
                r_val_best = sh_val
                w_val_best = w

        if r_val_best > best_score:
            best_score = r_val_best
            best_w = w_val_best

    return best_w
