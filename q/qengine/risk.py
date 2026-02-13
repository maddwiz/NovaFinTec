
import numpy as np, pandas as pd

def realized_vol(returns: pd.Series, look=20, eps=1e-12):
    return returns.rolling(look).std().fillna(0.0) + eps

def vol_target_sizer(returns: pd.Series, target_annu_vol=0.20, look=20, cap=0.25):
    target_daily = target_annu_vol / (252.0 ** 0.5)
    rv = realized_vol(returns, look=look)
    raw = target_daily / rv.replace(0, np.nan)
    raw = raw.clip(lower=0.0).fillna(0.0)
    pos = raw.clip(upper=cap)
    return pos

def drawdown_brake(equity_curve: pd.Series, thresh=-0.10):
    peak = equity_curve.cummax()
    dd = equity_curve/peak - 1.0
    brake = (dd - thresh) / (0 - thresh)
    brake = brake.clip(0,1)
    return 1.0 - brake

def entropy_flip_budget(signal: pd.Series, max_flips_per_60d=6):
    s = signal.fillna(0).astype(float).copy()
    flips = (s.diff().abs() > 1e-9) & (s != 0)
    window = 60
    budget_used = flips.rolling(window).sum().fillna(0.0)
    allowed = (budget_used <= max_flips_per_60d).astype(float)
    out = s.copy()
    for i in range(1, len(out)):
        if allowed.iloc[i] < 1.0:
            out.iloc[i] = out.iloc[i-1]
    return out

def apply_risk(signal: pd.Series, close: pd.Series, target_vol=0.20, cap=0.25, dd_th=-0.10, max_flips_60=6):
    r = np.log(close).diff().fillna(0.0)
    size = vol_target_sizer(r, target_annu_vol=target_vol, look=20, cap=cap)
    sig = entropy_flip_budget(signal, max_flips_per_60d=max_flips_60)
    pos = sig * size
    strat_r = pos.shift(1).fillna(0.0) * r
    eq = (1.0 + strat_r).cumprod()
    brake = drawdown_brake(eq, thresh=dd_th)
    pos = pos * brake
    return pos, strat_r, eq
