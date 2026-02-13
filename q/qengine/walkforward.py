
import numpy as np, pandas as pd
from .utils import annualize_sharpe, max_drawdown

def rolling_folds(df: pd.DataFrame, start_window=252*3, step=63):
    n = len(df)
    i = start_window
    while i + step <= n:
        train = np.zeros(n, dtype=bool)
        test = np.zeros(n, dtype=bool)
        train[:i] = True
        test[i:i+step] = True
        yield train, test
        i += step

def evaluate_strategy(close: pd.Series, position: pd.Series):
    r = np.log(close).diff().fillna(0.0)
    strat = position.shift(1).fillna(0.0) * r
    eq = (1.0 + strat).cumprod()
    hit = (np.sign(position.shift(1).fillna(0.0)) == np.sign(r)).mean()
    sharpe = annualize_sharpe(strat.values) if strat.std(ddof=1) > 0 else 0.0
    mdd = max_drawdown(eq.values)
    return {"hit": float(hit), "sharpe": float(sharpe), "max_dd": float(mdd)}

def walkforward(df_features: pd.DataFrame, make_position_fn):
    metrics = []
    positions = []
    for train_mask, test_mask in rolling_folds(df_features):
        train_df = df_features.iloc[train_mask]
        test_df  = df_features.iloc[test_mask]
        pos = make_position_fn(train_df, test_df)
        positions.append(pos)
        res = evaluate_strategy(test_df["Close"], pos)
        metrics.append(res)
    if len(positions)==0:
        return pd.DataFrame(), pd.DataFrame()
    full_pos = pd.concat(positions).sort_index()
    full_metrics = pd.DataFrame(metrics)
    return full_pos, full_metrics
