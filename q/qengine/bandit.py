import numpy as np
import pandas as pd

class ExpWeightsBandit:
    """
    Exponentially Weighted Forecaster (a.k.a. Hedge) for static ensemble weights.
    We "fit" on TRAIN only (directional reward), then FREEZE weights for TEST.
    """
    def __init__(self, eta: float = 0.4):
        self.eta = float(eta)
        self.weights_ = None
        self.keys_ = None

    def fit(self, signals: dict, returns: pd.Series):
        """
        signals: dict[name -> pd.Series of signal direction (-1/0/+1)]
        returns: pd.Series of realized returns on the same index
        """
        keys = list(signals.keys())
        S = pd.concat(signals, axis=1).dropna()
        R = returns.reindex(S.index).fillna(0.0)

        K = len(keys)
        w = np.ones(K) / max(K, 1)

        # online update: reward = +1 if yesterday's sign matches today's return, else -1
        for i in range(1, len(S)):
            sign_pred = np.sign(S.iloc[i-1].values)  # predicted direction at t-1
            r = R.iloc[i]
            sign_real = 1.0 if r > 0 else (-1.0 if r < 0 else 0.0)
            rewards = np.where(sign_pred == sign_real, 1.0, -1.0)

            w = w * np.exp(self.eta * rewards)
            s = w.sum()
            if s > 0:
                w = w / s
            else:
                w = np.ones(K) / max(K, 1)

        self.weights_ = {keys[k]: float(w[k]) for k in range(K)}
        self.keys_ = keys
        return self

    def get_weights(self):
        if self.weights_ is None:
            return {}
        return dict(self.weights_)
