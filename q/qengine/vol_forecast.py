"""HAR-RV volatility forecasting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_har_features(returns: np.ndarray) -> pd.DataFrame:
    r = np.asarray(returns, float).ravel()
    rv = r ** 2
    rv_s = pd.Series(rv)

    rv_d = rv_s
    rv_w = rv_s.rolling(5, min_periods=3).mean()
    rv_m = rv_s.rolling(22, min_periods=10).mean()

    abs_r = np.abs(r)
    abs_s = pd.Series(abs_r)
    med = abs_s.rolling(63, min_periods=20).median()
    mad = (abs_s - med).abs().rolling(63, min_periods=20).median()
    jump_flag = abs_s > (med + 3.0 * 1.4826 * mad)
    jump = np.where(jump_flag.values, rv, 0.0)
    jump_s = pd.Series(jump).rolling(5, min_periods=1).mean()

    rv_neg = np.where(r < 0.0, r ** 2, 0.0)
    rv_pos = np.where(r > 0.0, r ** 2, 0.0)
    rsv = pd.Series(rv_neg - rv_pos).rolling(5, min_periods=3).mean()

    return pd.DataFrame(
        {
            "rv_d": rv_d,
            "rv_w": rv_w,
            "rv_m": rv_m,
            "jump": jump_s,
            "rsv": rsv,
        }
    )


def fit_har(features: pd.DataFrame, target: pd.Series, min_obs: int = 126) -> dict:
    df = features.copy()
    df["target"] = pd.Series(target).values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) < int(max(20, min_obs)):
        return {
            "intercept": 0.0,
            "coefficients": {"rv_d": 0.0, "rv_w": 0.0, "rv_m": 0.0, "jump": 0.0, "rsv": 0.0},
            "r_squared": 0.0,
            "n": 0,
        }

    X = df[["rv_d", "rv_w", "rv_m", "jump", "rsv"]].values.astype(float)
    y = df["target"].values.astype(float)
    X = np.column_stack([np.ones(len(X), dtype=float), X])

    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.zeros(X.shape[1], dtype=float)

    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot

    return {
        "intercept": float(beta[0]),
        "coefficients": {
            "rv_d": float(beta[1]),
            "rv_w": float(beta[2]),
            "rv_m": float(beta[3]),
            "jump": float(beta[4]),
            "rsv": float(beta[5]),
        },
        "r_squared": float(r2),
        "n": int(len(df)),
    }


def forecast_vol(
    returns: np.ndarray,
    train_window: int = 504,
    step: int = 21,
) -> np.ndarray:
    r = np.asarray(returns, float).ravel()
    n = len(r)
    if n == 0:
        return np.zeros(0, dtype=float)

    forecasts = np.full(n, np.nan, dtype=float)
    features = compute_har_features(r)
    rv_d = r ** 2

    tw = int(max(126, train_window))
    st = int(max(1, step))

    for t in range(tw, n, st):
        i0 = max(0, t - tw)
        train_f = features.iloc[i0:t]
        train_target = pd.Series(rv_d[i0 + 1 : t + 1])

        L = min(len(train_f), len(train_target))
        if L < 126:
            continue
        train_f = train_f.iloc[-L:]
        train_target = train_target.iloc[-L:]

        model = fit_har(train_f, train_target)
        if int(model.get("n", 0)) <= 0:
            continue

        coefs = model.get("coefficients", {}) or {}
        beta = np.array(
            [
                float(model.get("intercept", 0.0)),
                float(coefs.get("rv_d", 0.0)),
                float(coefs.get("rv_w", 0.0)),
                float(coefs.get("rv_m", 0.0)),
                float(coefs.get("jump", 0.0)),
                float(coefs.get("rsv", 0.0)),
            ],
            dtype=float,
        )

        end = min(t + st, n)
        for j in range(t, end):
            row = features.iloc[j]
            x = np.array([1.0, row["rv_d"], row["rv_w"], row["rv_m"], row["jump"], row["rsv"]], dtype=float)
            rv_hat = float(np.dot(beta, x))
            rv_hat = max(1e-12, rv_hat)
            forecasts[j] = float(np.sqrt(rv_hat * 252.0))

    return pd.Series(forecasts).ffill().bfill().fillna(0.0).values.astype(float)
