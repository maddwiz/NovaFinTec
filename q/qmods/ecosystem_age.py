from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_health_series(hive_signals: pd.DataFrame, hive: str) -> pd.Series:
    h = hive_signals[hive_signals["HIVE"] == hive].copy()
    h = h.sort_values("DATE")
    if "hive_health" in h.columns:
        s = pd.to_numeric(h["hive_health"], errors="coerce")
    else:
        sig = pd.to_numeric(h.get("hive_signal", 0.0), errors="coerce").fillna(0.0)
        mu = sig.rolling(63, min_periods=20).mean()
        sd = sig.rolling(63, min_periods=20).std(ddof=1).replace(0, np.nan)
        s = np.tanh((mu / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0) / 2.0)
    s.index = pd.to_datetime(h["DATE"], errors="coerce")
    return s.sort_index().dropna()


def govern_hive_weights(
    cross_hive_weights: pd.DataFrame,
    hive_signals: pd.DataFrame,
    half_life_days: int = 63,
    atrophy_floor: float = 0.10,
    inertia: float = 0.85,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply ecosystem aging logic:
    - aging/vitality from EW health
    - atrophy floor to avoid hard zeroing
    - inertia to avoid churn
    """
    w = cross_hive_weights.copy()
    w["DATE"] = pd.to_datetime(w["DATE"], errors="coerce")
    w = w.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
    hives = [c for c in w.columns if c != "DATE"]
    if not hives:
        return w, {"hives": [], "events": []}

    hs = hive_signals.copy()
    hs["DATE"] = pd.to_datetime(hs["DATE"], errors="coerce")
    hs = hs.dropna(subset=["DATE"]).sort_values(["DATE", "HIVE"])

    governed = w[hives].astype(float).copy()
    events = []
    vitality_latest = {}

    alpha = 1.0 - np.exp(np.log(0.5) / max(2.0, float(half_life_days)))

    for hive in hives:
        s = _safe_health_series(hs, hive).reindex(w["DATE"]).ffill().fillna(0.0)
        ema = s.ewm(alpha=alpha, adjust=False).mean()
        trend = (ema - ema.shift(10)).fillna(0.0)
        vitality = np.clip(0.55 + 0.35 * ema + 0.10 * np.tanh(3.0 * trend), atrophy_floor, 1.25)
        vitality_latest[hive] = float(vitality.iloc[-1]) if len(vitality) else float(atrophy_floor)
        governed[hive] = governed[hive].values * vitality.values

        # Event tags
        if vitality_latest[hive] < 0.35:
            events.append({"hive": hive, "event": "atrophy", "score": vitality_latest[hive]})
        if float(governed[hive].iloc[-1]) > 0.55 and float(s.std(ddof=1) or 0.0) > 0.30:
            events.append({"hive": hive, "event": "split_candidate", "score": float(governed[hive].iloc[-1])})

    # Normalize each date
    row_sum = governed.sum(axis=1).replace(0.0, np.nan)
    governed = governed.div(row_sum, axis=0).fillna(1.0 / max(1, len(hives)))

    # Inertia smoothing
    gov_arr = governed.values
    for t in range(1, len(gov_arr)):
        gov_arr[t] = inertia * gov_arr[t - 1] + (1.0 - inertia) * gov_arr[t]
        s = gov_arr[t].sum()
        if s > 0:
            gov_arr[t] /= s
    governed = pd.DataFrame(gov_arr, columns=hives)

    # Fusion candidates: highly correlated hive-signal pairs
    piv = hs.pivot(index="DATE", columns="HIVE", values="hive_signal").reindex(columns=hives).fillna(0.0)
    if piv.shape[1] >= 2:
        corr = piv.corr()
        for i, h1 in enumerate(hives):
            for h2 in hives[i + 1 :]:
                c = float(corr.loc[h1, h2]) if h1 in corr.index and h2 in corr.columns else 0.0
                if c > 0.90:
                    events.append({"hive": f"{h1}+{h2}", "event": "fusion_candidate", "score": c})

    out = pd.concat([w[["DATE"]], governed], axis=1)
    summary = {
        "hives": hives,
        "half_life_days": int(half_life_days),
        "atrophy_floor": float(atrophy_floor),
        "inertia": float(inertia),
        "latest_raw_weights": {h: float(w[h].iloc[-1]) for h in hives},
        "latest_governed_weights": {h: float(out[h].iloc[-1]) for h in hives},
        "latest_vitality": vitality_latest,
        "events": events,
    }
    return out, summary
