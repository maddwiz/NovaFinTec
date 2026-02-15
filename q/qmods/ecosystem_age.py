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


def _safe_signal_series(hive_signals: pd.DataFrame, hive: str) -> pd.Series:
    h = hive_signals[hive_signals["HIVE"] == hive].copy()
    h = h.sort_values("DATE")
    s = pd.to_numeric(h.get("hive_signal", 0.0), errors="coerce").fillna(0.0)
    s.index = pd.to_datetime(h["DATE"], errors="coerce")
    return s.sort_index().dropna()


def _renorm_row(row: np.ndarray, n: int) -> np.ndarray:
    row = np.asarray(row, float)
    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    row = np.clip(row, 0.0, None)
    s = float(np.sum(row))
    if (not np.isfinite(s)) or s <= 0:
        return np.full(n, 1.0 / max(1, n), dtype=float)
    return row / s


def _redistribute_excess(row: np.ndarray, excess: float, mask: np.ndarray, weights: np.ndarray):
    e = float(max(0.0, excess))
    if e <= 0:
        return
    m = np.asarray(mask, bool)
    if not np.any(m):
        return
    w = np.asarray(weights, float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.where(m, np.clip(w, 0.0, None), 0.0)
    s = float(np.sum(w))
    if s <= 0:
        w = np.where(m, 1.0, 0.0)
        s = float(np.sum(w))
    row += e * (w / (s + 1e-12))


def govern_hive_weights(
    cross_hive_weights: pd.DataFrame,
    hive_signals: pd.DataFrame,
    half_life_days: int = 63,
    atrophy_floor: float = 0.10,
    inertia: float = 0.85,
    atrophy_trigger: float = 0.32,
    atrophy_cap: float = 0.06,
    split_trigger: float = 0.55,
    split_vol_trigger: float = 0.22,
    split_intensity: float = 0.25,
    fusion_corr: float = 0.92,
    fusion_intensity: float = 0.12,
    recovery_slope_trigger: float = 0.015,
    split_cooloff_strength: float = 0.35,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply ecosystem aging logic with actionable controls:
    - vitality scaling from EW health
    - atrophy cap for persistent weak hives (with redistribution)
    - split pressure on dominant volatile hives
    - fusion flow for highly correlated hive pairs
    - inertia smoothing to avoid churn
    """
    w = cross_hive_weights.copy()
    w["DATE"] = pd.to_datetime(w["DATE"], errors="coerce")
    w = w.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)

    # Keep only hive columns; ignore diagnostics such as arb_alpha/arb_inertia.
    hives = [c for c in w.columns if c != "DATE" and not str(c).startswith("arb_")]
    if not hives:
        return w[["DATE"]].copy(), {"hives": [], "events": []}

    hs = hive_signals.copy()
    hs["DATE"] = pd.to_datetime(hs["DATE"], errors="coerce")
    hs = hs.dropna(subset=["DATE"]).sort_values(["DATE", "HIVE"])

    T = len(w)
    H = len(hives)
    # Force a writable dense copy; some pandas/NumPy builds can expose read-only views.
    governed = w[hives].to_numpy(dtype=float, copy=True)
    vitality = np.zeros((T, H), dtype=float)
    vol = np.zeros((T, H), dtype=float)

    events = []
    max_events = 64
    event_counts = {"atrophy_applied": 0, "split_applied": 0, "fusion_applied": 0, "recovery_shielded": 0}
    vitality_latest = {}
    action_pressure = np.zeros(T, dtype=float)

    alpha = 1.0 - np.exp(np.log(0.5) / max(2.0, float(half_life_days)))

    for j, hive in enumerate(hives):
        hs_h = _safe_health_series(hs, hive).reindex(w["DATE"]).ffill().fillna(0.0)
        sig_h = _safe_signal_series(hs, hive).reindex(w["DATE"]).ffill().fillna(0.0)
        ema = hs_h.ewm(alpha=alpha, adjust=False).mean()
        trend = (ema - ema.shift(10)).fillna(0.0)
        vit = np.clip(0.55 + 0.35 * ema.values + 0.10 * np.tanh(3.0 * trend.values), atrophy_floor, 1.25)
        vitality[:, j] = vit
        vol[:, j] = sig_h.rolling(21, min_periods=7).std(ddof=1).fillna(0.0).values
        vitality_latest[hive] = float(vit[-1]) if len(vit) else float(atrophy_floor)
        governed[:, j] = governed[:, j] * vit

    # Correlation map for fusion dynamics.
    piv = hs.pivot(index="DATE", columns="HIVE", values="hive_signal").reindex(columns=hives).fillna(0.0)
    corr = piv.corr() if piv.shape[1] >= 2 else pd.DataFrame(index=hives, columns=hives, data=np.eye(H))
    fusion_pairs = []
    for i, h1 in enumerate(hives):
        for j, h2 in enumerate(hives[i + 1 :], start=i + 1):
            c = float(corr.loc[h1, h2]) if h1 in corr.index and h2 in corr.columns else 0.0
            if c >= fusion_corr:
                fusion_pairs.append((i, j, c))
                if len(events) < max_events:
                    events.append({"hive": f"{h1}+{h2}", "event": "fusion_candidate", "score": c})

    # Apply dynamic actions row-by-row.
    for t in range(T):
        row = governed[t].copy()
        vrow = vitality[t]
        volrow = vol[t]
        actions_t = 0

        # 1) Atrophy caps for weak hives with redistribution.
        for j in range(H):
            prev_v = float(vitality[t - 1, j]) if t > 0 else float(vrow[j])
            dv = float(vrow[j] - prev_v)
            if vrow[j] < atrophy_trigger and row[j] > atrophy_cap:
                if dv > float(recovery_slope_trigger):
                    event_counts["recovery_shielded"] += 1
                    if len(events) < max_events:
                        events.append({"hive": hives[j], "event": "recovery_shielded", "score": float(dv)})
                    continue

                shortfall = float(np.clip((atrophy_trigger - vrow[j]) / max(1e-9, atrophy_trigger), 0.0, 1.0))
                cap_dyn = float(np.clip(atrophy_cap * (1.0 - 0.45 * shortfall), 0.01, atrophy_cap))
                excess = float(row[j] - cap_dyn)
                row[j] = float(cap_dyn)
                mask = np.ones(H, dtype=bool)
                mask[j] = False
                _redistribute_excess(row, excess, mask, vrow)
                event_counts["atrophy_applied"] += 1
                actions_t += 1
                if len(events) < max_events:
                    events.append({"hive": hives[j], "event": "atrophy_applied", "score": float(vrow[j]), "cap": float(cap_dyn)})

        # 2) Split pressure for dominant volatile hive.
        dom = int(np.argmax(row))
        prev_press = float(action_pressure[t - 1]) if t > 0 else 0.0
        split_tr_eff = float(np.clip(split_trigger + 0.08 * prev_press, 0.10, 0.98))
        split_int_eff = float(np.clip(split_intensity * (1.0 - split_cooloff_strength * prev_press), 0.01, 1.0))
        if row[dom] > split_tr_eff and volrow[dom] > split_vol_trigger:
            pressure = float((row[dom] - split_tr_eff) / max(1e-9, 1.0 - split_tr_eff))
            cut = float(np.clip(split_int_eff * pressure * row[dom], 0.0, 0.25))
            if cut > 0:
                row[dom] -= cut
                mask = np.ones(H, dtype=bool)
                mask[dom] = False
                # Prefer lower-correlation recipients when splitting.
                corr_pen = np.ones(H, dtype=float)
                if hives[dom] in corr.index:
                    corr_pen = 1.0 - np.clip(np.abs(corr.loc[hives[dom], hives].values.astype(float)), 0.0, 1.0)
                _redistribute_excess(row, cut, mask, vrow * corr_pen)
                event_counts["split_applied"] += 1
                actions_t += 1
                if len(events) < max_events:
                    events.append({"hive": hives[dom], "event": "split_applied", "score": float(cut)})

        # 3) Fusion flow across highly correlated pairs: move from weaker vitality to stronger.
        for i, j, c in fusion_pairs:
            if row[i] <= 0 or row[j] <= 0:
                continue
            if abs(vrow[i] - vrow[j]) < 1e-6:
                continue
            donor = i if vrow[i] < vrow[j] else j
            recv = j if donor == i else i
            flow_scale = float(np.clip((c - fusion_corr) / max(1e-9, 1.0 - fusion_corr), 0.0, 1.0))
            flow = float(fusion_intensity * flow_scale * min(row[i], row[j]))
            flow = min(flow, row[donor] * 0.6)
            if flow <= 0:
                continue
            row[donor] -= flow
            row[recv] += flow
            event_counts["fusion_applied"] += 1
            actions_t += 1

        governed[t] = _renorm_row(row, H)
        action_pressure[t] = float(actions_t) / float(max(1, H))

    # Inertia smoothing
    ib = float(np.clip(inertia, 0.0, 0.98))
    if ib > 0.0 and T > 1:
        for t in range(1, T):
            governed[t] = ib * governed[t - 1] + (1.0 - ib) * governed[t]
            governed[t] = _renorm_row(governed[t], H)

    out = pd.concat([w[["DATE"]], pd.DataFrame(governed, columns=hives)], axis=1)
    summary = {
        "hives": hives,
        "half_life_days": int(half_life_days),
        "atrophy_floor": float(atrophy_floor),
        "inertia": float(inertia),
        "parameters": {
            "atrophy_trigger": float(atrophy_trigger),
            "atrophy_cap": float(atrophy_cap),
            "split_trigger": float(split_trigger),
            "split_vol_trigger": float(split_vol_trigger),
            "split_intensity": float(split_intensity),
            "fusion_corr": float(fusion_corr),
            "fusion_intensity": float(fusion_intensity),
            "recovery_slope_trigger": float(recovery_slope_trigger),
            "split_cooloff_strength": float(split_cooloff_strength),
        },
        "latest_raw_weights": {h: float(w[h].iloc[-1]) for h in hives},
        "latest_governed_weights": {h: float(out[h].iloc[-1]) for h in hives},
        "latest_vitality": vitality_latest,
        "event_counts": event_counts,
        "action_pressure_mean": float(np.mean(action_pressure)) if len(action_pressure) else 0.0,
        "action_pressure_max": float(np.max(action_pressure)) if len(action_pressure) else 0.0,
        "action_pressure_series": action_pressure.tolist(),
        "events": events,
    }
    return out, summary
