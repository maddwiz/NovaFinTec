#!/usr/bin/env python3
"""
Calendar/event alpha overlay.

Components:
  - Turn-of-month seasonality (last 2 + first 3 business days)
  - Weekday drift profile (small bias)
  - Optional explicit event impulses from runs_plus/calendar_events.csv

Writes:
  - runs_plus/calendar_event_signal.csv
  - runs_plus/calendar_event_overlay.csv
  - runs_plus/calendar_event_info.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
DATA = ROOT / "data"
RUNS.mkdir(exist_ok=True)


def _append_card(title: str, html: str) -> None:
    if str(os.getenv("Q_DISABLE_REPORT_CARDS", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        return
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


def _target_len() -> int:
    p = RUNS / "asset_returns.csv"
    if not p.exists():
        return 0
    try:
        a = np.loadtxt(p, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(p, delimiter=",", skiprows=1)
        except Exception:
            return 0
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return int(a.shape[0])


def _align_tail(v: np.ndarray, t: int, fill: float) -> np.ndarray:
    x = np.asarray(v, float).ravel()
    if t <= 0:
        return x
    if x.size >= t:
        return x[-t:]
    out = np.full(t, float(fill), dtype=float)
    if x.size > 0:
        out[-x.size :] = x
        out[: t - x.size] = float(x[0])
    return out


def _load_calendar_index() -> pd.DatetimeIndex:
    preferred = ["SPY.csv", "QQQ.csv"]
    files = []
    for n in preferred:
        p = DATA / n
        if p.exists():
            files.append(p)
    files.extend([p for p in sorted(DATA.glob("*.csv")) if p.name not in {x.name for x in files}])

    for p in files:
        try:
            df = pd.read_csv(p, usecols=lambda c: str(c).lower() in {"date", "timestamp"})
        except Exception:
            continue
        if df.empty:
            continue
        dcol = None
        for c in df.columns:
            if str(c).lower() in {"date", "timestamp"}:
                dcol = c
                break
        if dcol is None:
            continue
        idx = pd.to_datetime(df[dcol], errors="coerce").dropna()
        if len(idx) >= 40:
            idx = pd.DatetimeIndex(idx).sort_values().unique()
            return idx
    return pd.DatetimeIndex([])


def _turn_of_month_signal(idx: pd.DatetimeIndex) -> pd.Series:
    if len(idx) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(0.0, index=idx)
    periods = idx.to_period("M")
    for m in periods.unique():
        m_idx = idx[periods == m]
        if len(m_idx) == 0:
            continue
        n = len(m_idx)
        pos = np.arange(1, n + 1)
        mask = (pos <= 3) | (pos >= max(1, n - 1))
        vals = np.where(mask, 1.0, 0.0)
        s.loc[m_idx] = vals
    return s


def _weekday_signal(idx: pd.DatetimeIndex) -> pd.Series:
    if len(idx) == 0:
        return pd.Series(dtype=float)
    # Small canonical profile: Monday softer, Friday firmer.
    mapping = {
        0: -0.10,  # Monday
        1: 0.03,   # Tuesday
        2: 0.02,   # Wednesday
        3: 0.01,   # Thursday
        4: 0.08,   # Friday
    }
    out = [float(mapping.get(int(d.weekday()), 0.0)) for d in idx]
    return pd.Series(out, index=idx, dtype=float)


def _event_signal(idx: pd.DatetimeIndex, event_file: Path) -> pd.Series:
    out = pd.Series(0.0, index=idx, dtype=float)
    if len(idx) == 0 or (not event_file.exists()):
        return out
    try:
        df = pd.read_csv(event_file)
    except Exception:
        return out
    if df.empty:
        return out

    dcol = None
    for c in ["DATE", "Date", "date", "timestamp", "Timestamp"]:
        if c in df.columns:
            dcol = c
            break
    if dcol is None:
        return out

    ed = pd.to_datetime(df[dcol], errors="coerce").dt.normalize()
    if "direction" in df.columns:
        direction = pd.to_numeric(df["direction"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    else:
        direction = pd.Series(1.0, index=df.index)
    if "impact" in df.columns:
        impact = pd.to_numeric(df["impact"], errors="coerce").fillna(1.0).clip(0.0, 3.0)
    else:
        impact = pd.Series(1.0, index=df.index)

    ev = pd.DataFrame({"date": ed, "v": direction * impact}).dropna()
    if ev.empty:
        return out
    agg = ev.groupby("date")["v"].sum()
    idx_norm = pd.DatetimeIndex(idx).normalize()
    out_vals = []
    for d in idx_norm:
        out_vals.append(float(agg.get(d, 0.0)))
    return pd.Series(out_vals, index=idx, dtype=float)


def main() -> int:
    t = _target_len()
    idx = _load_calendar_index()
    if len(idx) == 0:
        if t <= 0:
            print("(!) Calendar index unavailable; skipping.")
            return 0
        sig = np.zeros(t, dtype=float)
        ov = np.ones(t, dtype=float)
        np.savetxt(RUNS / "calendar_event_signal.csv", sig, delimiter=",")
        np.savetxt(RUNS / "calendar_event_overlay.csv", ov, delimiter=",")
        (RUNS / "calendar_event_info.json").write_text(
            json.dumps({"ok": False, "reason": "missing_calendar_index", "rows": int(t)}, indent=2),
            encoding="utf-8",
        )
        return 0

    include_tom = str(os.getenv("Q_CALENDAR_INCLUDE_TOM", "1")).strip().lower() in {"1", "true", "yes", "on"}
    include_dow = str(os.getenv("Q_CALENDAR_INCLUDE_DOW", "1")).strip().lower() in {"1", "true", "yes", "on"}
    tom_w = float(np.clip(float(os.getenv("Q_CALENDAR_TOM_WEIGHT", "0.90")), 0.0, 3.0))
    dow_w = float(np.clip(float(os.getenv("Q_CALENDAR_DOW_WEIGHT", "0.25")), 0.0, 3.0))
    event_w = float(np.clip(float(os.getenv("Q_CALENDAR_EVENT_WEIGHT", "0.60")), 0.0, 3.0))
    smooth_alpha = float(np.clip(float(os.getenv("Q_CALENDAR_SMOOTH_ALPHA", "0.10")), 0.01, 0.90))
    beta = float(np.clip(float(os.getenv("Q_CALENDAR_EVENT_BETA", "0.10")), 0.0, 1.2))
    floor = float(np.clip(float(os.getenv("Q_CALENDAR_EVENT_FLOOR", "0.90")), 0.2, 1.2))
    ceil = float(np.clip(float(os.getenv("Q_CALENDAR_EVENT_CEIL", "1.10")), floor, 1.8))
    event_file = Path(os.getenv("Q_CALENDAR_EVENT_FILE", str(RUNS / "calendar_events.csv")))

    tom = _turn_of_month_signal(idx) if include_tom else pd.Series(0.0, index=idx, dtype=float)
    dow = _weekday_signal(idx) if include_dow else pd.Series(0.0, index=idx, dtype=float)
    ev = _event_signal(idx, event_file)

    raw = (
        tom_w * tom.values
        + dow_w * dow.values
        + event_w * np.clip(ev.values, -2.0, 2.0)
    )
    raw = np.clip(raw, -6.0, 6.0)
    sig = np.tanh(raw / 2.0)
    sig = pd.Series(sig, index=idx).ewm(alpha=smooth_alpha, adjust=False).mean().clip(-1.0, 1.0).values
    ov = np.clip(1.0 + beta * sig, floor, ceil)

    if t > 0:
        sig = _align_tail(sig, t, 0.0)
        ov = _align_tail(ov, t, 1.0)

    np.savetxt(RUNS / "calendar_event_signal.csv", np.asarray(sig, float), delimiter=",")
    np.savetxt(RUNS / "calendar_event_overlay.csv", np.asarray(ov, float), delimiter=",")

    info = {
        "ok": True,
        "rows": int(len(sig)),
        "event_file": str(event_file),
        "event_file_exists": bool(event_file.exists()),
        "params": {
            "include_tom": bool(include_tom),
            "include_dow": bool(include_dow),
            "tom_weight": float(tom_w),
            "dow_weight": float(dow_w),
            "event_weight": float(event_w),
            "smooth_alpha": float(smooth_alpha),
            "beta": float(beta),
            "floor": float(floor),
            "ceil": float(ceil),
        },
        "signal_mean": float(np.mean(sig)),
        "signal_min": float(np.min(sig)),
        "signal_max": float(np.max(sig)),
        "overlay_mean": float(np.mean(ov)),
        "overlay_min": float(np.min(ov)),
        "overlay_max": float(np.max(ov)),
    }
    (RUNS / "calendar_event_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Calendar/Event Overlay ✔",
        (
            f"<p>rows={len(sig)}, event_file_exists={info['event_file_exists']}, "
            f"signal_mean={info['signal_mean']:.3f}, "
            f"overlay_range=[{info['overlay_min']:.3f},{info['overlay_max']:.3f}].</p>"
        ),
    )
    print(f"✅ Wrote {RUNS/'calendar_event_signal.csv'}")
    print(f"✅ Wrote {RUNS/'calendar_event_overlay.csv'}")
    print(f"✅ Wrote {RUNS/'calendar_event_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
