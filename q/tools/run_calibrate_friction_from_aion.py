#!/usr/bin/env python3
"""
Calibrate Q transaction-friction assumptions from live AION execution logs.

Reads:
  - aion/logs/shadow_trades.csv (or Q_AION_SHADOW_TRADES override)
  - aion/logs/runtime_monitor.json (optional)
  - runs_plus/daily_costs_info.json (optional baseline fallback)

Writes:
  - runs_plus/friction_calibration.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.aion_feedback import resolve_shadow_trades_path

RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _safe_float(x, default=None):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_dt_utc(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    return s


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    ww = np.asarray(w, float)
    xx = np.asarray(x, float)
    sw = float(np.sum(ww))
    if sw <= 1e-9:
        return float(np.mean(xx))
    return float(np.sum(xx * ww) / sw)


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


def _runtime_monitor_path() -> Path:
    env = str(os.getenv("Q_AION_RUNTIME_MONITOR", "")).strip()
    if env:
        return Path(env)
    home = str(os.getenv("Q_AION_HOME", str(ROOT.parent / "aion"))).strip()
    return Path(home) / "logs" / "runtime_monitor.json"


def _shadow_stats(path: Path, lookback: int) -> dict:
    out = {
        "path": str(path),
        "samples": 0,
        "avg_slippage_bps": None,
        "p50_slippage_bps": None,
        "p90_slippage_bps": None,
        "avg_fill_ratio": None,
        "last_ts": None,
        "age_hours": None,
    }
    if (not path.exists()) or path.is_dir():
        return out
    try:
        df = pd.read_csv(path)
    except Exception:
        return out
    if df.empty or "slippage_bps" not in df.columns:
        return out
    sl = pd.to_numeric(df.get("slippage_bps"), errors="coerce")
    fr = pd.to_numeric(df.get("fill_ratio"), errors="coerce")
    qty = pd.to_numeric(df.get("qty"), errors="coerce")
    mask = sl.notna() & np.isfinite(sl.values)
    sub = df.loc[mask].copy()
    if sub.empty:
        return out
    sub["slippage_bps"] = pd.to_numeric(sub["slippage_bps"], errors="coerce").fillna(0.0).astype(float)
    sub["fill_ratio"] = pd.to_numeric(sub.get("fill_ratio"), errors="coerce").fillna(1.0).clip(0.0, 1.0).astype(float)
    sub["qty"] = pd.to_numeric(sub.get("qty"), errors="coerce").fillna(1.0).abs().clip(1.0, 5000.0).astype(float)
    sub = sub.tail(max(5, int(lookback)))
    vals = sub["slippage_bps"].to_numpy(dtype=float)
    w = np.sqrt(sub["qty"].to_numpy(dtype=float)) * np.clip(sub["fill_ratio"].to_numpy(dtype=float), 0.25, 1.0)
    out["samples"] = int(len(vals))
    if len(vals):
        out["avg_slippage_bps"] = float(_weighted_mean(vals, w))
        out["p50_slippage_bps"] = float(np.quantile(vals, 0.50))
        out["p90_slippage_bps"] = float(np.quantile(vals, 0.90))
        out["avg_fill_ratio"] = float(np.mean(sub["fill_ratio"].to_numpy(dtype=float)))
    if "timestamp" in sub.columns:
        ts = _parse_dt_utc(sub["timestamp"])
        ts = ts.dropna()
        if len(ts):
            last = ts.iloc[-1]
            out["last_ts"] = last.isoformat()
            now = pd.Timestamp.now(tz="UTC")
            out["age_hours"] = float(max(0.0, (now - last).total_seconds() / 3600.0))
    return out


def _monitor_stats(path: Path, lookback: int) -> dict:
    out = {
        "path": str(path),
        "samples": 0,
        "avg_slippage_bps": None,
        "p50_slippage_bps": None,
        "p90_slippage_bps": None,
    }
    obj = _load_json(path)
    if not obj:
        return out
    pts = []
    for x in obj.get("slippage_points", []) if isinstance(obj.get("slippage_points"), list) else []:
        v = _safe_float(x, None)
        if v is not None:
            pts.append(float(v))
    if not pts:
        evts = obj.get("execution_events", [])
        if isinstance(evts, list):
            for e in evts:
                if not isinstance(e, dict):
                    continue
                v = _safe_float(e.get("slippage_bps"), None)
                if v is not None:
                    pts.append(float(v))
    if not pts:
        return out
    vals = np.asarray(pts[-max(5, int(lookback)):], dtype=float)
    out["samples"] = int(len(vals))
    out["avg_slippage_bps"] = float(np.mean(vals))
    out["p50_slippage_bps"] = float(np.quantile(vals, 0.50))
    out["p90_slippage_bps"] = float(np.quantile(vals, 0.90))
    return out


def _recommend(shadow: dict, monitor: dict, baseline_bps: float) -> dict:
    min_samples = int(np.clip(_safe_float(os.getenv("Q_FRICTION_CALIB_MIN_SAMPLES", 20), 20), 1, 2000))
    stale_max_h = float(np.clip(_safe_float(os.getenv("Q_FRICTION_CALIB_MAX_SHADOW_AGE_HOURS", 72.0), 72.0), 1.0, 24.0 * 30.0))
    blend_shadow = float(np.clip(_safe_float(os.getenv("Q_FRICTION_CALIB_SHADOW_WEIGHT", 0.70), 0.70), 0.0, 1.0))
    spread_mult = float(np.clip(_safe_float(os.getenv("Q_FRICTION_CALIB_SPREAD_MULT", 0.35), 0.35), 0.0, 3.0))
    fill_target = float(np.clip(_safe_float(os.getenv("Q_FRICTION_CALIB_FILL_TARGET", 0.95), 0.95), 0.50, 1.0))
    fill_pen_mult = float(np.clip(_safe_float(os.getenv("Q_FRICTION_CALIB_FILL_PENALTY_BPS", 40.0), 40.0), 0.0, 200.0))
    min_bps = float(np.clip(_safe_float(os.getenv("Q_FRICTION_CALIB_MIN_BPS", 2.0), 2.0), 0.0, 100.0))
    max_bps = float(np.clip(_safe_float(os.getenv("Q_FRICTION_CALIB_MAX_BPS", 60.0), 60.0), min_bps, 200.0))
    vol_mult = float(np.clip(_safe_float(os.getenv("Q_FRICTION_CALIB_VOL_SCALED_MULT", 0.75), 0.75), 0.0, 5.0))
    max_vol_scaled = float(np.clip(_safe_float(os.getenv("Q_FRICTION_CALIB_MAX_VOL_SCALED_BPS", 25.0), 25.0), 0.0, 200.0))

    ss = int(shadow.get("samples") or 0)
    ms = int(monitor.get("samples") or 0)
    age_h = _safe_float(shadow.get("age_hours"), None)
    sw = blend_shadow if ss > 0 else 0.0
    mw = (1.0 - blend_shadow) if ms > 0 else 0.0
    reasons = []
    if (age_h is not None) and age_h > stale_max_h:
        sw *= 0.35
        reasons.append("shadow_stale_weight_reduced")
    if ss == 0 and ms > 0:
        reasons.append("shadow_missing_use_monitor")
    if ms == 0 and ss > 0:
        reasons.append("monitor_missing_use_shadow")
    if (sw + mw) <= 1e-9:
        sw = 1.0
        mw = 0.0
        reasons.append("fallback_no_samples")

    def _pick(metric: str, default: float = 0.0) -> float:
        sv = _safe_float(shadow.get(metric), None)
        mv = _safe_float(monitor.get(metric), None)
        if sv is None and mv is None:
            return float(default)
        if sv is None:
            return float(mv)
        if mv is None:
            return float(sv)
        return float((sw * sv + mw * mv) / max(1e-9, sw + mw))

    mean_slip = _pick("avg_slippage_bps", default=baseline_bps)
    p50 = _pick("p50_slippage_bps", default=mean_slip)
    p90 = _pick("p90_slippage_bps", default=mean_slip)
    disp = max(0.0, p90 - p50)
    avg_fill = _safe_float(shadow.get("avg_fill_ratio"), None)
    fill_pen = 0.0
    if avg_fill is not None:
        fill_pen = max(0.0, fill_target - avg_fill) * fill_pen_mult

    raw_base = mean_slip + spread_mult * disp + fill_pen
    suggested_base = float(np.clip(raw_base, min_bps, max_bps))
    suggested_base = max(float(baseline_bps), suggested_base)
    suggested_vol = float(np.clip(vol_mult * disp, 0.0, max_vol_scaled))

    total_samples = ss + ms
    ok = bool(total_samples >= min_samples)
    if not ok:
        reasons.append(f"insufficient_samples<{min_samples} ({total_samples})")

    return {
        "ok": ok,
        "reasons": reasons,
        "weights": {"shadow": float(sw), "monitor": float(mw)},
        "stats": {
            "blended_mean_slippage_bps": float(mean_slip),
            "blended_p50_slippage_bps": float(p50),
            "blended_p90_slippage_bps": float(p90),
            "blended_dispersion_bps": float(disp),
            "avg_fill_ratio": avg_fill,
            "fill_penalty_bps": float(fill_pen),
            "samples_total": int(total_samples),
            "samples_shadow": int(ss),
            "samples_monitor": int(ms),
        },
        "recommended_cost_base_bps": float(suggested_base),
        "recommended_cost_vol_scaled_bps": float(suggested_vol),
        "baseline_cost_base_bps": float(baseline_bps),
    }


def main() -> int:
    shadow_path = resolve_shadow_trades_path(root=ROOT)
    monitor_path = _runtime_monitor_path()
    shadow_lookback = int(np.clip(_safe_float(os.getenv("Q_FRICTION_CALIB_SHADOW_LOOKBACK", 400), 400), 5, 100000))
    monitor_lookback = int(np.clip(_safe_float(os.getenv("Q_FRICTION_CALIB_MONITOR_LOOKBACK", 400), 400), 5, 100000))

    cinfo = _load_json(RUNS / "daily_costs_info.json")
    baseline_bps = float(np.clip(_safe_float(cinfo.get("cost_base_bps", 10.0), 10.0), 0.0, 100.0))

    shadow = _shadow_stats(shadow_path, lookback=shadow_lookback)
    monitor = _monitor_stats(monitor_path, lookback=monitor_lookback)
    rec = _recommend(shadow, monitor, baseline_bps=baseline_bps)

    out = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "ok": bool(rec.get("ok", False)),
        "shadow": shadow,
        "monitor": monitor,
        "recommendation": rec,
    }
    (RUNS / "friction_calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    rs = rec.get("stats", {})
    html = (
        f"<p>Friction calibration: ok={bool(rec.get('ok', False))}, "
        f"samples={int(rs.get('samples_total', 0))}, "
        f"base_bps={float(rec.get('recommended_cost_base_bps', baseline_bps)):.2f}, "
        f"vol_scaled_bps={float(rec.get('recommended_cost_vol_scaled_bps', 0.0)):.2f}.</p>"
    )
    if rec.get("reasons"):
        html += f"<p>Notes: {', '.join(str(x) for x in rec.get('reasons', []))}</p>"
    _append_card("Friction Calibration ✔", html)

    print(f"✅ Wrote {RUNS/'friction_calibration.json'}")
    print(
        "Friction calibration:",
        f"ok={bool(rec.get('ok', False))}",
        f"base_bps={float(rec.get('recommended_cost_base_bps', baseline_bps)):.2f}",
        f"vol_scaled_bps={float(rec.get('recommended_cost_vol_scaled_bps', 0.0)):.2f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
