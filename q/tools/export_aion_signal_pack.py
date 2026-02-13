#!/usr/bin/env python3
"""
Export a compact signal pack for AION consumption.

Primary input: runs_plus/walk_forward_table_plus.csv
Primary output: runs_plus/q_signal_overlay.json
Optional mirror: a second JSON path (e.g. AION state folder).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _canonical_symbol(sym: str) -> str:
    s = str(sym or "").strip().upper()
    if not s:
        return ""
    if "_" in s:
        s = s.split("_", 1)[0]
    return s


def _build_scores(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()
    for col in ["sharpe", "hit", "meta_weight", "council_weight"]:
        if col not in w.columns:
            w[col] = 0.0
        w[col] = pd.to_numeric(w[col], errors="coerce").fillna(0.0)

    # Forecast direction preference from model-side voting.
    vote = w["meta_weight"] + w["council_weight"]
    sign_vote = np.sign(vote)
    sign_fallback = np.sign(w["sharpe"])
    direction = np.where(sign_vote != 0.0, sign_vote, sign_fallback)

    sharpe_q = np.tanh(w["sharpe"] / 2.0)
    hit_q = np.clip((w["hit"] - 0.5) / 0.2, -1.0, 1.0)
    quality = 0.65 * sharpe_q + 0.35 * hit_q

    # Bias is directional and bounded to avoid oversized external influence.
    bias = np.clip(direction * np.abs(quality), -1.0, 1.0)
    confidence = np.clip(0.50 + 0.50 * np.abs(quality), 0.0, 1.0)

    out = pd.DataFrame(
        {
            "symbol": w["asset"].astype(str).str.upper(),
            "bias": bias.astype(float),
            "confidence": confidence.astype(float),
            "sharpe": w["sharpe"].astype(float),
            "hit": w["hit"].astype(float),
        }
    )
    out["rank"] = out["confidence"] * out["bias"].abs()
    return out

def _build_scores_from_final_weights(weights_path: Path) -> pd.DataFrame:
    if not weights_path.exists():
        return pd.DataFrame(columns=["symbol", "bias", "confidence", "sharpe", "hit", "rank"])
    try:
        w = np.loadtxt(weights_path, delimiter=",")
    except Exception:
        try:
            w = np.loadtxt(weights_path, delimiter=",", skiprows=1)
        except Exception:
            return pd.DataFrame(columns=["symbol", "bias", "confidence", "sharpe", "hit", "rank"])
    w = np.asarray(w, float)
    if w.ndim == 1:
        row = w.ravel()
    elif w.ndim == 2 and w.shape[0] >= 1:
        row = w[-1].ravel()
    else:
        return pd.DataFrame(columns=["symbol", "bias", "confidence", "sharpe", "hit", "rank"])

    syms = []
    an = RUNS / "asset_names.csv"
    if an.exists():
        try:
            adf = pd.read_csv(an)
            if len(adf.columns) >= 1:
                c0 = adf.columns[0]
                syms = [str(x).upper().strip() for x in adf[c0].tolist() if str(x).strip()]
        except Exception:
            syms = []
    if not syms:
        syms = sorted([p.stem.replace("_prices", "").upper() for p in (ROOT / "data").glob("*.csv") if p.is_file()])
    if not syms:
        syms = [f"ASSET_{i+1}" for i in range(len(row))]
    n = min(len(syms), len(row))
    syms = syms[:n]
    vals = row[:n].astype(float)
    absv = np.abs(vals)
    scale = float(np.nanpercentile(absv, 75)) if np.isfinite(absv).any() else 1.0
    scale = max(scale, 1e-6)
    bias = np.clip(vals / (2.5 * scale), -1.0, 1.0)
    confidence = np.clip(absv / (2.0 * scale), 0.0, 1.0)
    out = pd.DataFrame({"symbol": syms, "bias": bias, "confidence": confidence})
    out["sharpe"] = 0.0
    out["hit"] = 0.5
    out["rank"] = out["confidence"] * out["bias"].abs()
    return out


def _collapse_to_canonical(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()

    rows = []
    scored = scored.copy()
    scored["symbol"] = scored["symbol"].map(_canonical_symbol)
    scored = scored[scored["symbol"] != ""]

    for symbol, g in scored.groupby("symbol"):
        conf = pd.to_numeric(g["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        w = conf.values.astype(float)
        w_sum = float(w.sum())
        if w_sum <= 1e-12:
            w = np.ones(len(g), dtype=float)
            w_sum = float(w.sum())

        bias_vals = pd.to_numeric(g["bias"], errors="coerce").fillna(0.0).values.astype(float)
        sharpe_vals = pd.to_numeric(g["sharpe"], errors="coerce").fillna(0.0).values.astype(float)
        hit_vals = pd.to_numeric(g["hit"], errors="coerce").fillna(0.5).values.astype(float)

        bias = float(np.dot(w, bias_vals) / w_sum)
        sharpe = float(np.dot(w, sharpe_vals) / w_sum)
        hit = float(np.dot(w, hit_vals) / w_sum)
        confidence = float(conf.max())

        rows.append(
            {
                "symbol": symbol,
                "bias": _clamp(bias, -1.0, 1.0),
                "confidence": _clamp(confidence, 0.0, 1.0),
                "sharpe": sharpe,
                "hit": _clamp(hit, 0.0, 1.0),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank"] = out["confidence"] * out["bias"].abs()
    return out.sort_values("rank", ascending=False).reset_index(drop=True)


def _load_watchlist(path_value: str) -> set[str]:
    raw = str(path_value or "").strip()
    if not raw:
        return set()
    p = Path(raw)
    if not p.exists() or p.is_dir():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        sym = _canonical_symbol(line)
        if sym:
            out.add(sym)
    return out


def _load_json(path: Path):
    if not path.exists() or path.is_dir():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _hours_since_file(path: Path) -> float | None:
    if not path.exists():
        return None
    ts = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    return float((dt.datetime.now(dt.timezone.utc) - ts).total_seconds() / 3600.0)


def _load_series(path: Path):
    if not path.exists() or path.is_dir():
        return None
    try:
        a = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    a = np.asarray(a, float)
    if a.ndim == 2 and a.shape[1] >= 1:
        a = a[:, -1]
    a = a.ravel()
    return a if len(a) else None


def _latest_series_value(path: Path, lo: float, hi: float, default: float = 1.0):
    s = _load_series(path)
    if s is None or len(s) == 0:
        return float(default), False
    v = _safe_float(s[-1], default=default)
    return _clamp(v, lo, hi), True


def _runtime_context(runs_dir: Path):
    comps = {}
    specs = {
        "global_governor": ("global_governor.csv", 0.45, 1.10, 1.0),
        "quality_governor": ("quality_governor.csv", 0.55, 1.15, 1.0),
        "quality_runtime_modifier": ("quality_runtime_modifier.csv", 0.55, 1.15, 1.0),
        "meta_mix_reliability_governor": ("meta_mix_reliability_governor.csv", 0.70, 1.20, 1.0),
        "hive_diversification_governor": ("hive_diversification_governor.csv", 0.80, 1.05, 1.0),
        "dream_coherence_governor": ("dream_coherence_governor.csv", 0.70, 1.20, 1.0),
        "novaspine_context_boost": ("novaspine_context_boost.csv", 0.85, 1.15, 1.0),
        "novaspine_hive_boost": ("novaspine_hive_boost.csv", 0.85, 1.15, 1.0),
        "heartbeat_exposure_scaler": ("heartbeat_exposure_scaler.csv", 0.40, 1.20, 1.0),
        "dna_stress_governor": ("dna_stress_governor.csv", 0.70, 1.15, 1.0),
        "reflex_health_governor": ("reflex_health_governor.csv", 0.70, 1.15, 1.0),
        "symbolic_governor": ("symbolic_governor.csv", 0.70, 1.15, 1.0),
        "legacy_exposure": ("legacy_exposure.csv", 0.40, 1.30, 1.0),
    }
    active_vals = []
    for k, (fname, lo, hi, dflt) in specs.items():
        v, found = _latest_series_value(runs_dir / fname, lo=lo, hi=hi, default=dflt)
        comps[k] = {"value": float(v), "found": bool(found)}
        if found:
            active_vals.append(float(v))

    if active_vals:
        arr = np.clip(np.asarray(active_vals, float), 0.20, 2.00)
        mult = float(np.exp(np.mean(np.log(arr + 1e-12))))
    else:
        mult = 1.0
    mult = float(np.clip(mult, 0.50, 1.10))

    if mult < 0.72:
        regime = "defensive"
    elif mult > 0.98:
        regime = "risk_on"
    else:
        regime = "balanced"

    return {
        "runtime_multiplier": mult,
        "regime": regime,
        "components": comps,
        "active_component_count": int(len(active_vals)),
    }


def _quality_gate(
    health_json: Path,
    alerts_json: Path,
    min_health_score: float,
    max_health_age_hours: float,
):
    issues = []
    health = _load_json(health_json) or {}
    alerts = _load_json(alerts_json) or {}

    score = _safe_float(health.get("health_score", 0.0), default=0.0)
    age_h = _hours_since_file(health_json)
    alerts_ok = bool(alerts.get("ok", True))

    if score < float(min_health_score):
        issues.append(f"health_score<{min_health_score} ({score:.1f})")
    if age_h is None:
        issues.append("system_health missing")
    elif age_h > float(max_health_age_hours):
        issues.append(f"system_health stale>{max_health_age_hours}h ({age_h:.2f}h)")
    if not alerts_ok:
        issues.append("health alerts not clear")

    return {
        "ok": len(issues) == 0,
        "score": score,
        "health_age_hours": age_h,
        "alerts_ok": alerts_ok,
        "issues": issues,
    }


def _build_global_overlay(scored: pd.DataFrame) -> dict:
    if scored.empty:
        return {"bias": 0.0, "confidence": 0.0}
    conf = pd.to_numeric(scored["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0).values.astype(float)
    abs_bias = pd.to_numeric(scored["bias"], errors="coerce").fillna(0.0).abs().values.astype(float)
    w = conf * np.maximum(abs_bias, 0.05)
    w_sum = float(w.sum())
    if w_sum <= 1e-12:
        w = np.ones(len(scored), dtype=float)
        w_sum = float(w.sum())

    bias_vals = pd.to_numeric(scored["bias"], errors="coerce").fillna(0.0).values.astype(float)
    bias = float(np.dot(w, bias_vals) / w_sum)
    conf_global = float(np.dot(w, conf) / w_sum)
    return {
        "bias": round(_clamp(bias, -1.0, 1.0), 6),
        "confidence": round(_clamp(conf_global, 0.0, 1.0), 6),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", default=str(RUNS / "walk_forward_table_plus.csv"))
    ap.add_argument("--out-json", default=str(RUNS / "q_signal_overlay.json"))
    ap.add_argument("--out-csv", default=str(RUNS / "q_signal_overlay.csv"))
    ap.add_argument("--mirror-json", default="", help="Optional second JSON write target.")
    ap.add_argument("--min-confidence", type=float, default=0.56)
    ap.add_argument("--max-symbols", type=int, default=80)
    ap.add_argument(
        "--watchlist-txt",
        default="",
        help="Optional watchlist file. If provided, exported symbols are filtered to this set.",
    )
    ap.add_argument("--health-json", default=str(RUNS / "system_health.json"))
    ap.add_argument("--alerts-json", default=str(RUNS / "health_alerts.json"))
    ap.add_argument("--min-health-score", type=float, default=70.0)
    ap.add_argument("--max-health-age-hours", type=float, default=8.0)
    ap.add_argument(
        "--allow-degraded",
        action="store_true",
        help="If set, still export scored signals even when quality gate fails.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    wf_path = Path(args.wf)
    used_fallback = False
    if wf_path.exists():
        df = pd.read_csv(wf_path)
        if "asset" not in df.columns:
            raise SystemExit(f"{wf_path} must include an 'asset' column.")
        scored = _build_scores(df)
    else:
        scored = _build_scores_from_final_weights(RUNS / "portfolio_weights_final.csv")
        used_fallback = True

    scored_raw = scored.copy()
    scored = scored.replace([np.inf, -np.inf], np.nan).dropna(subset=["symbol", "bias", "confidence"])
    scored = scored[scored["confidence"] >= float(args.min_confidence)].copy()
    scored = scored[scored["bias"].abs() > 1e-6].copy()
    if used_fallback and scored.empty and not scored_raw.empty:
        alt = scored_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=["symbol", "bias", "confidence"]).copy()
        alt = alt[alt["bias"].abs() > 1e-6].copy()
        alt = alt[alt["confidence"] >= 0.10].copy()
        scored = alt.sort_values("rank", ascending=False).head(max(1, min(20, int(args.max_symbols))))
    scored = _collapse_to_canonical(scored)

    watchlist = _load_watchlist(args.watchlist_txt)
    if watchlist:
        scored = scored[scored["symbol"].isin(watchlist)].copy()

    scored = scored.sort_values("rank", ascending=False).head(max(1, int(args.max_symbols)))
    qgate = _quality_gate(
        health_json=Path(args.health_json),
        alerts_json=Path(args.alerts_json),
        min_health_score=float(args.min_health_score),
        max_health_age_hours=float(args.max_health_age_hours),
    )
    ctx = _runtime_context(RUNS)
    degrade = (not qgate["ok"]) and (not bool(args.allow_degraded))
    if degrade:
        scored = scored.iloc[0:0].copy()
    else:
        health_scale = _clamp(_safe_float(qgate.get("score", 0.0), default=0.0) / 100.0, 0.70, 1.05)
        runtime_scale = _clamp(_safe_float(ctx.get("runtime_multiplier", 1.0), default=1.0), 0.50, 1.10)
        conf_scale = _clamp(health_scale * runtime_scale, 0.45, 1.10)
        bias_scale = _clamp(math.sqrt(max(0.0, runtime_scale)), 0.70, 1.05)
        if not scored.empty:
            scored = scored.copy()
            scored["confidence"] = np.clip(scored["confidence"].astype(float) * conf_scale, 0.0, 1.0)
            scored["bias"] = np.clip(scored["bias"].astype(float) * bias_scale, -1.0, 1.0)
            scored = scored[scored["confidence"] >= 0.05].copy()
            scored["rank"] = scored["confidence"] * scored["bias"].abs()
            scored = scored.sort_values("rank", ascending=False)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    global_overlay = _build_global_overlay(scored)
    if degrade:
        global_overlay = {"bias": 0.0, "confidence": 0.0}
    payload = {
        "generated_at": ts,
        "source": "q.walk_forward_plus",
        "global": global_overlay,
        "signals": {
            row["symbol"]: {
                "bias": round(_clamp(row["bias"], -1.0, 1.0), 6),
                "confidence": round(_clamp(row["confidence"], 0.0, 1.0), 6),
            }
            for _, row in scored.iterrows()
        },
        "coverage": {
            "symbols": int(len(scored)),
            "watchlist_filtered": bool(watchlist),
        },
        "runtime_context": ctx,
        "quality_gate": qgate,
        "degraded_safe_mode": bool(degrade),
        "source_mode": "wf_table" if not used_fallback else "final_weights_fallback",
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scored_out = scored[["symbol", "bias", "confidence", "sharpe", "hit"]].copy()
    scored_out.insert(0, "generated_at", ts)
    scored_out.to_csv(out_csv, index=False)

    mirror = str(args.mirror_json).strip()
    if mirror:
        mirror_path = Path(mirror)
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        mirror_path.write_text(json.dumps(payload, indent=2))
        print(f"✅ Mirrored JSON: {mirror_path}")

    print(f"✅ Wrote {out_json}")
    print(f"✅ Wrote {out_csv}")
    print(f"Signals exported: {len(payload['signals'])}")
    print(f"Global overlay: bias={_safe_float(global_overlay.get('bias')):.4f}, confidence={_safe_float(global_overlay.get('confidence')):.4f}")
    if degrade:
        print(f"(!) Degraded safe mode enabled: {', '.join(qgate['issues'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
