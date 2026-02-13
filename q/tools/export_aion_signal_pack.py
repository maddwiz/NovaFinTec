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
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    wf_path = Path(args.wf)
    if not wf_path.exists():
        raise SystemExit(f"Missing input: {wf_path}")

    df = pd.read_csv(wf_path)
    if "asset" not in df.columns:
        raise SystemExit(f"{wf_path} must include an 'asset' column.")

    scored = _build_scores(df)
    scored = scored.replace([np.inf, -np.inf], np.nan).dropna(subset=["symbol", "bias", "confidence"])
    scored = scored[scored["confidence"] >= float(args.min_confidence)].copy()
    scored = scored[scored["bias"].abs() > 1e-6].copy()
    scored = _collapse_to_canonical(scored)

    watchlist = _load_watchlist(args.watchlist_txt)
    if watchlist:
        scored = scored[scored["symbol"].isin(watchlist)].copy()

    scored = scored.sort_values("rank", ascending=False).head(max(1, int(args.max_symbols)))

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    global_overlay = _build_global_overlay(scored)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
