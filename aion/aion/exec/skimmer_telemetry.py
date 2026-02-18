"""
Structured JSONL telemetry for day-skimmer decisions.
"""

from __future__ import annotations

import datetime as dt
import math

from .telemetry import DecisionTelemetry


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(v)


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _norm_str(x, default: str = "") -> str:
    s = str(x if x is not None else default).strip()
    return s if s else str(default)


def _norm_list(vals) -> list[str]:
    if not isinstance(vals, list):
        return []
    out = []
    for v in vals:
        s = str(v).strip()
        if s:
            out.append(s)
    return out


def _norm_category_scores(raw: dict | None) -> dict:
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in raw.items():
        kk = str(k).strip()
        if not kk:
            continue
        out[kk] = float(max(0.0, min(1.0, _safe_float(v, 0.0))))
    return out


class SkimmerTelemetry:
    def __init__(self, cfg_mod, filename: str = "skimmer_decisions.jsonl"):
        self.cfg = cfg_mod
        self.skimmer_sink = DecisionTelemetry(cfg_mod, filename=str(filename))
        canonical_name = str(getattr(cfg_mod, "TELEMETRY_DECISIONS_FILE", "trade_decisions.jsonl"))
        self.trade_sink = DecisionTelemetry(cfg_mod, filename=canonical_name)

    def log_decision(
        self,
        *,
        symbol: str,
        action: str,
        confluence_score: float = 0.0,
        category_scores: dict | None = None,
        session_phase: str = "unknown",
        session_type: str = "unknown",
        patterns_detected: list[str] | None = None,
        entry_price: float | None = None,
        stop_price: float | None = None,
        shares: int | None = None,
        risk_amount: float | None = None,
        reasons: list[str] | None = None,
        extras: dict | None = None,
        timestamp: dt.datetime | None = None,
    ) -> dict:
        ts = timestamp if isinstance(timestamp, dt.datetime) else dt.datetime.now(dt.timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        else:
            ts = ts.astimezone(dt.timezone.utc)

        rec = {
            "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": _norm_str(symbol, "").upper(),
            "action": _norm_str(action, "NOOP").upper(),
            "confluence_score": float(max(0.0, min(1.0, _safe_float(confluence_score, 0.0)))),
            "category_scores": _norm_category_scores(category_scores),
            "session_phase": _norm_str(session_phase, "unknown").lower(),
            "session_type": _norm_str(session_type, "unknown").lower(),
            "patterns_detected": _norm_list(patterns_detected if patterns_detected is not None else []),
            "entry_price": None if entry_price is None else _safe_float(entry_price, 0.0),
            "stop_price": None if stop_price is None else _safe_float(stop_price, 0.0),
            "shares": None if shares is None else max(0, _safe_int(shares, 0)),
            "risk_amount": None if risk_amount is None else max(0.0, _safe_float(risk_amount, 0.0)),
            "reasons": _norm_list(reasons if reasons is not None else []),
        }
        if isinstance(extras, dict) and extras:
            rec["extras"] = dict(extras)
        self.skimmer_sink.write(rec, timestamp=ts)

        x = rec.get("extras", {}) if isinstance(rec.get("extras"), dict) else {}
        q_bias = _safe_float(x.get("q_overlay_bias", 0.0), 0.0)
        q_conf = _safe_float(x.get("q_overlay_confidence", 0.0), 0.0)
        risk_distance = None
        if rec.get("entry_price") is not None and rec.get("stop_price") is not None:
            risk_distance = abs(_safe_float(rec.get("entry_price"), 0.0) - _safe_float(rec.get("stop_price"), 0.0))
        trade_rec = {
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": rec["symbol"],
            "decision": rec["action"],
            "q_overlay_bias": float(q_bias),
            "q_overlay_confidence": float(q_conf),
            "confluence_score": float(rec["confluence_score"]),
            "category_scores": rec.get("category_scores", {}),
            "entry_category_scores": x.get("entry_category_scores"),
            "intraday_alignment_score": float(rec["confluence_score"]),
            "regime": rec["session_type"],
            "governor_compound_scalar": x.get("governor_compound_scalar"),
            "entry_price": rec.get("entry_price"),
            "stop_price": rec.get("stop_price"),
            "risk_distance": risk_distance,
            "position_size_shares": rec.get("shares"),
            "book_imbalance": None,
            "reasons": rec.get("reasons", []),
            "pnl_realized": x.get("pnl_realized"),
            "slippage_bps": x.get("slippage_bps"),
            "estimated_slippage_bps": x.get("estimated_slippage_bps"),
        }
        self.trade_sink.write(trade_rec, timestamp=ts)
        return rec
