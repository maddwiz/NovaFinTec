"""
Structured JSONL telemetry for day-skimmer decisions.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path


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
        state_dir = Path(getattr(cfg_mod, "STATE_DIR", Path.cwd()))
        state_dir.mkdir(parents=True, exist_ok=True)
        self.path = state_dir / str(filename)
        self._active_day: str | None = None

    def _rotate_if_needed(self, ts_utc: dt.datetime):
        day = ts_utc.strftime("%Y%m%d")
        if self._active_day is None:
            self._active_day = day
            return
        if day == self._active_day:
            return
        if self.path.exists() and self.path.stat().st_size > 0:
            archived = self.path.with_name(f"{self.path.stem}.{self._active_day}{self.path.suffix}")
            suffix = 1
            while archived.exists():
                archived = self.path.with_name(f"{self.path.stem}.{self._active_day}.{suffix}{self.path.suffix}")
                suffix += 1
            self.path.replace(archived)
        self._active_day = day

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
        self._rotate_if_needed(ts)

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

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=True, separators=(",", ":"), default=str) + "\n")
        return rec
