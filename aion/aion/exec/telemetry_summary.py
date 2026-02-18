"""
Summarize decision telemetry for operations dashboards.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(v)


def _parse_ts(raw) -> dt.datetime | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
        t = dt.datetime.fromisoformat(s)
        if t.tzinfo is None:
            return t.replace(tzinfo=dt.timezone.utc)
        return t.astimezone(dt.timezone.utc)
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = str(ln).strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    except Exception:
        return []
    return out


def _all_decision_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    files = sorted(parent.glob(f"{stem}*{suffix}"))
    ranked = []
    for p in files:
        mtime = p.stat().st_mtime if p.exists() else 0.0
        ranked.append((mtime, p))
    ranked.sort(key=lambda x: x[0])
    return [p for _, p in ranked]


def _decision(row: dict) -> str:
    if "decision" in row:
        return str(row.get("decision", "")).strip().upper()
    if "action" in row:
        return str(row.get("action", "")).strip().upper()
    return ""


def _closed_trade_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        dec = _decision(row)
        if not dec:
            continue
        pnl = row.get("pnl_realized")
        if pnl is None and isinstance(row.get("extras"), dict):
            pnl = row["extras"].get("pnl_realized")
        if pnl is None:
            continue
        if ("EXIT" in dec) or ("PARTIAL" in dec):
            x = dict(row)
            x["pnl_realized"] = _safe_float(pnl, 0.0)
            out.append(x)
    return out


def _top_signal_category(row: dict) -> str | None:
    pools = []
    if isinstance(row.get("entry_category_scores"), dict):
        pools.append(row.get("entry_category_scores"))
    if isinstance(row.get("category_scores"), dict):
        pools.append(row.get("category_scores"))
    if isinstance(row.get("extras"), dict):
        ex = row.get("extras", {})
        if isinstance(ex.get("entry_category_scores"), dict):
            pools.append(ex.get("entry_category_scores"))
        if isinstance(ex.get("category_scores"), dict):
            pools.append(ex.get("category_scores"))

    for pool in pools:
        clean = {}
        for k, v in pool.items():
            key = str(k).strip()
            if not key:
                continue
            clean[key] = _safe_float(v, 0.0)
        if not clean:
            continue
        best = max(clean.items(), key=lambda kv: kv[1])[0]
        if best:
            return str(best)
    return None


def _win_loss_ratio(pnls: list[float]) -> float:
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    if not wins:
        return 0.0
    if not losses:
        return float("inf")
    avg_win = float(sum(wins) / len(wins))
    avg_loss = float(abs(sum(losses) / len(losses)))
    if avg_loss <= 1e-12:
        return float("inf")
    return float(avg_win / avg_loss)


def build_telemetry_summary(
    *,
    decisions_path: Path,
    rolling_window: int = 20,
) -> dict:
    rows: list[dict] = []
    for p in _all_decision_files(decisions_path):
        rows.extend(_read_jsonl(p))

    enriched: list[dict] = []
    for row in rows:
        ts = _parse_ts(row.get("timestamp") or row.get("ts") or row.get("time"))
        if ts is None:
            continue
        x = dict(row)
        x["_ts"] = ts
        enriched.append(x)
    enriched.sort(key=lambda r: r["_ts"])

    closed = _closed_trade_rows(enriched)
    pnls = [float(r["pnl_realized"]) for r in closed]

    recent = pnls[-max(1, int(rolling_window)) :] if pnls else []
    rolling_hit = 0.0
    if recent:
        rolling_hit = float(sum(1 for p in recent if p > 0) / len(recent))

    total_hit = 0.0
    if pnls:
        total_hit = float(sum(1 for p in pnls if p > 0) / len(pnls))

    actual_slips = []
    est_slips = []
    for row in enriched:
        s_act = row.get("slippage_bps")
        s_est = row.get("estimated_slippage_bps")
        if s_act is None and isinstance(row.get("extras"), dict):
            s_act = row["extras"].get("slippage_bps")
            s_est = row["extras"].get("estimated_slippage_bps", s_est)
        if s_act is not None:
            actual_slips.append(_safe_float(s_act, 0.0))
        if s_est is not None:
            est_slips.append(_safe_float(s_est, 0.0))
    avg_actual_slip = float(sum(actual_slips) / len(actual_slips)) if actual_slips else 0.0
    avg_est_slip = float(sum(est_slips) / len(est_slips)) if est_slips else 0.0

    regime_pnl = defaultdict(float)
    for row in closed:
        reg = str(row.get("regime") or row.get("session_type") or "unknown").strip().lower() or "unknown"
        regime_pnl[reg] += float(row["pnl_realized"])
    best_regime = None
    worst_regime = None
    if regime_pnl:
        best_regime = max(regime_pnl.items(), key=lambda kv: kv[1])[0]
        worst_regime = min(regime_pnl.items(), key=lambda kv: kv[1])[0]

    win_reasons = Counter()
    loss_reasons = Counter()
    win_signals = Counter()
    loss_signals = Counter()
    for row in closed:
        reasons = row.get("reasons")
        if not isinstance(reasons, list):
            reasons = []
        pnl = float(row["pnl_realized"])
        for rr in reasons:
            r = str(rr).strip()
            if not r:
                continue
            if pnl > 0:
                win_reasons[r] += 1
            elif pnl < 0:
                loss_reasons[r] += 1

        sig = _top_signal_category(row)
        if sig:
            if pnl > 0:
                win_signals[sig] += 1
            elif pnl < 0:
                loss_signals[sig] += 1

    top_win_reasons = [{"reason": k, "count": int(v)} for k, v in win_reasons.most_common(5)]
    top_loss_reasons = [{"reason": k, "count": int(v)} for k, v in loss_reasons.most_common(5)]
    top_win_signals = [{"category": k, "count": int(v)} for k, v in win_signals.most_common(5)]
    top_loss_signals = [{"category": k, "count": int(v)} for k, v in loss_signals.most_common(5)]

    out = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "decisions_rows": int(len(enriched)),
        "closed_trade_events": int(len(closed)),
        "rolling_window": int(max(1, int(rolling_window))),
        "rolling_hit_rate": float(rolling_hit),
        "total_hit_rate": float(total_hit),
        "win_loss_ratio": float(_win_loss_ratio(pnls)),
        "avg_slippage_bps": float(avg_actual_slip),
        "avg_estimated_slippage_bps": float(avg_est_slip),
        "slippage_delta_bps": float(avg_actual_slip - avg_est_slip),
        "most_profitable_regime": best_regime,
        "worst_regime": worst_regime,
        "top_win_reasons": top_win_reasons,
        "top_loss_reasons": top_loss_reasons,
        "top_win_signal_category": (None if not top_win_signals else str(top_win_signals[0]["category"])),
        "top_loss_signal_category": (None if not top_loss_signals else str(top_loss_signals[0]["category"])),
        "top_win_signal_categories": top_win_signals,
        "top_loss_signal_categories": top_loss_signals,
    }
    return out


def write_telemetry_summary(
    *,
    decisions_path: Path,
    output_path: Path,
    rolling_window: int = 20,
) -> dict:
    out = build_telemetry_summary(
        decisions_path=decisions_path,
        rolling_window=rolling_window,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
