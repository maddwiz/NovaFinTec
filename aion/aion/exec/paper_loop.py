import datetime as dt
import json
import math
import time
from pathlib import Path
import numpy as np

from .. import config as cfg
from ..brain.signals import (
    build_trade_signal,
    compute_features,
    confidence_tag,
    intraday_entry_alignment,
    multi_timeframe_alignment,
    opposite_confidence,
)
from ..brain.external_signals import (
    blend_external_signals,
    load_external_signal_bundle,
    runtime_overlay_scale,
)
from ..brain.novaspine_bridge import build_trade_event, emit_trade_event, replay_trade_outbox
from ..data.ib_client import disconnect, hist_bars_cached, ib
from ..execution.simulator import ExecutionSimulator
from ..ml.meta_label import MetaLabelModel
from ..monitoring.runtime_monitor import RuntimeMonitor
from ..portfolio.optimizer import allocate_candidates
from ..risk.event_filter import EventRiskFilter
from ..risk.kill_switch import KillSwitch
from ..risk.exposure_gate import check_exposure
from ..risk.governor_diagnostics import build_diagnostic, write_governor_diagnostics
from ..risk.governor_hierarchy import GovernorAction, resolve_governor_action
from ..risk.policy import apply_policy_caps, load_policy, symbol_allowed
from ..risk.position_sizing import gross_leverage_ok, risk_qty
from .alerting import send_alert
from .audit_log import audit_log
from .order_events import attach_order_status_handler
from .kill_switch import KillSwitchWatcher
from .health_aggregator import write_system_health
from .order_state import save_order_state
from .reconciliation import reconcile_on_startup
from .shadow_state import apply_shadow_fill
from .telemetry import DecisionTelemetry
from .telemetry_summary import write_telemetry_summary
from ..utils.logging_utils import log_alert, log_equity, log_run, log_signal, log_trade

WATCHLIST_TXT = cfg.STATE_DIR / "watchlist.txt"
PROFILE_JSON = cfg.STATE_DIR / "strategy_profile.json"
RUNTIME_STATE_FILE = cfg.RUNTIME_STATE_FILE
RUNTIME_CONTROLS_FILE = cfg.STATE_DIR / "runtime_controls.json"


def _write_json_atomic(path: Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)


def _execute_fill(
    exe,
    *,
    side: str,
    qty: int,
    ref_price: float,
    atr_pct: float,
    confidence: float,
    allow_partial: bool,
    symbol: str,
    ib_client,
):
    try:
        return exe.execute(
            side=side,
            qty=qty,
            ref_price=ref_price,
            atr_pct=atr_pct,
            confidence=confidence,
            allow_partial=allow_partial,
            symbol=symbol,
            ib_client=ib_client,
        )
    except TypeError:
        # Backward-compatible path for test stubs and legacy executors.
        return exe.execute(
            side=side,
            qty=qty,
            ref_price=ref_price,
            atr_pct=atr_pct,
            confidence=confidence,
            allow_partial=allow_partial,
        )


def _extract_open_orders(client) -> list[dict]:
    rows: list[dict] = []
    try:
        trades = client.openTrades() or []
    except Exception:
        trades = []
    for t in trades:
        try:
            rows.append(
                {
                    "order_id": int(getattr(getattr(t, "order", None), "orderId", 0)),
                    "symbol": str(getattr(getattr(t, "contract", None), "symbol", "")).upper(),
                    "action": str(getattr(getattr(t, "order", None), "action", "")).upper(),
                    "qty": int(getattr(getattr(t, "order", None), "totalQuantity", 0)),
                    "status": str(getattr(getattr(t, "orderStatus", None), "status", "")),
                }
            )
        except Exception:
            continue
    return rows


def _ib_positions_market_value(client) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        positions = client.positions() or []
    except Exception:
        return out
    for p in positions:
        try:
            sym = str(getattr(getattr(p, "contract", None), "symbol", "")).upper()
            if not sym:
                continue
            if hasattr(p, "marketValue"):
                mv = float(getattr(p, "marketValue"))
            elif hasattr(p, "avgCost"):
                mv = float(getattr(p, "position", 0.0)) * float(getattr(p, "avgCost", 0.0))
            else:
                mv = float(getattr(p, "position", 0.0)) * float(getattr(p, "marketPrice", 0.0))
            out[sym] = float(mv)
        except Exception:
            continue
    return out


def _ib_net_liquidation(client, fallback: float) -> float:
    try:
        rows = client.accountSummary() or []
    except Exception:
        rows = []
    for row in rows:
        try:
            if str(getattr(row, "tag", "")).strip().lower() == "netliquidation":
                v = float(getattr(row, "value", 0.0))
                if math.isfinite(v) and v > 0:
                    return float(v)
        except Exception:
            continue
    v = float(fallback)
    return float(v) if math.isfinite(v) and v > 0 else 0.0


def _order_signed_notional(side: str, qty: int, price: float) -> float:
    q = max(0, int(qty))
    px = max(0.0, float(price))
    sign = 1.0 if str(side).upper() == "BUY" else -1.0
    return sign * q * px


def _run_exposure_gate(
    *,
    client,
    symbol: str,
    side: str,
    qty: int,
    price: float,
    fallback_nlv: float,
) -> tuple[bool, str, dict]:
    if client is None:
        return True, "ib_unavailable", {
            "current_exposure_pct": 0.0,
            "proposed_exposure_pct": 0.0,
            "limit_pct": float(cfg.MAX_GROSS_EXPOSURE_PCT),
        }
    positions = _ib_positions_market_value(client)
    nlv = _ib_net_liquidation(client, fallback=fallback_nlv)
    sym = str(symbol).upper()
    signed_notional = _order_signed_notional(side=side, qty=qty, price=price)
    existing = float(positions.get(sym, 0.0))
    projected = existing + signed_notional

    # If trade reduces absolute exposure in symbol, allow by construction.
    if abs(projected) <= abs(existing):
        return True, "reduces_existing_exposure", {
            "current_exposure_pct": float(sum(abs(v) for v in positions.values()) / max(1e-9, nlv if nlv > 0 else 1.0)),
            "proposed_exposure_pct": float(sum(abs(v) for v in positions.values()) / max(1e-9, nlv if nlv > 0 else 1.0)),
            "limit_pct": float(cfg.MAX_GROSS_EXPOSURE_PCT),
        }

    incremental = abs(projected) - abs(existing)
    gate = check_exposure(
        current_positions=positions,
        proposed_symbol=sym,
        proposed_value=float(incremental),
        net_liquidation=float(nlv),
        max_gross_exposure_pct=float(getattr(cfg, "MAX_GROSS_EXPOSURE_PCT", 0.95)),
        max_single_position_pct=float(getattr(cfg, "MAX_SINGLE_POSITION_PCT", 0.20)),
        max_correlated_exposure_pct=float(getattr(cfg, "MAX_CORRELATED_EXPOSURE_PCT", 0.40)),
        correlated_symbols=None,
    )
    return bool(gate.allowed), str(gate.reason), {
        "current_exposure_pct": float(gate.current_exposure_pct),
        "proposed_exposure_pct": float(gate.proposed_exposure_pct),
        "limit_pct": float(gate.limit_pct),
    }


def _ib_req_id(client) -> int:
    try:
        c = getattr(client, "client", None)
        getter = getattr(c, "getReqId", None)
        if callable(getter):
            return int(getter())
    except Exception:
        pass
    return 0


def _persist_ib_order_state(client) -> None:
    save_order_state(
        state_dir=Path(cfg.STATE_DIR),
        next_valid_id=_ib_req_id(client),
        open_orders=_extract_open_orders(client),
    )


def _write_runtime_governor_diagnostics(gov_results: list[dict], gov_action: GovernorAction) -> None:
    rows: list[dict] = []
    for item in (gov_results or []):
        name = str(item.get("name", "runtime_governor")).strip() or "runtime_governor"
        score = float(_safe_float(item.get("score", 1.0), 1.0))
        raw_threshold = item.get("threshold", None)
        threshold = None if raw_threshold is None else float(_safe_float(raw_threshold, 0.0))
        reason = str(item.get("reason", "runtime")).strip() or "runtime"
        action = "veto" if (threshold is not None and score <= threshold) else "pass"
        rows.append(
            build_diagnostic(
                name=name,
                values=[score],
                threshold=threshold,
                action=action,
                reason=reason,
                floor=0.0,
            )
        )

    if not rows:
        rows.append(
            build_diagnostic(
                name="runtime_governor_stack",
                values=[1.0],
                threshold=0.0,
                action="pass",
                reason="no_flags",
                floor=0.0,
            )
        )

    rows.append(
        {
            "name": "governor_hierarchy_action",
            "score": float(int(gov_action)),
            "min": float(int(gov_action)),
            "max": float(int(gov_action)),
            "threshold": None,
            "action": str(gov_action.name).lower(),
            "reason": "resolved_hierarchy",
            "pct_below_floor": 0.0,
        }
    )
    write_governor_diagnostics(Path(cfg.STATE_DIR) / "governor_diagnostics.json", rows)


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_watchlist():
    if not WATCHLIST_TXT.exists():
        return []
    return [s.strip().upper() for s in WATCHLIST_TXT.read_text().splitlines() if s.strip()]


def load_profile() -> dict:
    if not PROFILE_JSON.exists():
        return {}
    try:
        data = json.loads(PROFILE_JSON.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _isfinite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _compact_memory_runtime_context(
    *,
    ext_runtime_scale: float,
    ext_position_risk_scale: float,
    ext_runtime_diag: dict | None,
    ext_overlay_age_hours: float | None,
    ext_overlay_age_source: str,
    memory_feedback_status: str,
    memory_feedback_risk_scale: float,
    memory_feedback_turnover_pressure: float | None,
    memory_feedback_turnover_dampener: float | None,
    memory_feedback_block_new_entries: bool,
    aion_feedback_status: str,
    aion_feedback_source: str,
    aion_feedback_source_selected: str,
    aion_feedback_source_preference: str,
    aion_feedback_risk_scale: float,
    aion_feedback_stale: bool,
    aion_feedback_block_new_entries: bool,
    policy_block_new_entries: bool,
    killswitch_block_new_entries: bool,
    exec_governor_state: str,
    exec_governor_block_new_entries: bool,
) -> dict:
    diag = ext_runtime_diag if isinstance(ext_runtime_diag, dict) else {}
    flags_raw = diag.get("flags", [])
    flags = []
    if isinstance(flags_raw, list):
        for raw in flags_raw:
            key = str(raw).strip().lower()
            if key and key not in flags:
                flags.append(key)

    return {
        "external_runtime_scale": float(_safe_float(ext_runtime_scale, 1.0)),
        "external_position_risk_scale": float(_safe_float(ext_position_risk_scale, 1.0)),
        "external_regime": str(diag.get("regime", "unknown")).strip().lower() or "unknown",
        "external_overlay_stale": bool(diag.get("overlay_stale", False)),
        "external_overlay_age_hours": (
            None if ext_overlay_age_hours is None else float(_safe_float(ext_overlay_age_hours, 0.0))
        ),
        "external_overlay_age_source": str(ext_overlay_age_source or "unknown").strip().lower() or "unknown",
        "external_risk_flags": flags,
        "memory_feedback_status": str(memory_feedback_status or "unknown").strip().lower() or "unknown",
        "memory_feedback_risk_scale": float(_safe_float(memory_feedback_risk_scale, 1.0)),
        "memory_feedback_turnover_pressure": (
            None
            if memory_feedback_turnover_pressure is None
            else float(_safe_float(memory_feedback_turnover_pressure, 0.0))
        ),
        "memory_feedback_turnover_dampener": (
            None
            if memory_feedback_turnover_dampener is None
            else float(_safe_float(memory_feedback_turnover_dampener, 0.0))
        ),
        "memory_feedback_block_new_entries": bool(memory_feedback_block_new_entries),
        "aion_feedback_status": str(aion_feedback_status or "unknown").strip().lower() or "unknown",
        "aion_feedback_source": str(aion_feedback_source or "unknown").strip().lower() or "unknown",
        "aion_feedback_source_selected": (
            str(aion_feedback_source_selected or aion_feedback_source or "unknown").strip().lower() or "unknown"
        ),
        "aion_feedback_source_preference": (
            str(aion_feedback_source_preference or "auto").strip().lower() or "auto"
        ),
        "aion_feedback_risk_scale": float(_safe_float(aion_feedback_risk_scale, 1.0)),
        "aion_feedback_stale": bool(aion_feedback_stale),
        "aion_feedback_block_new_entries": bool(aion_feedback_block_new_entries),
        "policy_block_new_entries": bool(policy_block_new_entries),
        "killswitch_block_new_entries": bool(killswitch_block_new_entries),
        "exec_governor_state": str(exec_governor_state or "off").strip().lower() or "off",
        "exec_governor_block_new_entries": bool(exec_governor_block_new_entries),
    }


def _emit_trade_memory_event(
    event_type: str,
    symbol: str,
    side: str,
    qty: int,
    entry: float,
    exit_: float,
    pnl: float,
    reason: str,
    confidence: float,
    regime: str,
    monitor: RuntimeMonitor | None = None,
    extra: dict | None = None,
):
    if not bool(getattr(cfg, "MEMORY_ENABLE", False)):
        return

    try:
        ev = build_trade_event(
            event_type=str(event_type),
            symbol=str(symbol).upper(),
            side=str(side).upper(),
            qty=max(0, int(qty)),
            entry=float(entry),
            exit=float(exit_),
            pnl=float(pnl),
            reason=str(reason),
            confidence=float(confidence),
            regime=str(regime),
            extra=(extra if isinstance(extra, dict) else None),
        )
        res = emit_trade_event(ev, cfg)
        if not bool(res.get("ok", True)):
            msg = (
                f"NovaSpine emit issue event={event_type} symbol={symbol} "
                f"backend={res.get('backend')} error={res.get('error', 'unknown')}"
            )
            log_run(msg)
            if monitor is not None and cfg.MONITORING_ENABLED:
                monitor.record_system_event("novaspine_emit_fail", msg)
    except Exception as exc:
        msg = f"NovaSpine emit exception event={event_type} symbol={symbol}: {exc}"
        log_run(msg)
        if monitor is not None and cfg.MONITORING_ENABLED:
            monitor.record_system_event("novaspine_emit_exception", str(exc))


def _scale_external_signal(sig: dict | None, scale: float, max_bias: float):
    if not isinstance(sig, dict):
        return None
    s = max(0.0, min(2.0, _safe_float(scale, 1.0)))
    conf = max(0.0, min(1.0, _safe_float(sig.get("confidence"), 0.0) * s))
    bias = _safe_float(sig.get("bias"), 0.0) * math.sqrt(max(0.0, s))
    mb = abs(float(max_bias))
    bias = max(-mb, min(mb, bias))
    return {"bias": float(bias), "confidence": float(conf)}


def _runtime_risk_caps(
    max_trades_cap: int,
    max_open_positions_cap: int,
    ext_runtime_scale: float,
    ext_runtime_diag: dict | None,
):
    cap_scale = max(0.50, min(1.00, _safe_float(ext_runtime_scale, 1.0)))
    max_trades_cap_runtime = max(1, int(round(int(max_trades_cap) * cap_scale)))
    max_open_positions_runtime = max(1, int(round(int(max_open_positions_cap) * cap_scale)))

    diag = ext_runtime_diag if isinstance(ext_runtime_diag, dict) else {}
    flags = [str(x).strip().lower() for x in diag.get("flags", [])] if isinstance(diag.get("flags", []), list) else []
    regime = str(diag.get("regime", "")).strip().lower()
    degraded = bool(diag.get("degraded", False))
    quality_ok = bool(diag.get("quality_gate_ok", True))

    # Fracture and degraded states should tighten concurrency beyond pure scalar.
    if "drift_alert" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(2, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.60))))
    elif "drift_warn" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(3, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.82))))
    if "quality_governor_step_spike" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(3, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.75))))
    if "overlay_stale" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(2, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.70))))

    # Fracture and degraded states should tighten concurrency beyond pure scalar.
    if "fracture_alert" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(2, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.60))))
    elif "fracture_warn" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(3, int(max_open_positions_cap))))

    # Execution-aware hard/tight flags from Q constraint runtime.
    if "exec_risk_hard" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(2, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.55))))
    elif "exec_risk_tight" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(3, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.80))))

    # Nested-WF leakage quality flags from Q runtime context.
    if "nested_leakage_alert" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(2, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.65))))
    elif "nested_leakage_warn" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(3, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.85))))

    # Hive ecosystem stress flags from Q hive/cross-hive diagnostics.
    if "hive_stress_alert" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(2, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.60))))
    elif "hive_stress_warn" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(4, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.80))))
    if "hive_crowding_alert" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(2, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.62))))
    elif "hive_crowding_warn" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(4, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.82))))
    if "hive_entropy_alert" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(3, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.70))))
    elif "hive_entropy_warn" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(4, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.86))))
    if "hive_turnover_alert" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(2, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.60))))
    elif "hive_turnover_warn" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(4, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.80))))
    if "memory_turnover_alert" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(2, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.58))))
    elif "memory_turnover_warn" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(4, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.78))))

    # Heartbeat stress flags from Q heartbeat module.
    if "heartbeat_alert" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(2, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.58))))
    elif "heartbeat_warn" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(4, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.78))))

    # Council divergence flags from Q council mix instability.
    if "council_divergence_alert" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(2, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.60))))
    elif "council_divergence_warn" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(4, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.80))))
    if "aion_outcome_alert" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(2, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.58))))
    elif "aion_outcome_warn" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(4, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.78))))
    if "aion_outcome_stale" in flags:
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, min(3, int(max_open_positions_cap))))
        max_trades_cap_runtime = min(max_trades_cap_runtime, max(1, int(round(int(max_trades_cap) * 0.72))))

    if degraded or (not quality_ok) or regime == "defensive":
        max_open_positions_runtime = min(max_open_positions_runtime, max(1, int(round(int(max_open_positions_cap) * 0.75))))

    max_open_positions_runtime = max(1, min(int(max_open_positions_cap), int(max_open_positions_runtime)))
    max_trades_cap_runtime = max(1, min(int(max_trades_cap), int(max_trades_cap_runtime)))
    return int(max_trades_cap_runtime), int(max_open_positions_runtime)


def _runtime_position_risk_scale(
    ext_runtime_scale: float,
    ext_runtime_diag: dict | None,
):
    """
    Convert runtime overlay diagnostics into a per-position risk scalar.
    Used to throttle sizing (risk_per_trade / max_notional / gross leverage)
    in addition to concurrency caps.
    """
    scale = max(0.35, min(1.00, _safe_float(ext_runtime_scale, 1.0)))
    diag = ext_runtime_diag if isinstance(ext_runtime_diag, dict) else {}
    flags = [str(x).strip().lower() for x in diag.get("flags", [])] if isinstance(diag.get("flags", []), list) else []
    regime = str(diag.get("regime", "")).strip().lower()
    degraded = bool(diag.get("degraded", False))
    quality_ok = bool(diag.get("quality_gate_ok", True))
    overlay_stale = bool(diag.get("overlay_stale", False))

    if degraded:
        scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_DEGRADED_SCALE)))
    if not quality_ok:
        scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_QFAIL_SCALE)))
    if overlay_stale:
        scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_STALE_SCALE)))
    if flags:
        scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_FLAG_SCALE))) ** len(flags)
        if "drift_alert" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_DRIFT_ALERT_SCALE)))
        elif "drift_warn" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_DRIFT_WARN_SCALE)))
        if "quality_governor_step_spike" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_QUALITY_STEP_SCALE)))
        if "fracture_alert" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_FRACTURE_ALERT_SCALE)))
        elif "fracture_warn" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_FRACTURE_WARN_SCALE)))
        if "exec_risk_hard" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_EXEC_HARD_SCALE)))
        elif "exec_risk_tight" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_EXEC_TIGHT_SCALE)))
        if "hive_stress_alert" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_HIVE_ALERT_SCALE)))
        elif "hive_stress_warn" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_HIVE_WARN_SCALE)))
        if "hive_crowding_alert" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_HIVE_ALERT_SCALE)))
        elif "hive_crowding_warn" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_HIVE_WARN_SCALE)))
        if "hive_entropy_alert" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_HIVE_ALERT_SCALE)))
        elif "hive_entropy_warn" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_HIVE_WARN_SCALE)))
        if "hive_turnover_alert" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_HIVE_ALERT_SCALE)))
        elif "hive_turnover_warn" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_HIVE_WARN_SCALE)))
        if "memory_turnover_alert" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_HIVE_ALERT_SCALE)))
        elif "memory_turnover_warn" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_HIVE_WARN_SCALE)))
        if "heartbeat_alert" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_HEARTBEAT_ALERT_SCALE)))
        elif "heartbeat_warn" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_HEARTBEAT_WARN_SCALE)))
        if "council_divergence_alert" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_COUNCIL_ALERT_SCALE)))
        elif "council_divergence_warn" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_COUNCIL_WARN_SCALE)))
        if "aion_outcome_alert" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_AION_OUTCOME_ALERT_SCALE)))
        elif "aion_outcome_warn" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_AION_OUTCOME_WARN_SCALE)))
        if "aion_outcome_stale" in flags:
            scale *= float(max(0.20, min(1.20, cfg.EXT_SIGNAL_RUNTIME_RISK_AION_OUTCOME_STALE_SCALE)))
    if regime == "defensive":
        scale *= 0.90
    return float(max(0.20, min(1.00, scale)))


def _overlay_entry_gate(ext_runtime_diag: dict | None, overlay_age_hours: float | None):
    """
    Decide whether external overlay conditions should hard-block new entries.
    """
    if not bool(getattr(cfg, "EXT_SIGNAL_ENABLED", True)):
        return False, []
    diag = ext_runtime_diag if isinstance(ext_runtime_diag, dict) else {}
    flags = [str(x).strip().lower() for x in diag.get("flags", [])] if isinstance(diag.get("flags", []), list) else []
    quality_ok = bool(diag.get("quality_gate_ok", True))
    overlay_stale = bool(diag.get("overlay_stale", False) or ("overlay_stale" in flags))

    reasons: list[str] = []
    if bool(getattr(cfg, "EXT_SIGNAL_BLOCK_CRITICAL", True)):
        crit = {str(x).strip().lower() for x in getattr(cfg, "EXT_SIGNAL_BLOCK_CRITICAL_FLAGS", []) if str(x).strip()}
        hits = sorted(x for x in flags if x in crit)
        for hit in hits:
            reasons.append(f"critical_flag:{hit}")

    if bool(getattr(cfg, "EXT_SIGNAL_BLOCK_ON_QUALITY_FAIL", False)) and (not quality_ok):
        reasons.append("quality_gate_fail")

    stale_h = _safe_float(getattr(cfg, "EXT_SIGNAL_BLOCK_STALE_HOURS", 0.0), 0.0)
    if stale_h > 0.0 and overlay_stale:
        age = _safe_float(overlay_age_hours, None)
        if age is None or age >= stale_h:
            reasons.append("overlay_stale")

    return bool(reasons), reasons


def _memory_feedback_controls(
    *,
    max_trades_cap_runtime: int,
    max_open_positions_runtime: int,
    risk_per_trade_runtime: float,
    max_position_notional_pct_runtime: float,
    max_gross_leverage_runtime: float,
    memory_feedback: dict | None,
):
    out = {
        "active": False,
        "status": "unknown",
        "reasons": [],
        "risk_scale": 1.0,
        "trades_scale": 1.0,
        "open_scale": 1.0,
        "turnover_pressure": None,
        "turnover_dampener": None,
        "block_new_entries": False,
        "max_trades_cap_runtime": int(max(1, int(max_trades_cap_runtime))),
        "max_open_positions_runtime": int(max(1, int(max_open_positions_runtime))),
        "risk_per_trade_runtime": float(max(1e-5, float(risk_per_trade_runtime))),
        "max_position_notional_pct_runtime": float(max(1e-5, float(max_position_notional_pct_runtime))),
        "max_gross_leverage_runtime": float(max(0.05, float(max_gross_leverage_runtime))),
    }
    if not bool(getattr(cfg, "EXT_SIGNAL_MEMORY_FEEDBACK_ENABLED", True)):
        return out
    if not isinstance(memory_feedback, dict):
        return out

    active = bool(memory_feedback.get("active", False))
    if not active:
        return out

    mn = _safe_float(getattr(cfg, "EXT_SIGNAL_MEMORY_FEEDBACK_MIN_SCALE", 0.70), 0.70)
    mx = _safe_float(getattr(cfg, "EXT_SIGNAL_MEMORY_FEEDBACK_MAX_SCALE", 1.12), 1.12)
    lo = max(0.20, min(mn, mx))
    hi = max(lo, max(mn, mx))
    risk_scale = max(lo, min(hi, _safe_float(memory_feedback.get("risk_scale", 1.0), 1.0)))
    trades_scale = max(0.50, min(1.25, _safe_float(memory_feedback.get("max_trades_scale", 1.0), 1.0)))
    open_scale = max(0.50, min(1.25, _safe_float(memory_feedback.get("max_open_scale", 1.0), 1.0)))
    status = str(memory_feedback.get("status", "unknown")).strip().lower() or "unknown"
    reasons = [str(x).strip().lower() for x in memory_feedback.get("reasons", []) if str(x).strip()]
    reasons = list(dict.fromkeys(reasons))

    out["active"] = True
    out["status"] = status
    out["reasons"] = reasons
    out["risk_scale"] = float(risk_scale)
    out["trades_scale"] = float(trades_scale)
    out["open_scale"] = float(open_scale)
    out["turnover_pressure"] = _safe_float(memory_feedback.get("turnover_pressure"), None)
    out["turnover_dampener"] = _safe_float(memory_feedback.get("turnover_dampener"), None)

    out["max_trades_cap_runtime"] = max(
        1,
        min(
            int(max_trades_cap_runtime),
            int(round(int(max_trades_cap_runtime) * trades_scale)),
        ),
    )
    out["max_open_positions_runtime"] = max(
        1,
        min(
            int(max_open_positions_runtime),
            int(round(int(max_open_positions_runtime) * open_scale)),
        ),
    )
    out["risk_per_trade_runtime"] = max(1e-5, float(risk_per_trade_runtime) * float(risk_scale))
    out["max_position_notional_pct_runtime"] = max(
        1e-5, float(max_position_notional_pct_runtime) * max(0.55, float(risk_scale))
    )
    out["max_gross_leverage_runtime"] = max(
        0.05, float(max_gross_leverage_runtime) * max(0.60, float(risk_scale))
    )

    explicit_block = bool(memory_feedback.get("block_new_entries", False))
    alert_th = _safe_float(getattr(cfg, "EXT_SIGNAL_MEMORY_FEEDBACK_ALERT_THRESHOLD", 0.84), 0.84)
    block_on_alert = bool(getattr(cfg, "EXT_SIGNAL_MEMORY_FEEDBACK_BLOCK_ON_ALERT", False))
    alert_state = bool(status in {"alert", "hard"} or float(risk_scale) <= float(alert_th))
    out["block_new_entries"] = bool(explicit_block or (block_on_alert and alert_state))
    return out


def _aion_feedback_controls(aion_feedback: dict | None):
    out = {
        "active": False,
        "status": "unknown",
        "source": "unknown",
        "source_selected": "unknown",
        "source_preference": "auto",
        "reasons": [],
        "risk_scale": 1.0,
        "closed_trades": 0,
        "hit_rate": None,
        "profit_factor": None,
        "expectancy": None,
        "drawdown_norm": None,
        "age_hours": None,
        "max_age_hours": None,
        "stale": False,
        "last_closed_ts": None,
        "path": "",
        "block_new_entries": False,
    }
    if not bool(getattr(cfg, "EXT_SIGNAL_AION_FEEDBACK_ENABLED", True)):
        return out
    if not isinstance(aion_feedback, dict):
        return out

    active = bool(aion_feedback.get("active", False))
    if not active:
        return out

    status = str(aion_feedback.get("status", "unknown")).strip().lower() or "unknown"
    reasons = [str(x).strip().lower() for x in aion_feedback.get("reasons", []) if str(x).strip()]
    reasons = list(dict.fromkeys(reasons))
    risk_scale = max(0.20, min(1.20, _safe_float(aion_feedback.get("risk_scale", 1.0), 1.0)))
    closed_trades = max(0, _safe_int(aion_feedback.get("closed_trades"), 0))
    hit_rate = _safe_float(aion_feedback.get("hit_rate"), None)
    profit_factor = _safe_float(aion_feedback.get("profit_factor"), None)
    expectancy = _safe_float(aion_feedback.get("expectancy"), None)
    drawdown_norm = _safe_float(aion_feedback.get("drawdown_norm"), None)
    age_hours = _safe_float(aion_feedback.get("age_hours"), None)
    max_age_hours = max(0.0, _safe_float(aion_feedback.get("max_age_hours"), _safe_float(getattr(cfg, "EXT_SIGNAL_AION_FEEDBACK_MAX_AGE_HOURS", 72.0), 72.0)))
    stale = bool(aion_feedback.get("stale", False))
    if (not stale) and age_hours is not None and max_age_hours > 0.0:
        stale = bool(age_hours > max_age_hours)
    last_closed_ts = str(aion_feedback.get("last_closed_ts", "")).strip() or None
    path = str(aion_feedback.get("path", "")).strip()
    source = str(aion_feedback.get("source", aion_feedback.get("source_selected", ""))).strip().lower() or "unknown"
    source_selected = str(aion_feedback.get("source_selected", source)).strip().lower() or source
    source_preference = str(aion_feedback.get("source_preference", "auto")).strip().lower() or "auto"

    out["active"] = True
    out["status"] = status
    out["source"] = source
    out["source_selected"] = source_selected
    out["source_preference"] = source_preference
    out["reasons"] = list(reasons)
    out["risk_scale"] = float(risk_scale)
    out["closed_trades"] = int(closed_trades)
    out["hit_rate"] = hit_rate
    out["profit_factor"] = profit_factor
    out["expectancy"] = expectancy
    out["drawdown_norm"] = drawdown_norm
    out["age_hours"] = age_hours
    out["max_age_hours"] = max_age_hours if max_age_hours > 0.0 else None
    out["stale"] = bool(stale)
    out["last_closed_ts"] = last_closed_ts
    out["path"] = path

    if stale and "stale_feedback" not in out["reasons"]:
        out["reasons"].append("stale_feedback")
    if stale and bool(getattr(cfg, "EXT_SIGNAL_AION_FEEDBACK_IGNORE_STALE", True)):
        out["status"] = "stale"
        out["risk_scale"] = 1.0
        out["block_new_entries"] = False
        return out

    explicit_block = bool(aion_feedback.get("block_new_entries", False))
    alert_th = _safe_float(getattr(cfg, "EXT_SIGNAL_AION_FEEDBACK_ALERT_THRESHOLD", 0.82), 0.82)
    block_on_alert = bool(getattr(cfg, "EXT_SIGNAL_AION_FEEDBACK_BLOCK_ON_ALERT", False))
    min_closed = max(1, _safe_int(getattr(cfg, "EXT_SIGNAL_AION_FEEDBACK_MIN_CLOSED_TRADES", 8), 8))
    alert_state = bool(status in {"alert", "hard"} or float(risk_scale) <= float(alert_th))
    enough_closed = bool(closed_trades >= min_closed)
    out["block_new_entries"] = bool(explicit_block or (block_on_alert and alert_state and enough_closed))
    return out


def _execution_quality_governor(
    *,
    max_trades_per_day: int,
    max_open_positions: int,
    risk_per_trade: float,
    max_position_notional_pct: float,
    max_gross_leverage: float,
    monitor: RuntimeMonitor | None,
    now_utc: dt.datetime | None = None,
):
    """
    Use recent fills (slippage + execution rate) as a live execution/turnover governor.
    """
    out = {
        "state": "off",
        "reasons": [],
        "recent_executions": 0,
        "exec_rate_per_min": 0.0,
        "avg_slippage_bps": None,
        "p90_slippage_bps": None,
        "block_new_entries": False,
        "max_trades_per_day": int(max(1, int(max_trades_per_day))),
        "max_open_positions": int(max(1, int(max_open_positions))),
        "risk_per_trade": float(max(1e-5, float(risk_per_trade))),
        "max_position_notional_pct": float(max(1e-5, float(max_position_notional_pct))),
        "max_gross_leverage": float(max(0.05, float(max_gross_leverage))),
    }
    if not bool(getattr(cfg, "EXEC_GOVERNOR_ENABLED", True)):
        return out

    mstate = getattr(monitor, "state", {}) if monitor is not None else {}
    events = mstate.get("execution_events", []) if isinstance(mstate, dict) else []
    lookback_min = max(1, int(getattr(cfg, "EXEC_GOVERNOR_LOOKBACK_MIN", 25)))
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    recent_slip = []
    if isinstance(events, list):
        for evt in events:
            if not isinstance(evt, dict):
                continue
            ts_raw = evt.get("ts")
            slip = _safe_float(evt.get("slippage_bps"), None)
            if slip is None:
                continue
            try:
                ts = dt.datetime.fromisoformat(str(ts_raw))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=dt.timezone.utc)
                else:
                    ts = ts.astimezone(dt.timezone.utc)
            except Exception:
                continue
            if (now - ts).total_seconds() <= lookback_min * 60:
                recent_slip.append(float(slip))

    # Backward-compatible fallback if execution_events are unavailable.
    if not recent_slip and isinstance(mstate, dict):
        sp = mstate.get("slippage_points", [])
        if isinstance(sp, list) and sp:
            tail_n = min(len(sp), max(1, int(getattr(cfg, "EXEC_GOVERNOR_MIN_EXECUTIONS", 6))))
            for raw in sp[-tail_n:]:
                v = _safe_float(raw, None)
                if v is not None:
                    recent_slip.append(float(v))

    n = int(len(recent_slip))
    out["recent_executions"] = n
    out["exec_rate_per_min"] = float(n / max(1, lookback_min))
    if n:
        out["avg_slippage_bps"] = float(sum(recent_slip) / n)
        try:
            arr = sorted(float(x) for x in recent_slip)
            pidx = int(round(0.90 * (len(arr) - 1)))
            out["p90_slippage_bps"] = float(arr[max(0, min(len(arr) - 1, pidx))])
        except Exception:
            out["p90_slippage_bps"] = float(out["avg_slippage_bps"])

    if n < max(1, int(getattr(cfg, "EXEC_GOVERNOR_MIN_EXECUTIONS", 6))):
        out["state"] = "insufficient_data"
        return out

    warn_slip = float(getattr(cfg, "EXEC_GOVERNOR_SLIP_WARN_BPS", 20.0))
    alert_slip = float(getattr(cfg, "EXEC_GOVERNOR_SLIP_ALERT_BPS", 28.0))
    warn_rate = float(getattr(cfg, "EXEC_GOVERNOR_RATE_WARN_PER_MIN", 0.60))
    alert_rate = float(getattr(cfg, "EXEC_GOVERNOR_RATE_ALERT_PER_MIN", 1.20))

    avg_slip = _safe_float(out.get("avg_slippage_bps"), 0.0)
    p90_slip = _safe_float(out.get("p90_slippage_bps"), avg_slip)
    rate = _safe_float(out.get("exec_rate_per_min"), 0.0)

    state = "ok"
    reasons = []
    if avg_slip >= alert_slip or p90_slip >= (alert_slip * 1.10) or rate >= alert_rate:
        state = "alert"
    elif avg_slip >= warn_slip or p90_slip >= (warn_slip * 1.15) or rate >= warn_rate:
        state = "warn"

    if avg_slip >= warn_slip:
        reasons.append("slippage_elevated")
    if p90_slip >= (warn_slip * 1.15):
        reasons.append("slippage_tail_elevated")
    if rate >= warn_rate:
        reasons.append("execution_rate_elevated")

    out["state"] = state
    out["reasons"] = reasons
    if state == "warn":
        tr_scale = _safe_float(getattr(cfg, "EXEC_GOVERNOR_WARN_TRADES_SCALE", 0.80), 0.80)
        op_scale = _safe_float(getattr(cfg, "EXEC_GOVERNOR_WARN_OPEN_SCALE", 0.80), 0.80)
        rk_scale = _safe_float(getattr(cfg, "EXEC_GOVERNOR_WARN_RISK_SCALE", 0.86), 0.86)
    elif state == "alert":
        tr_scale = _safe_float(getattr(cfg, "EXEC_GOVERNOR_ALERT_TRADES_SCALE", 0.60), 0.60)
        op_scale = _safe_float(getattr(cfg, "EXEC_GOVERNOR_ALERT_OPEN_SCALE", 0.60), 0.60)
        rk_scale = _safe_float(getattr(cfg, "EXEC_GOVERNOR_ALERT_RISK_SCALE", 0.72), 0.72)
    else:
        tr_scale = 1.0
        op_scale = 1.0
        rk_scale = 1.0

    out["max_trades_per_day"] = max(1, min(int(max_trades_per_day), int(round(int(max_trades_per_day) * max(0.20, min(1.20, tr_scale))))))
    out["max_open_positions"] = max(1, min(int(max_open_positions), int(round(int(max_open_positions) * max(0.20, min(1.20, op_scale))))))
    rk = max(0.20, min(1.20, rk_scale))
    out["risk_per_trade"] = max(1e-5, float(risk_per_trade) * rk)
    out["max_position_notional_pct"] = max(1e-5, float(max_position_notional_pct) * max(0.50, rk))
    out["max_gross_leverage"] = max(0.05, float(max_gross_leverage) * max(0.55, rk))
    out["block_new_entries"] = bool(state == "alert" and bool(getattr(cfg, "EXEC_GOVERNOR_BLOCK_ON_ALERT", False)))
    return out


def _daily_loss_limits_hit(caps: dict, day_start_equity: float, equity: float):
    start = max(1.0, _safe_float(day_start_equity, 0.0))
    eq = _safe_float(equity, start)
    daily_loss_abs = max(0.0, start - eq)
    daily_loss_pct = daily_loss_abs / start
    lim_abs = _safe_float(caps.get("daily_loss_limit_abs"), None)
    lim_pct = _safe_float(caps.get("daily_loss_limit_pct"), None)
    hit_abs = bool(lim_abs is not None and lim_abs > 0 and daily_loss_abs >= lim_abs)
    hit_pct = bool(lim_pct is not None and lim_pct > 0 and daily_loss_pct >= lim_pct)
    return hit_abs or hit_pct, daily_loss_abs, daily_loss_pct


def _normalize_position(raw: dict):
    if not isinstance(raw, dict):
        return None
    side = str(raw.get("side", "")).upper()
    if side not in {"LONG", "SHORT"}:
        return None
    qty = _safe_int(raw.get("qty"), 0)
    if qty <= 0:
        return None
    entry = _safe_float(raw.get("entry"), 0.0)
    if entry <= 0:
        return None

    out = {
        "qty": qty,
        "side": side,
        "entry": entry,
        "entry_ts": str(raw.get("entry_ts", "")),
        "mark_price": _safe_float(raw.get("mark_price"), entry),
        "stop": _safe_float(raw.get("stop"), entry),
        "target": _safe_float(raw.get("target"), entry),
        "trail_stop": _safe_float(raw.get("trail_stop"), entry),
        "trail_mult": max(0.1, _safe_float(raw.get("trail_mult"), cfg.TRAILING_STOP_ATR_MULTIPLE)),
        "init_risk": max(1e-6, _safe_float(raw.get("init_risk"), entry * 0.0035)),
        "bars_held": max(0, _safe_int(raw.get("bars_held"), 0)),
        "partial_taken": bool(raw.get("partial_taken", False)),
        "peak_price": _safe_float(raw.get("peak_price"), entry),
        "trough_price": _safe_float(raw.get("trough_price"), entry),
        "confidence": _safe_float(raw.get("confidence"), 0.0),
        "regime": str(raw.get("regime", "mixed")),
        "atr_pct": max(0.0, _safe_float(raw.get("atr_pct"), 0.0)),
        "stop_atr_mult": max(0.1, _safe_float(raw.get("stop_atr_mult"), getattr(cfg, "STOP_ATR_LONG", 1.0))),
        "stop_vol_expanded": bool(raw.get("stop_vol_expanded", False)),
    }
    return out


def load_runtime_state(today: str):
    if not cfg.RESTORE_RUNTIME_STATE:
        return None
    if not RUNTIME_STATE_FILE.exists():
        return None
    try:
        payload = json.loads(RUNTIME_STATE_FILE.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("day", "")) != today:
        return None

    open_positions = {}
    raw_pos = payload.get("open_positions", {})
    if isinstance(raw_pos, dict):
        for sym, pos in raw_pos.items():
            n = _normalize_position(pos)
            if n is not None:
                open_positions[str(sym).upper()] = n

    cooldown = {}
    raw_cd = payload.get("cooldown", {})
    if isinstance(raw_cd, dict):
        for sym, val in raw_cd.items():
            v = _safe_int(val, 0)
            if v > 0:
                cooldown[str(sym).upper()] = v

    return {
        "cash": _safe_float(payload.get("cash"), cfg.EQUITY_START),
        "closed_pnl": _safe_float(payload.get("closed_pnl"), 0.0),
        "trades_today": max(0, _safe_int(payload.get("trades_today"), 0)),
        "open_positions": open_positions,
        "cooldown": cooldown,
    }


def save_runtime_state(today: str, cash: float, closed_pnl: float, trades_today: int, open_positions: dict, cooldown: dict):
    payload = {
        "saved_at": dt.datetime.now().isoformat(),
        "day": today,
        "cash": float(cash),
        "closed_pnl": float(closed_pnl),
        "trades_today": int(trades_today),
        "open_positions": open_positions,
        "cooldown": cooldown,
    }
    try:
        RUNTIME_STATE_FILE.write_text(json.dumps(payload, indent=2, default=str))
    except Exception:
        pass


def save_runtime_controls(payload: dict):
    try:
        RUNTIME_CONTROLS_FILE.write_text(json.dumps(payload, indent=2, default=str))
    except Exception:
        pass


def save_runtime_controls_heartbeat(
    *,
    day_key: str,
    trades_today: int,
    open_positions: dict,
    watchlist_size: int = 0,
    status: str = "running",
):
    save_runtime_controls(
        {
            "ts": dt.datetime.now().isoformat(),
            "day": str(day_key),
            "trading_mode": str(getattr(cfg, "TRADING_MODE", "long_term")),
            "hist_bar_size": str(getattr(cfg, "HIST_BAR_SIZE", "")),
            "hist_duration": str(getattr(cfg, "HIST_DURATION", "")),
            "hist_use_rth": bool(getattr(cfg, "HIST_USE_RTH", True)),
            "loop_seconds": int(cfg.LOOP_SECONDS),
            "watchlist_size": max(0, _safe_int(watchlist_size, 0)),
            "trades_today": max(0, _safe_int(trades_today, 0)),
            "open_positions": max(0, len(open_positions) if isinstance(open_positions, dict) else 0),
            "max_trades_cap_runtime": int(cfg.MAX_TRADES_PER_DAY),
            "max_open_positions_runtime": int(cfg.MAX_OPEN_POSITIONS),
            "risk_per_trade_runtime": float(cfg.RISK_PER_TRADE),
            "max_position_notional_pct_runtime": float(cfg.MAX_POSITION_NOTIONAL_PCT),
            "max_gross_leverage_runtime": float(cfg.MAX_GROSS_LEVERAGE),
            "heartbeat_status": str(status or "running"),
        }
    )


def _position_pnl(position: dict, mark_price: float) -> float:
    qty = position["qty"]
    if position["side"] == "LONG":
        return (mark_price - position["entry"]) * qty
    return (position["entry"] - mark_price) * qty


def _mark_open_pnl(open_positions: dict, last_prices: dict) -> float:
    open_pnl = 0.0
    for sym, pos in open_positions.items():
        px = last_prices.get(sym, pos.get("mark_price", pos["entry"]))
        pos["mark_price"] = px
        open_pnl += _position_pnl(pos, px)
    return open_pnl


def _equity_from_cash_and_positions(cash: float, open_positions: dict, last_prices: dict) -> float:
    equity = float(cash)
    for sym, pos in open_positions.items():
        px = float(last_prices.get(sym, pos.get("mark_price", pos["entry"])))
        qty = int(pos.get("qty", 0))
        if qty <= 0:
            continue
        if pos.get("side") == "LONG":
            equity += px * qty
        else:
            equity -= px * qty
    return equity


def _partial_profit_target_price(entry: float, init_risk: float, side: str, r_multiple: float) -> float:
    r = max(1e-9, float(init_risk))
    m = max(0.1, float(r_multiple))
    if str(side).upper() == "LONG":
        return float(entry + r * m)
    return float(entry - r * m)


def _partial_close_qty(qty: int, fraction: float) -> int:
    q = max(0, int(qty))
    if q <= 0:
        return 0
    f = max(0.05, min(0.95, float(fraction)))
    close_qty = int(math.floor(q * f))
    close_qty = max(1, close_qty)
    if q > 1:
        close_qty = min(close_qty, q - 1)
    return max(0, close_qty)


def _trailing_stop_candidate(side: str, extreme_price: float, atr_value: float, atr_multiple: float) -> float:
    atr = max(1e-9, float(atr_value))
    mult = max(0.1, float(atr_multiple))
    if str(side).upper() == "LONG":
        return float(extreme_price - atr * mult)
    return float(extreme_price + atr * mult)


def _entry_stop_atr_multiple(side: str, close_returns) -> tuple[float, bool]:
    side_up = str(side).upper()
    if side_up == "SHORT":
        base = float(getattr(cfg, "STOP_ATR_SHORT", getattr(cfg, "STOP_ATR_MULT", 1.0)))
    else:
        base = float(getattr(cfg, "STOP_ATR_LONG", getattr(cfg, "STOP_ATR_MULT", 1.0)))
    base = max(0.1, base)

    if not bool(getattr(cfg, "STOP_VOL_ADAPTIVE", True)):
        return base, False

    x = np.asarray(close_returns, float).ravel()
    x = x[np.isfinite(x)]
    lookback = max(5, int(getattr(cfg, "STOP_VOL_LOOKBACK", 20)))
    if x.size < max(lookback + 5, 30):
        return base, False

    roll_vals = []
    for i in range(lookback - 1, len(x)):
        seg = x[i - lookback + 1 : i + 1]
        sd = float(np.std(seg, ddof=1)) if len(seg) > 1 else 0.0
        if math.isfinite(sd):
            roll_vals.append(sd)
    if len(roll_vals) < max(lookback, 20):
        return base, False

    rv = np.asarray(roll_vals, float)
    hist = rv[-min(252, len(rv)) :]
    curr = float(rv[-1])
    p90 = float(np.percentile(hist, 90.0))
    if (not math.isfinite(curr)) or (not math.isfinite(p90)) or p90 <= 0:
        return base, False
    if curr <= p90:
        return base, False

    expand = float(getattr(cfg, "STOP_VOL_EXPANSION_MULT", 1.3))
    return max(0.1, base * max(1.0, expand)), True


def _effective_stop_price(pos: dict) -> tuple[float, bool]:
    base = float(pos.get("stop", pos.get("entry", 0.0)))
    side = str(pos.get("side", "")).upper()
    trailing_active = bool(getattr(cfg, "TRAILING_STOP_ENABLED", True) and bool(pos.get("partial_taken", False)))
    if not trailing_active:
        return base, False

    trail = float(pos.get("trail_stop", base))
    if side == "LONG":
        return max(base, trail), True
    if side == "SHORT":
        return min(base, trail), True
    return base, False


def _hours_since_entry(pos: dict) -> float | None:
    raw = str(pos.get("entry_ts", "")).strip()
    if not raw:
        return None
    try:
        ts = dt.datetime.fromisoformat(raw)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    now_utc = dt.datetime.now(dt.timezone.utc)
    return max(0.0, float((now_utc - ts.astimezone(dt.timezone.utc)).total_seconds() / 3600.0))


def _telemetry_write(telemetry_sink: DecisionTelemetry | None, record: dict):
    if telemetry_sink is None or (not bool(getattr(cfg, "TELEMETRY_ENABLED", True))):
        return
    try:
        telemetry_sink.write(record)
    except Exception as exc:
        log_run(f"Telemetry write exception: {exc}")


def _exit_decision_from_reason(reason: str) -> str:
    key = str(reason or "").strip().lower()
    mapping = {
        "target_hit": "EXIT_TARGET_HIT",
        "initial_stop": "EXIT_INITIAL_STOP",
        "trailing_stop": "EXIT_TRAILING_STOP",
        "time_stop": "EXIT_TIME_STOP",
        "opposite_high_confidence_signal": "EXIT_SIGNAL_FLIP",
    }
    return mapping.get(key, "EXIT")


def _telemetry_log_trade_decision(
    telemetry_sink: DecisionTelemetry | None,
    *,
    symbol: str,
    decision: str,
    q_overlay_bias: float | None = None,
    q_overlay_confidence: float | None = None,
    confluence_score: float | None = None,
    intraday_alignment_score: float | None = None,
    regime: str | None = None,
    governor_compound_scalar: float | None = None,
    entry_price: float | None = None,
    stop_price: float | None = None,
    risk_distance: float | None = None,
    position_size_shares: int | None = None,
    book_imbalance=None,
    reasons: list[str] | None = None,
    pnl_realized: float | None = None,
    slippage_bps: float | None = None,
    estimated_slippage_bps: float | None = None,
    extras: dict | None = None,
):
    if telemetry_sink is None or (not bool(getattr(cfg, "TELEMETRY_ENABLED", True))):
        return

    rs = [str(x).strip() for x in (reasons or []) if str(x).strip()]
    risk = risk_distance
    if (risk is None) and (entry_price is not None) and (stop_price is not None):
        risk = abs(_safe_float(entry_price, 0.0) - _safe_float(stop_price, 0.0))

    rec = {
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": str(symbol or "").upper(),
        "decision": str(decision or "").upper(),
        "q_overlay_bias": float(_safe_float(q_overlay_bias, 0.0)),
        "q_overlay_confidence": float(max(0.0, min(1.0, _safe_float(q_overlay_confidence, 0.0)))),
        "confluence_score": float(max(0.0, min(1.0, _safe_float(confluence_score, 0.0)))),
        "intraday_alignment_score": float(max(0.0, min(1.0, _safe_float(intraday_alignment_score, 0.0)))),
        "regime": str(regime or "unknown").strip().lower() or "unknown",
        "governor_compound_scalar": (
            None
            if governor_compound_scalar is None
            else float(max(0.0, min(2.0, _safe_float(governor_compound_scalar, 1.0))))
        ),
        "entry_price": (None if entry_price is None else float(_safe_float(entry_price, 0.0))),
        "stop_price": (None if stop_price is None else float(_safe_float(stop_price, 0.0))),
        "risk_distance": (None if risk is None else float(max(0.0, _safe_float(risk, 0.0)))),
        "position_size_shares": (None if position_size_shares is None else int(max(0, _safe_int(position_size_shares, 0)))),
        "book_imbalance": book_imbalance,
        "reasons": rs,
    }
    if pnl_realized is not None:
        rec["pnl_realized"] = float(_safe_float(pnl_realized, 0.0))
    if slippage_bps is not None:
        rec["slippage_bps"] = float(_safe_float(slippage_bps, 0.0))
    if estimated_slippage_bps is not None:
        rec["estimated_slippage_bps"] = float(_safe_float(estimated_slippage_bps, 0.0))
    if isinstance(extras, dict) and extras:
        rec["extras"] = dict(extras)
    _telemetry_write(telemetry_sink, rec)


def _maybe_update_telemetry_summary(last_update_mono: float) -> float:
    if not bool(getattr(cfg, "TELEMETRY_ENABLED", True)):
        return float(last_update_mono)
    now_mono = float(time.monotonic())
    interval = max(30.0, float(getattr(cfg, "LOOP_SECONDS", 30)))
    if (now_mono - float(last_update_mono)) < interval:
        return float(last_update_mono)
    try:
        decisions_path = Path(cfg.STATE_DIR) / str(getattr(cfg, "TELEMETRY_DECISIONS_FILE", "trade_decisions.jsonl"))
        output_path = Path(getattr(cfg, "TELEMETRY_SUMMARY_FILE", Path(cfg.STATE_DIR) / "telemetry_summary.json"))
        window = int(max(1, _safe_int(getattr(cfg, "TELEMETRY_SUMMARY_WINDOW", 20), 20)))
        write_telemetry_summary(
            decisions_path=decisions_path,
            output_path=output_path,
            rolling_window=window,
        )
    except Exception as exc:
        log_run(f"Telemetry summary refresh exception: {exc}")
    return now_mono


def _close_position(
    open_positions: dict,
    symbol: str,
    fill_price: float,
    reason: str,
    cash: float,
    closed_pnl: float,
    ks: KillSwitch,
    today: str,
    fill_ratio: float = 1.0,
    slippage_bps: float = 0.0,
    monitor: RuntimeMonitor | None = None,
    memory_runtime_context: dict | None = None,
    telemetry_sink: DecisionTelemetry | None = None,
    governor_compound_scalar: float | None = None,
    filled_qty: int | None = None,
    sync_shadow: bool = True,
):
    pos = open_positions[symbol]
    prev_qty = int(pos["qty"])
    qty = int(prev_qty if filled_qty is None else max(0, min(prev_qty, int(filled_qty))))
    if qty <= 0:
        return cash, closed_pnl

    if pos["side"] == "LONG":
        pnl = (fill_price - pos["entry"]) * qty
        cash += qty * fill_price
        side = "EXIT_SELL"
        shadow_action = "SELL"
    else:
        pnl = (pos["entry"] - fill_price) * qty
        cash -= qty * fill_price
        side = "EXIT_BUY"
        shadow_action = "BUY"

    closed_pnl += pnl
    ks.register_trade(pnl, today)
    actual_r = float(pnl / max(1e-9, float(pos.get("init_risk", 1e-6)) * max(1, qty)))
    hrs = _hours_since_entry(pos)

    log_trade(
        now(),
        symbol,
        side,
        qty,
        pos["entry"],
        fill_price,
        pnl,
        reason,
        confidence=float(pos.get("confidence", 0.0)),
        regime=str(pos.get("regime", "")),
        stop=float(pos.get("stop", 0.0)),
        target=float(pos.get("target", 0.0)),
        trail=float(pos.get("trail_stop", 0.0)),
        fill_ratio=fill_ratio,
        slippage_bps=slippage_bps,
    )
    event_extra = {
        "fill_ratio": float(fill_ratio),
        "slippage_bps": float(slippage_bps),
        "stop": float(pos.get("stop", 0.0)),
        "target": float(pos.get("target", 0.0)),
        "trail": float(pos.get("trail_stop", 0.0)),
        "bars_held": int(pos.get("bars_held", 0)),
        "r_captured": float(actual_r),
        "time_in_trade_hours": hrs,
    }
    reason_key = str(reason).strip().lower()
    if reason_key in {"trailing_stop", "initial_stop", "target_hit", "time_stop", "opposite_high_confidence_signal"}:
        event_extra["type"] = reason_key
    if isinstance(memory_runtime_context, dict) and memory_runtime_context:
        event_extra["runtime_context"] = dict(memory_runtime_context)

    _emit_trade_memory_event(
        event_type="trade.exit",
        symbol=symbol,
        side=str(pos["side"]),
        qty=qty,
        entry=float(pos["entry"]),
        exit_=float(fill_price),
        pnl=float(pnl),
        reason=str(reason),
        confidence=float(pos.get("confidence", 0.0)),
        regime=str(pos.get("regime", "")),
        monitor=monitor,
        extra=event_extra,
    )
    _telemetry_log_trade_decision(
        telemetry_sink,
        symbol=symbol,
        decision=_exit_decision_from_reason(reason),
        q_overlay_bias=_safe_float(pos.get("q_overlay_bias"), 0.0),
        q_overlay_confidence=_safe_float(pos.get("q_overlay_confidence"), 0.0),
        confluence_score=_safe_float(pos.get("confidence"), 0.0),
        intraday_alignment_score=_safe_float(pos.get("intraday_score"), 0.0),
        regime=str(pos.get("regime", "unknown")),
        governor_compound_scalar=governor_compound_scalar,
        entry_price=float(pos.get("entry", 0.0)),
        stop_price=float(pos.get("stop", 0.0)),
        risk_distance=float(pos.get("init_risk", 0.0)),
        position_size_shares=int(qty),
        reasons=[str(reason)],
        pnl_realized=float(pnl),
        slippage_bps=float(slippage_bps),
        estimated_slippage_bps=float(slippage_bps),
        extras={
            "fill_ratio": float(fill_ratio),
            "r_captured": float(actual_r),
            "time_in_trade_hours": hrs,
            "remaining_qty": int(max(0, prev_qty - qty)),
        },
    )
    if bool(sync_shadow):
        try:
            apply_shadow_fill(
                Path(cfg.STATE_DIR) / "shadow_trades.json",
                symbol=str(symbol).upper(),
                action=str(shadow_action),
                filled_qty=int(qty),
                avg_fill_price=float(fill_price),
                timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
        except Exception as exc:
            log_run(f"shadow update failed on close {symbol}: {exc}")
    remaining = max(0, prev_qty - qty)
    if remaining > 0:
        pos["qty"] = int(remaining)
        return cash, closed_pnl

    del open_positions[symbol]
    return cash, closed_pnl


def _partial_close(
    pos: dict,
    symbol: str,
    price: float,
    exe: ExecutionSimulator,
    cash: float,
    closed_pnl: float,
    ks: KillSwitch,
    today: str,
    monitor: RuntimeMonitor | None = None,
    memory_runtime_context: dict | None = None,
    telemetry_sink: DecisionTelemetry | None = None,
    governor_compound_scalar: float | None = None,
    ib_client=None,
):
    qty = int(pos["qty"])
    close_qty = _partial_close_qty(qty, float(getattr(cfg, "PARTIAL_PROFIT_FRACTION", cfg.PARTIAL_CLOSE_FRACTION)))
    if close_qty <= 0:
        return cash, closed_pnl, False

    side = "SELL" if pos["side"] == "LONG" else "BUY"
    intent = {
        "event": "ORDER_INTENT",
        "symbol": str(symbol).upper(),
        "side": str(side).upper(),
        "qty": int(close_qty),
        "ref_price": float(price),
        "reason": "partial_profit_1R",
    }
    audit_log(intent, log_dir=Path(cfg.STATE_DIR))
    allowed, gate_reason, gate_diag = _run_exposure_gate(
        client=ib_client,
        symbol=symbol,
        side=side,
        qty=close_qty,
        price=float(price),
        fallback_nlv=max(1.0, float(abs(cash) + abs(pos.get("qty", 0)) * abs(price))),
    )
    gate_event = "EXPOSURE_GATE_PASS" if allowed else "EXPOSURE_GATE_VETO"
    audit_log(
        {
            "event": gate_event,
            "symbol": str(symbol).upper(),
            "side": str(side).upper(),
            "qty": int(close_qty),
            "reason": str(gate_reason),
            **gate_diag,
        },
        log_dir=Path(cfg.STATE_DIR),
    )
    if not allowed:
        log_run(f"EXPOSURE GATE VETO: {symbol} {side} qty={close_qty} reason={gate_reason}")
        send_alert(f"Exposure gate vetoed {symbol} partial close: {gate_reason}", level="WARNING")
        return cash, closed_pnl, False

    audit_log(
        {
            "event": "ORDER_SUBMITTED",
            "symbol": str(symbol).upper(),
            "side": str(side).upper(),
            "qty": int(close_qty),
            "reason": "partial_profit_1R",
        },
        log_dir=Path(cfg.STATE_DIR),
    )
    fill = _execute_fill(
        exe,
        side=side,
        qty=close_qty,
        ref_price=price,
        atr_pct=float(pos.get("atr_pct", 0.0)),
        confidence=float(pos.get("confidence", 0.5)),
        allow_partial=False,
        symbol=str(symbol),
        ib_client=ib_client,
    )
    filled_qty = int(max(0, min(close_qty, int(getattr(fill, "filled_qty", 0)))))
    if filled_qty <= 0:
        audit_log(
            {
                "event": "ORDER_REJECTED",
                "symbol": str(symbol).upper(),
                "side": str(side).upper(),
                "qty": int(close_qty),
                "reason": "partial_profit_1R",
            },
            log_dir=Path(cfg.STATE_DIR),
        )
        return cash, closed_pnl, False

    if str(getattr(fill, "source", "simulator")).strip().lower() != "ib_paper":
        try:
            apply_shadow_fill(
                Path(cfg.STATE_DIR) / "shadow_trades.json",
                symbol=str(symbol).upper(),
                action=str(side).upper(),
                filled_qty=int(filled_qty),
                avg_fill_price=float(fill.avg_fill),
                timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
        except Exception as exc:
            log_run(f"shadow update failed on partial close {symbol}: {exc}")

    if pos["side"] == "LONG":
        pnl = (fill.avg_fill - pos["entry"]) * filled_qty
        cash += filled_qty * fill.avg_fill
        event = "PARTIAL_SELL"
    else:
        pnl = (pos["entry"] - fill.avg_fill) * filled_qty
        cash -= filled_qty * fill.avg_fill
        event = "PARTIAL_BUY"

    closed_pnl += pnl
    ks.register_trade(pnl, today)

    pos["qty"] -= filled_qty
    pos["partial_taken"] = True
    actual_r = float((fill.avg_fill - pos["entry"]) / max(1e-9, pos.get("init_risk", 1e-6)))
    if str(pos["side"]).upper() == "SHORT":
        actual_r = float((pos["entry"] - fill.avg_fill) / max(1e-9, pos.get("init_risk", 1e-6)))
    hrs = _hours_since_entry(pos)

    log_trade(
        now(),
        symbol,
        event,
        filled_qty,
        pos["entry"],
        fill.avg_fill,
        pnl,
        "partial_profit_1R",
        confidence=float(pos.get("confidence", 0.0)),
        regime=str(pos.get("regime", "")),
        stop=float(pos.get("stop", 0.0)),
        target=float(pos.get("target", 0.0)),
        trail=float(pos.get("trail_stop", 0.0)),
        fill_ratio=fill.fill_ratio,
        slippage_bps=fill.est_slippage_bps,
    )
    event_extra = {
        "fill_ratio": float(fill.fill_ratio),
        "slippage_bps": float(fill.est_slippage_bps),
        "stop": float(pos.get("stop", 0.0)),
        "target": float(pos.get("target", 0.0)),
        "trail": float(pos.get("trail_stop", 0.0)),
        "bars_held": int(pos.get("bars_held", 0)),
        "remaining_qty": int(pos.get("qty", 0)),
        "r_captured": float(actual_r),
        "time_in_trade_hours": hrs,
        "type": "partial_profit",
    }
    if isinstance(memory_runtime_context, dict) and memory_runtime_context:
        event_extra["runtime_context"] = dict(memory_runtime_context)

    _emit_trade_memory_event(
        event_type="trade.partial_exit",
        symbol=symbol,
        side=str(pos["side"]),
        qty=int(filled_qty),
        entry=float(pos["entry"]),
        exit_=float(fill.avg_fill),
        pnl=float(pnl),
        reason="partial_profit_1R",
        confidence=float(pos.get("confidence", 0.0)),
        regime=str(pos.get("regime", "")),
        monitor=monitor,
        extra=event_extra,
    )
    _telemetry_log_trade_decision(
        telemetry_sink,
        symbol=symbol,
        decision="PARTIAL_PROFIT_1R",
        q_overlay_bias=_safe_float(pos.get("q_overlay_bias"), 0.0),
        q_overlay_confidence=_safe_float(pos.get("q_overlay_confidence"), 0.0),
        confluence_score=_safe_float(pos.get("confidence"), 0.0),
        intraday_alignment_score=_safe_float(pos.get("intraday_score"), 0.0),
        regime=str(pos.get("regime", "unknown")),
        governor_compound_scalar=governor_compound_scalar,
        entry_price=float(pos.get("entry", 0.0)),
        stop_price=float(pos.get("stop", 0.0)),
        risk_distance=float(pos.get("init_risk", 0.0)),
        position_size_shares=int(filled_qty),
        reasons=["partial_profit_1R"],
        pnl_realized=float(pnl),
        slippage_bps=float(fill.est_slippage_bps),
        estimated_slippage_bps=float(fill.est_slippage_bps),
        extras={
            "fill_ratio": float(fill.fill_ratio),
            "remaining_qty": int(pos.get("qty", 0)),
            "r_captured": float(actual_r),
            "time_in_trade_hours": hrs,
        },
    )
    audit_log(
        {
            "event": "ORDER_PARTIAL_FILL" if filled_qty < close_qty else "ORDER_FILLED",
            "symbol": str(symbol).upper(),
            "side": str(side).upper(),
            "qty_requested": int(close_qty),
            "qty_filled": int(filled_qty),
            "avg_fill_price": float(fill.avg_fill),
            "reason": "partial_profit_1R",
        },
        log_dir=Path(cfg.STATE_DIR),
    )

    fully_closed = pos["qty"] <= 0
    return cash, closed_pnl, fully_closed


def main() -> int:
    if str(getattr(cfg, "TRADING_MODE", "long_term")).strip().lower() == "day_skimmer":
        from .skimmer_loop import run_day_skimmer_loop

        return int(run_day_skimmer_loop())

    equity = cfg.EQUITY_START
    cash = cfg.EQUITY_START
    open_positions = {}
    cooldown = {}
    closed_pnl = 0.0
    trades_today = 0
    day_key = dt.datetime.now().strftime("%Y-%m-%d")
    day_start_equity = float(equity)

    restored = load_runtime_state(day_key)
    if restored:
        cash = float(restored["cash"])
        closed_pnl = float(restored["closed_pnl"])
        trades_today = int(restored["trades_today"])
        open_positions = dict(restored["open_positions"])
        cooldown = dict(restored["cooldown"])
        log_run(
            f"Runtime state restored: positions={len(open_positions)} trades_today={trades_today} cash={cash:.2f}"
        )
        day_start_equity = float(_equity_from_cash_and_positions(cash, open_positions, {}))

    ks = KillSwitch(
        cfg.LOG_DIR / "killswitch.json",
        daily_limit=cfg.DAILY_DRAWDOWN_LIMIT,
        total_limit=cfg.TOTAL_DRAWDOWN_LIMIT,
        max_consecutive_losses=cfg.MAX_CONSECUTIVE_LOSSES,
    )
    ks.load()
    if cfg.AUTO_RESET_KILLSWITCH_ON_START and (
        ks.state.get("tripped_day") or ks.state.get("tripped_total")
    ):
        today0 = dt.datetime.now().strftime("%Y-%m-%d")
        ks.hard_reset(today=today0, equity=equity)
        log_run("KillSwitch state auto-reset on startup (paper mode).")

    exe = ExecutionSimulator(cfg)
    event_filter = EventRiskFilter(cfg)
    meta_model = MetaLabelModel(cfg)
    monitor = RuntimeMonitor(cfg)
    telemetry_sink = (
        DecisionTelemetry(cfg, filename=str(getattr(cfg, "TELEMETRY_DECISIONS_FILE", "trade_decisions.jsonl")))
        if bool(getattr(cfg, "TELEMETRY_ENABLED", True))
        else None
    )
    last_telemetry_summary_ts = 0.0
    ib_fail_streak = 0
    last_ext_runtime_sig = None
    last_policy_sig = None
    last_policy_loss_hit = False
    last_overlay_gate_sig = None
    last_exec_governor_sig = None
    last_memory_feedback_sig = None
    last_aion_feedback_sig = None
    last_memory_replay_ts = 0.0
    last_memory_replay_ts_utc = None
    last_memory_replay_result = {
        "ok": True,
        "replayed": 0,
        "failed": 0,
        "processed_files": 0,
        "queued_files": 0,
        "remaining_files": 0,
        "error": None,
    }
    kill_switch = KillSwitchWatcher(state_dir=Path(cfg.STATE_DIR), poll_seconds=5.0)
    order_state_save_interval_sec = 300.0
    last_order_state_save_ts = 0.0
    canary_start_time = dt.datetime.now(dt.timezone.utc)
    canary_timeout_checked = False
    cached_ext_bundle = None
    last_overlay_poll_ts = 0.0

    if cfg.META_LABEL_ENABLED:
        samples = meta_model.fit_from_trades(cfg.LOG_DIR / "shadow_trades.csv")
        if samples > 0:
            log_run(f"Meta-label model trained on {samples} samples.")

    log_run(
        "Aion brain paper loop started (god-mode) "
        f"[trading_mode={cfg.TRADING_MODE} bar={cfg.HIST_BAR_SIZE} loop={cfg.LOOP_SECONDS}s "
        f"max_trades={cfg.MAX_TRADES_PER_DAY} max_hold={cfg.MAX_HOLD_CYCLES}]"
    )
    try:
        ib_client_boot = ib()
        attach_order_status_handler(
            ib_client_boot,
            state_dir=Path(cfg.STATE_DIR),
            shadow_path=Path(cfg.STATE_DIR) / "shadow_trades.json",
        )
        if bool(getattr(cfg, "AION_BLOCK_LIVE_ORDERS", True)) and (not bool(getattr(cfg, "AION_PAPER_MODE", True))):
            def _blocked_place_order(*_args, **_kwargs):
                log_run("PAPER-ONLY GUARD: placeOrder blocked")
                audit_log(
                    {"event": "LIVE_ORDER_BLOCKED", "reason": "AION_BLOCK_LIVE_ORDERS active"},
                    log_dir=Path(cfg.STATE_DIR),
                )
                return None

            try:
                setattr(ib_client_boot, "placeOrder", _blocked_place_order)
                log_run("PAPER-ONLY GUARD enabled: IB placeOrder patched")
            except Exception:
                pass

        rec = reconcile_on_startup(
            ib_client=ib_client_boot,
            shadow_path=Path(cfg.STATE_DIR) / "shadow_trades.json",
            auto_fix=bool(getattr(cfg, "RECONCILE_AUTO_FIX", True)),
            max_auto_fix_value=float(getattr(cfg, "RECONCILE_MAX_AUTO_FIX_VALUE", 5000.0)),
        )
        _write_json_atomic(
            Path(cfg.STATE_DIR) / "reconciliation_result.json",
            {
                "passed": bool(rec.passed),
                "mismatches": list(rec.mismatches),
                "ibkr_positions": dict(rec.ibkr_positions),
                "shadow_positions": dict(rec.shadow_positions),
                "action_taken": str(rec.action_taken),
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        if rec.mismatches:
            for mm in rec.mismatches:
                audit_log(
                    {"event": "RECONCILIATION_MISMATCH", **mm, "action_taken": str(rec.action_taken)},
                    log_dir=Path(cfg.STATE_DIR),
                )
            log_run(f"RECONCILIATION: {len(rec.mismatches)} mismatch(es), action={rec.action_taken}")
            send_alert(
                f"AION reconciliation mismatch count={len(rec.mismatches)} action={rec.action_taken}",
                level="WARNING",
            )
        if (not rec.passed) and str(rec.action_taken) == "manual_review_required":
            log_run("Startup halted: reconciliation requires manual review.")
            send_alert("AION startup blocked: reconciliation requires manual review.", level="CRITICAL")
            return 2

        _persist_ib_order_state(ib_client_boot)
        last_order_state_save_ts = float(time.monotonic())
    except Exception as exc:
        log_run(f"Startup safety bootstrap warning: {exc}")
        audit_log(
            {"event": "STARTUP_BOOTSTRAP_WARNING", "error": str(exc)},
            log_dir=Path(cfg.STATE_DIR),
        )

    try:
        while True:
            try:
                ib_client = ib()
                attach_order_status_handler(
                    ib_client,
                    state_dir=Path(cfg.STATE_DIR),
                    shadow_path=Path(cfg.STATE_DIR) / "shadow_trades.json",
                )
                if ib_fail_streak > 0:
                    log_run(f"IB reconnected after {ib_fail_streak} failed cycle(s).")
                    if cfg.MONITORING_ENABLED:
                        monitor.record_system_event("ib_reconnected", f"recovered_after={ib_fail_streak}")
                ib_fail_streak = 0
                now_mono = float(time.monotonic())
                if (now_mono - float(last_order_state_save_ts)) >= order_state_save_interval_sec:
                    _persist_ib_order_state(ib_client)
                    last_order_state_save_ts = now_mono
            except Exception as exc:
                ib_fail_streak += 1
                if ib_fail_streak == 1 or (ib_fail_streak % max(1, cfg.IB_RECONNECT_LOG_EVERY) == 0):
                    log_run(
                        f"IB unavailable ({cfg.IB_HOST}:{cfg.IB_PORT}) cycle={ib_fail_streak}: {exc}"
                    )
                if cfg.MONITORING_ENABLED:
                    monitor.record_system_event("ib_connect_fail", str(exc))
                    for msg in monitor.check_alerts():
                        log_alert(msg)
                        log_run(f"ALERT: {msg}")
                save_runtime_controls_heartbeat(
                    day_key=day_key,
                    trades_today=trades_today,
                    open_positions=open_positions,
                    watchlist_size=0,
                    status="ib_unavailable",
                )
                save_runtime_state(day_key, cash, closed_pnl, trades_today, open_positions, cooldown)
                time.sleep(cfg.LOOP_SECONDS)
                continue

            if bool(getattr(cfg, "MEMORY_REPLAY_ENABLED", True)) and bool(getattr(cfg, "MEMORY_ENABLE", False)):
                replay_interval = max(5.0, float(getattr(cfg, "MEMORY_REPLAY_INTERVAL_SEC", 300.0)))
                now_mono = time.monotonic()
                if (now_mono - float(last_memory_replay_ts)) >= replay_interval:
                    replay_res = replay_trade_outbox(
                        cfg,
                        max_files=max(1, int(getattr(cfg, "MEMORY_REPLAY_MAX_FILES", 4))),
                        max_events=max(1, int(getattr(cfg, "MEMORY_REPLAY_MAX_EVENTS", 200))),
                    )
                    last_memory_replay_ts = now_mono
                    last_memory_replay_ts_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    last_memory_replay_result = dict(replay_res if isinstance(replay_res, dict) else {})
                    replayed = int(replay_res.get("replayed", 0))
                    failed = int(replay_res.get("failed", 0))
                    if replayed > 0 or failed > 0 or (not bool(replay_res.get("ok", True))):
                        msg = (
                            "NovaSpine outbox replay "
                            f"replayed={replayed} failed={failed} "
                            f"processed_files={int(replay_res.get('processed_files', 0))} "
                            f"remaining_files={int(replay_res.get('remaining_files', 0))}"
                        )
                        if replay_res.get("error"):
                            msg += f" error={replay_res.get('error')}"
                        log_run(msg)
                        if cfg.MONITORING_ENABLED and (failed > 0 or not bool(replay_res.get("ok", True))):
                            monitor.record_system_event("novaspine_replay_fail", msg)

            profile = load_profile()
            if profile.get("trading_enabled") is False:
                log_run("Adaptive profile disabled trading; sleeping.")
                save_runtime_controls_heartbeat(
                    day_key=day_key,
                    trades_today=trades_today,
                    open_positions=open_positions,
                    watchlist_size=0,
                    status="adaptive_disabled",
                )
                save_runtime_state(day_key, cash, closed_pnl, trades_today, open_positions, cooldown)
                time.sleep(cfg.LOOP_SECONDS)
                continue

            max_trades_cap = int(profile.get("max_trades_per_day", cfg.MAX_TRADES_PER_DAY))
            max_open_positions_cap = int(profile.get("max_open_positions", cfg.MAX_OPEN_POSITIONS))
            max_trades_cap_runtime = max_trades_cap
            max_open_positions_runtime = max_open_positions_cap
            risk_per_trade_runtime = float(cfg.RISK_PER_TRADE)
            max_position_notional_pct_runtime = float(cfg.MAX_POSITION_NOTIONAL_PCT)
            max_gross_leverage_runtime = float(cfg.MAX_GROSS_LEVERAGE)
            policy_caps = {
                "block_new_entries": False,
                "blocked_symbols": set(),
                "allowed_symbols": set(),
                "daily_loss_limit_abs": None,
                "daily_loss_limit_pct": None,
            }

            wl = load_watchlist()
            if not wl:
                log_run("No watchlist found. Run universe_scan first. Sleeping.")
                save_runtime_controls_heartbeat(
                    day_key=day_key,
                    trades_today=trades_today,
                    open_positions=open_positions,
                    watchlist_size=0,
                    status="watchlist_missing",
                )
                save_runtime_state(day_key, cash, closed_pnl, trades_today, open_positions, cooldown)
                time.sleep(cfg.LOOP_SECONDS)
                continue

            today = dt.datetime.now().strftime("%Y-%m-%d")
            if today != day_key:
                day_key = today
                trades_today = 0
                day_start_equity = float(equity)
                if cfg.META_LABEL_ENABLED:
                    samples = meta_model.fit_from_trades(cfg.LOG_DIR / "shadow_trades.csv")
                    if samples > 0:
                        log_run(f"Meta-label model refreshed on day change ({samples} samples).")

            canary_block_new_entries = False
            if not canary_timeout_checked:
                canary_age_minutes = (dt.datetime.now(dt.timezone.utc) - canary_start_time).total_seconds() / 60.0
                if canary_age_minutes > float(getattr(cfg, "CANARY_MAX_LIFETIME_MINUTES", 45)):
                    action = str(getattr(cfg, "CANARY_TIMEOUT_ACTION", "pass")).strip().lower() or "pass"
                    result = "TIMEOUT_PASSED" if action == "pass" else "TIMEOUT_FAILED"
                    audit_log(
                        {
                            "event": "CANARY_TIMEOUT",
                            "age_minutes": float(canary_age_minutes),
                            "result": str(result),
                        },
                        log_dir=Path(cfg.STATE_DIR),
                    )
                    log_run(
                        "CANARY TIMEOUT: "
                        f"{canary_age_minutes:.0f} min > {int(getattr(cfg, 'CANARY_MAX_LIFETIME_MINUTES', 45))} min limit"
                    )
                    send_alert(
                        f"Canary timeout after {canary_age_minutes:.0f} min result={result}",
                        level=("WARNING" if result == "TIMEOUT_PASSED" else "CRITICAL"),
                    )
                    canary_block_new_entries = (result == "TIMEOUT_FAILED")
                    canary_timeout_checked = True

            killswitch_block_new_entries = not ks.check(today, equity)
            if canary_block_new_entries:
                killswitch_block_new_entries = True
                ks.last_reason = "canary_timeout_failed"
            if killswitch_block_new_entries:
                log_run(f"KillSwitch tripped: {ks.last_reason}")

            if kill_switch.check():
                log_run("KILL SWITCH: Flattening all positions and shutting down")
                audit_log(
                    {"event": "KILL_SWITCH_TRIGGERED", "open_positions": int(len(open_positions))},
                    log_dir=Path(cfg.STATE_DIR),
                )
                send_alert("AION KILL SWITCH triggered - flattening all positions", level="CRITICAL")
                for sym in list(open_positions.keys()):
                    pos = open_positions.get(sym)
                    if not isinstance(pos, dict):
                        continue
                    qty = int(max(0, _safe_int(pos.get("qty"), 0)))
                    if qty <= 0:
                        open_positions.pop(sym, None)
                        continue
                    px = float(_safe_float(pos.get("mark_price", pos.get("entry", 0.0)), 0.0))
                    if px <= 0:
                        px = float(_safe_float(pos.get("entry", 0.0), 0.0))
                    side = "SELL" if str(pos.get("side", "")).upper() == "LONG" else "BUY"
                    fill = _execute_fill(
                        exe,
                        side=side,
                        qty=qty,
                        ref_price=px,
                        atr_pct=float(_safe_float(pos.get("atr_pct"), 0.0)),
                        confidence=1.0,
                        allow_partial=False,
                        symbol=str(sym),
                        ib_client=ib_client,
                    )
                    fq = int(max(0, min(qty, _safe_int(getattr(fill, "filled_qty", 0), 0))))
                    if fq <= 0:
                        continue
                    monitor.record_execution(float(_safe_float(getattr(fill, "est_slippage_bps", 0.0), 0.0)))
                    cash, closed_pnl = _close_position(
                        open_positions,
                        sym,
                        float(fill.avg_fill),
                        "kill_switch",
                        cash,
                        closed_pnl,
                        ks,
                        today,
                        fill_ratio=float(_safe_float(getattr(fill, "fill_ratio", 1.0), 1.0)),
                        slippage_bps=float(_safe_float(getattr(fill, "est_slippage_bps", 0.0), 0.0)),
                        monitor=monitor,
                        memory_runtime_context=None,
                        telemetry_sink=telemetry_sink,
                        governor_compound_scalar=None,
                        filled_qty=fq,
                        sync_shadow=(str(getattr(fill, "source", "simulator")).strip().lower() != "ib_paper"),
                    )
                    fully_closed = sym not in open_positions
                    if fully_closed:
                        cooldown[sym] = cfg.REENTRY_COOLDOWN_CYCLES
                kill_switch.acknowledge()
                save_runtime_state(day_key, cash, closed_pnl, trades_today, open_positions, cooldown)
                try:
                    write_system_health(state_dir=Path(cfg.STATE_DIR), log_dir=Path(cfg.LOG_DIR))
                except Exception:
                    pass
                try:
                    _persist_ib_order_state(ib_client)
                except Exception:
                    pass
                return 0

            for sym in list(cooldown.keys()):
                cooldown[sym] = max(0, cooldown[sym] - 1)
                if cooldown[sym] == 0:
                    cooldown.pop(sym, None)

            last_prices = {}
            cycle_conf = []
            entry_candidates = []
            external_signals = {}
            global_external = None
            ext_runtime_scale = 1.0
            ext_position_risk_scale = 1.0
            ext_overlay_age_hours = None
            ext_overlay_age_source = "unknown"
            ext_overlay_generated_at_utc = None
            ext_runtime_diag = {"active": False, "flags": [], "degraded": False, "quality_gate_ok": True}
            overlay_block_new_entries = False
            overlay_block_reasons = []
            exec_governor_block_new_entries = False
            exec_governor_state = "off"
            exec_governor_reasons = []
            exec_governor_recent_executions = 0
            exec_governor_exec_rate_per_min = 0.0
            exec_governor_avg_slippage_bps = None
            exec_governor_p90_slippage_bps = None
            memory_feedback_active = False
            memory_feedback_status = "unknown"
            memory_feedback_reasons = []
            memory_feedback_risk_scale = 1.0
            memory_feedback_trades_scale = 1.0
            memory_feedback_open_scale = 1.0
            memory_feedback_turnover_pressure = None
            memory_feedback_turnover_dampener = None
            memory_feedback_block_new_entries = False
            aion_feedback_active = False
            aion_feedback_status = "unknown"
            aion_feedback_source = "unknown"
            aion_feedback_source_selected = "unknown"
            aion_feedback_source_preference = "auto"
            aion_feedback_reasons = []
            aion_feedback_risk_scale = 1.0
            aion_feedback_closed_trades = 0
            aion_feedback_hit_rate = None
            aion_feedback_profit_factor = None
            aion_feedback_expectancy = None
            aion_feedback_drawdown_norm = None
            aion_feedback_age_hours = None
            aion_feedback_max_age_hours = None
            aion_feedback_stale = False
            aion_feedback_last_closed_ts = None
            aion_feedback_path = ""
            aion_feedback_block_new_entries = False
            if cfg.EXT_SIGNAL_ENABLED:
                now_mono_overlay = float(time.monotonic())
                overlay_poll_sec = max(5.0, float(getattr(cfg, "EXT_SIGNAL_POLL_SECONDS", 300)))
                if (cached_ext_bundle is None) or ((now_mono_overlay - float(last_overlay_poll_ts)) >= overlay_poll_sec):
                    ext_bundle = load_external_signal_bundle(
                        path=cfg.EXT_SIGNAL_FILE,
                        min_confidence=cfg.EXT_SIGNAL_MIN_CONFIDENCE,
                        max_bias=cfg.EXT_SIGNAL_MAX_BIAS,
                        max_age_hours=cfg.EXT_SIGNAL_MAX_AGE_HOURS,
                    )
                    cached_ext_bundle = ext_bundle if isinstance(ext_bundle, dict) else {}
                    last_overlay_poll_ts = now_mono_overlay
                    log_run("Overlay refreshed")
                ext_bundle = cached_ext_bundle if isinstance(cached_ext_bundle, dict) else {}
                if isinstance(ext_bundle, dict) and bool(ext_bundle.get("overlay_rejected", False)):
                    reason = str(ext_bundle.get("overlay_rejection_reason", "invalid_overlay"))
                    log_run(f"OVERLAY REJECTED: {reason}")
                    audit_log(
                        {"event": "OVERLAY_REJECTED", "reason": reason, "path": str(cfg.EXT_SIGNAL_FILE)},
                        log_dir=Path(cfg.STATE_DIR),
                    )
                ext_overlay_age_hours = (
                    _safe_float(ext_bundle.get("overlay_age_hours", None), None)
                    if isinstance(ext_bundle, dict)
                    else None
                )
                ext_overlay_age_source = (
                    str(ext_bundle.get("overlay_age_source", "unknown"))
                    if isinstance(ext_bundle, dict)
                    else "unknown"
                )
                ext_overlay_generated_at_utc = (
                    str(ext_bundle.get("overlay_generated_at_utc"))
                    if isinstance(ext_bundle, dict) and ext_bundle.get("overlay_generated_at_utc")
                    else None
                )
                external_signals = ext_bundle.get("signals", {}) if isinstance(ext_bundle, dict) else {}
                global_external = external_signals.get("__GLOBAL__")
                ext_runtime_scale, ext_runtime_diag = runtime_overlay_scale(
                    ext_bundle,
                    min_scale=cfg.EXT_SIGNAL_RUNTIME_MIN_SCALE,
                    max_scale=cfg.EXT_SIGNAL_RUNTIME_MAX_SCALE,
                    degraded_scale=cfg.EXT_SIGNAL_RUNTIME_DEGRADED_SCALE,
                    quality_fail_scale=cfg.EXT_SIGNAL_RUNTIME_QFAIL_SCALE,
                    flag_scale=cfg.EXT_SIGNAL_RUNTIME_FLAG_SCALE,
                    drift_warn_scale=cfg.EXT_SIGNAL_RUNTIME_DRIFT_WARN_SCALE,
                    drift_alert_scale=cfg.EXT_SIGNAL_RUNTIME_DRIFT_ALERT_SCALE,
                    quality_step_spike_scale=cfg.EXT_SIGNAL_RUNTIME_QUALITY_STEP_SPIKE_SCALE,
                    fracture_warn_scale=cfg.EXT_SIGNAL_RUNTIME_FRACTURE_WARN_SCALE,
                    fracture_alert_scale=cfg.EXT_SIGNAL_RUNTIME_FRACTURE_ALERT_SCALE,
                    exec_risk_tight_scale=cfg.EXT_SIGNAL_RUNTIME_EXEC_RISK_TIGHT_SCALE,
                    exec_risk_hard_scale=cfg.EXT_SIGNAL_RUNTIME_EXEC_RISK_HARD_SCALE,
                    nested_leak_warn_scale=cfg.EXT_SIGNAL_RUNTIME_NESTED_LEAK_WARN_SCALE,
                    nested_leak_alert_scale=cfg.EXT_SIGNAL_RUNTIME_NESTED_LEAK_ALERT_SCALE,
                    hive_stress_warn_scale=cfg.EXT_SIGNAL_RUNTIME_HIVE_WARN_SCALE,
                    hive_stress_alert_scale=cfg.EXT_SIGNAL_RUNTIME_HIVE_ALERT_SCALE,
                    heartbeat_warn_scale=cfg.EXT_SIGNAL_RUNTIME_HEARTBEAT_WARN_SCALE,
                    heartbeat_alert_scale=cfg.EXT_SIGNAL_RUNTIME_HEARTBEAT_ALERT_SCALE,
                    council_divergence_warn_scale=cfg.EXT_SIGNAL_RUNTIME_COUNCIL_WARN_SCALE,
                    council_divergence_alert_scale=cfg.EXT_SIGNAL_RUNTIME_COUNCIL_ALERT_SCALE,
                    aion_outcome_warn_scale=cfg.EXT_SIGNAL_RUNTIME_AION_OUTCOME_WARN_SCALE,
                    aion_outcome_alert_scale=cfg.EXT_SIGNAL_RUNTIME_AION_OUTCOME_ALERT_SCALE,
                    aion_outcome_stale_scale=cfg.EXT_SIGNAL_RUNTIME_AION_OUTCOME_STALE_SCALE,
                    overlay_stale_scale=cfg.EXT_SIGNAL_RUNTIME_STALE_SCALE,
                )
                max_trades_cap_runtime, max_open_positions_runtime = _runtime_risk_caps(
                    max_trades_cap=max_trades_cap,
                    max_open_positions_cap=max_open_positions_cap,
                    ext_runtime_scale=ext_runtime_scale,
                    ext_runtime_diag=ext_runtime_diag,
                )
                ext_position_risk_scale = _runtime_position_risk_scale(
                    ext_runtime_scale=ext_runtime_scale,
                    ext_runtime_diag=ext_runtime_diag,
                )
                overlay_block_new_entries, overlay_block_reasons = _overlay_entry_gate(
                    ext_runtime_diag=ext_runtime_diag,
                    overlay_age_hours=ext_overlay_age_hours,
                )
                risk_per_trade_runtime = max(1e-4, float(risk_per_trade_runtime) * ext_position_risk_scale)
                max_position_notional_pct_runtime = max(
                    0.01, float(max_position_notional_pct_runtime) * ext_position_risk_scale
                )
                max_gross_leverage_runtime = max(
                    0.30,
                    float(max_gross_leverage_runtime) * max(0.55, ext_position_risk_scale),
                )
                mem_feedback = ext_bundle.get("memory_feedback", {}) if isinstance(ext_bundle, dict) else {}
                mem_ctl = _memory_feedback_controls(
                    max_trades_cap_runtime=max_trades_cap_runtime,
                    max_open_positions_runtime=max_open_positions_runtime,
                    risk_per_trade_runtime=risk_per_trade_runtime,
                    max_position_notional_pct_runtime=max_position_notional_pct_runtime,
                    max_gross_leverage_runtime=max_gross_leverage_runtime,
                    memory_feedback=mem_feedback if isinstance(mem_feedback, dict) else {},
                )
                max_trades_cap_runtime = int(mem_ctl["max_trades_cap_runtime"])
                max_open_positions_runtime = int(mem_ctl["max_open_positions_runtime"])
                risk_per_trade_runtime = float(mem_ctl["risk_per_trade_runtime"])
                max_position_notional_pct_runtime = float(mem_ctl["max_position_notional_pct_runtime"])
                max_gross_leverage_runtime = float(mem_ctl["max_gross_leverage_runtime"])
                memory_feedback_active = bool(mem_ctl.get("active", False))
                memory_feedback_status = str(mem_ctl.get("status", "unknown"))
                memory_feedback_reasons = [str(x) for x in mem_ctl.get("reasons", []) if str(x)]
                memory_feedback_risk_scale = _safe_float(mem_ctl.get("risk_scale"), 1.0)
                memory_feedback_trades_scale = _safe_float(mem_ctl.get("trades_scale"), 1.0)
                memory_feedback_open_scale = _safe_float(mem_ctl.get("open_scale"), 1.0)
                memory_feedback_turnover_pressure = _safe_float(mem_ctl.get("turnover_pressure"), None)
                memory_feedback_turnover_dampener = _safe_float(mem_ctl.get("turnover_dampener"), None)
                memory_feedback_block_new_entries = bool(mem_ctl.get("block_new_entries", False))
                aion_fb = ext_bundle.get("aion_feedback", {}) if isinstance(ext_bundle, dict) else {}
                aion_ctl = _aion_feedback_controls(aion_fb if isinstance(aion_fb, dict) else {})
                aion_feedback_active = bool(aion_ctl.get("active", False))
                aion_feedback_status = str(aion_ctl.get("status", "unknown"))
                aion_feedback_source = str(aion_ctl.get("source", "unknown"))
                aion_feedback_source_selected = str(
                    aion_ctl.get("source_selected", aion_feedback_source)
                )
                aion_feedback_source_preference = str(aion_ctl.get("source_preference", "auto"))
                aion_feedback_reasons = [str(x) for x in aion_ctl.get("reasons", []) if str(x)]
                aion_feedback_risk_scale = _safe_float(aion_ctl.get("risk_scale"), 1.0)
                aion_feedback_closed_trades = max(0, _safe_int(aion_ctl.get("closed_trades"), 0))
                aion_feedback_hit_rate = _safe_float(aion_ctl.get("hit_rate"), None)
                aion_feedback_profit_factor = _safe_float(aion_ctl.get("profit_factor"), None)
                aion_feedback_expectancy = _safe_float(aion_ctl.get("expectancy"), None)
                aion_feedback_drawdown_norm = _safe_float(aion_ctl.get("drawdown_norm"), None)
                aion_feedback_age_hours = _safe_float(aion_ctl.get("age_hours"), None)
                aion_feedback_max_age_hours = _safe_float(aion_ctl.get("max_age_hours"), None)
                aion_feedback_stale = bool(aion_ctl.get("stale", False))
                aion_feedback_last_closed_ts = (
                    str(aion_ctl.get("last_closed_ts", "")).strip() or None
                )
                aion_feedback_path = str(aion_ctl.get("path", "")).strip()
                aion_feedback_block_new_entries = bool(aion_ctl.get("block_new_entries", False))
                sig = (
                    round(float(ext_runtime_scale), 4),
                    round(float(ext_position_risk_scale), 4),
                    tuple(sorted(str(x) for x in ext_runtime_diag.get("flags", []) if str(x))),
                    bool(ext_runtime_diag.get("degraded", False)),
                    bool(ext_runtime_diag.get("quality_gate_ok", True)),
                    bool(ext_runtime_diag.get("overlay_stale", False)),
                    None if ext_overlay_age_hours is None else round(float(ext_overlay_age_hours), 3),
                    str(ext_overlay_age_source),
                    ext_overlay_generated_at_utc if ext_overlay_generated_at_utc is not None else "na",
                    str(ext_runtime_diag.get("regime", "unknown")),
                    str(ext_runtime_diag.get("source_mode", "unknown")),
                    int(max_trades_cap_runtime),
                    int(max_open_positions_runtime),
                    round(float(risk_per_trade_runtime), 6),
                    round(float(max_position_notional_pct_runtime), 6),
                    round(float(max_gross_leverage_runtime), 6),
                )
                if sig != last_ext_runtime_sig:
                    flag_txt = ",".join(sig[2]) if sig[2] else "none"
                    log_run(
                        "External runtime overlay "
                        f"scale={sig[0]:.3f} pos_risk_scale={sig[1]:.3f} flags={flag_txt} "
                        f"degraded={sig[3]} quality_ok={sig[4]} overlay_stale={sig[5]} "
                        f"overlay_age_h={(f'{sig[6]:.2f}' if isinstance(sig[6], float) else 'na')} "
                        f"overlay_age_src={sig[7]} overlay_generated_at={sig[8]} "
                        f"regime={sig[9]} source={sig[10]} "
                        f"max_trades={sig[11]}/{max_trades_cap} max_open={sig[12]}/{max_open_positions_cap} "
                        f"risk_per_trade={sig[13]:.4f} max_notional_pct={sig[14]:.4f} max_gross_lev={sig[15]:.3f}"
                    )
                    if cfg.MONITORING_ENABLED and (sig[3] or (not sig[4]) or sig[5] or bool(sig[2])):
                        monitor.record_system_event(
                            "external_overlay_runtime",
                            f"scale={sig[0]:.3f} pos_risk_scale={sig[1]:.3f} flags={flag_txt} source={sig[10]} "
                            f"overlay_stale={sig[5]} overlay_age_h={(f'{sig[6]:.2f}' if isinstance(sig[6], float) else 'na')} "
                            f"overlay_generated_at={sig[8]} max_trades={sig[11]}/{max_trades_cap} "
                            f"max_open={sig[12]}/{max_open_positions_cap} risk_per_trade={sig[13]:.4f}",
                        )
                    last_ext_runtime_sig = sig
                mem_sig = (
                    bool(memory_feedback_active),
                    str(memory_feedback_status),
                    tuple(sorted(memory_feedback_reasons)),
                    bool(memory_feedback_block_new_entries),
                    round(float(memory_feedback_risk_scale), 4),
                    round(float(memory_feedback_trades_scale), 4),
                    round(float(memory_feedback_open_scale), 4),
                    None
                    if memory_feedback_turnover_pressure is None
                    else round(float(memory_feedback_turnover_pressure), 4),
                    None
                    if memory_feedback_turnover_dampener is None
                    else round(float(memory_feedback_turnover_dampener), 4),
                )
                if mem_sig != last_memory_feedback_sig and mem_sig[0]:
                    reason_txt = ",".join(mem_sig[2]) if mem_sig[2] else "none"
                    log_run(
                        "NovaSpine memory feedback "
                        f"status={mem_sig[1]} reasons={reason_txt} block_new={mem_sig[3]} "
                        f"risk_scale={mem_sig[4]:.3f} trades_scale={mem_sig[5]:.3f} open_scale={mem_sig[6]:.3f} "
                        f"turnover_pressure={(f'{mem_sig[7]:.3f}' if isinstance(mem_sig[7], float) else 'na')} "
                        f"turnover_dampener={(f'{mem_sig[8]:.3f}' if isinstance(mem_sig[8], float) else 'na')}"
                    )
                    if cfg.MONITORING_ENABLED and (mem_sig[1] in {"warn", "alert"} or mem_sig[3]):
                        monitor.record_system_event(
                            "novaspine_memory_feedback",
                            f"status={mem_sig[1]} reasons={reason_txt} block_new={mem_sig[3]} risk_scale={mem_sig[4]:.3f} "
                            f"turnover_pressure={(f'{mem_sig[7]:.3f}' if isinstance(mem_sig[7], float) else 'na')} "
                            f"turnover_dampener={(f'{mem_sig[8]:.3f}' if isinstance(mem_sig[8], float) else 'na')}",
                        )
                last_memory_feedback_sig = mem_sig
                aion_sig = (
                    bool(aion_feedback_active),
                    str(aion_feedback_status).strip().lower(),
                    str(aion_feedback_source).strip().lower(),
                    str(aion_feedback_source_selected).strip().lower(),
                    str(aion_feedback_source_preference).strip().lower(),
                    tuple(sorted(str(x).strip().lower() for x in aion_feedback_reasons if str(x).strip())),
                    round(float(aion_feedback_risk_scale), 4),
                    int(aion_feedback_closed_trades),
                    None if aion_feedback_hit_rate is None else round(float(aion_feedback_hit_rate), 4),
                    None if aion_feedback_profit_factor is None else round(float(aion_feedback_profit_factor), 4),
                    None if aion_feedback_expectancy is None else round(float(aion_feedback_expectancy), 4),
                    None if aion_feedback_drawdown_norm is None else round(float(aion_feedback_drawdown_norm), 4),
                    None if aion_feedback_age_hours is None else round(float(aion_feedback_age_hours), 3),
                    None if aion_feedback_max_age_hours is None else round(float(aion_feedback_max_age_hours), 3),
                    bool(aion_feedback_stale),
                    aion_feedback_last_closed_ts if aion_feedback_last_closed_ts is not None else "na",
                    bool(aion_feedback_block_new_entries),
                    str(aion_feedback_path),
                )
                if aion_sig != last_aion_feedback_sig and aion_sig[0]:
                    reason_txt = ",".join(aion_sig[5]) if aion_sig[5] else "none"
                    log_run(
                        "AION outcome feedback "
                        f"status={aion_sig[1]} src={aion_sig[2]} selected={aion_sig[3]} pref={aion_sig[4]} "
                        f"reasons={reason_txt} block_new={aion_sig[16]} risk_scale={aion_sig[6]:.3f} "
                        f"closed_trades={aion_sig[7]} hit_rate={(f'{aion_sig[8]:.3f}' if isinstance(aion_sig[8], float) else 'na')} "
                        f"profit_factor={(f'{aion_sig[9]:.3f}' if isinstance(aion_sig[9], float) else 'na')} "
                        f"drawdown_norm={(f'{aion_sig[11]:.3f}' if isinstance(aion_sig[11], float) else 'na')} "
                        f"age_h={(f'{aion_sig[12]:.2f}' if isinstance(aion_sig[12], float) else 'na')} "
                        f"stale={aion_sig[14]} last_closed_ts={aion_sig[15]}"
                    )
                    if cfg.MONITORING_ENABLED and (aion_sig[1] in {"warn", "alert"} or aion_sig[14]):
                        monitor.record_system_event(
                            "aion_outcome_feedback",
                            f"status={aion_sig[1]} src={aion_sig[2]} selected={aion_sig[3]} pref={aion_sig[4]} reasons={reason_txt} "
                            f"block_new={aion_sig[16]} risk_scale={aion_sig[6]:.3f} closed_trades={aion_sig[7]} "
                            f"stale={aion_sig[14]} last_closed_ts={aion_sig[15]}",
                        )
                last_aion_feedback_sig = aion_sig
                gate_sig = (bool(overlay_block_new_entries), tuple(sorted(str(x) for x in overlay_block_reasons)))
                if gate_sig != last_overlay_gate_sig:
                    reason_txt = ",".join(gate_sig[1]) if gate_sig[1] else "none"
                    log_run(
                        "External overlay entry gate "
                        f"blocked={gate_sig[0]} reasons={reason_txt} "
                        f"age_h={(f'{ext_overlay_age_hours:.2f}' if isinstance(ext_overlay_age_hours, float) else 'na')} "
                        f"age_src={ext_overlay_age_source}"
                    )
                    if cfg.MONITORING_ENABLED and gate_sig[0]:
                        monitor.record_system_event(
                            "external_overlay_entry_block",
                            f"reasons={reason_txt} age_h={(f'{ext_overlay_age_hours:.2f}' if isinstance(ext_overlay_age_hours, float) else 'na')}",
                        )
                    last_overlay_gate_sig = gate_sig

            if cfg.RISK_POLICY_ENFORCE:
                loaded_policy = load_policy(cfg.RISK_POLICY_FILE)
                policy_caps = apply_policy_caps(
                    loaded_policy,
                    max_trades_per_day=max_trades_cap_runtime,
                    max_open_positions=max_open_positions_runtime,
                    risk_per_trade=risk_per_trade_runtime,
                    max_position_notional_pct=max_position_notional_pct_runtime,
                    max_gross_leverage=max_gross_leverage_runtime,
                )
                max_trades_cap_runtime = int(policy_caps["max_trades_per_day"])
                max_open_positions_runtime = int(policy_caps["max_open_positions"])
                risk_per_trade_runtime = float(policy_caps["risk_per_trade"])
                max_position_notional_pct_runtime = float(policy_caps["max_position_notional_pct"])
                max_gross_leverage_runtime = float(policy_caps["max_gross_leverage"])

                policy_sig = (
                    bool(policy_caps.get("block_new_entries", False)),
                    int(max_trades_cap_runtime),
                    int(max_open_positions_runtime),
                    round(float(risk_per_trade_runtime), 6),
                    round(float(max_position_notional_pct_runtime), 6),
                    round(float(max_gross_leverage_runtime), 6),
                    len(policy_caps.get("blocked_symbols", set()) or set()),
                    len(policy_caps.get("allowed_symbols", set()) or set()),
                    policy_caps.get("daily_loss_limit_abs"),
                    policy_caps.get("daily_loss_limit_pct"),
                    bool(overlay_block_new_entries),
                    tuple(sorted(str(x) for x in overlay_block_reasons if str(x))),
                    bool(memory_feedback_block_new_entries),
                    tuple(sorted(str(x) for x in memory_feedback_reasons if str(x))),
                )
                if policy_sig != last_policy_sig:
                    reason_txt = ",".join(policy_sig[11]) if policy_sig[11] else "none"
                    mem_reason_txt = ",".join(policy_sig[13]) if policy_sig[13] else "none"
                    log_run(
                        "Risk policy active "
                        f"block_new={policy_sig[0]} max_trades={policy_sig[1]} max_open={policy_sig[2]} "
                        f"risk_per_trade={policy_sig[3]:.4f} max_notional_pct={policy_sig[4]:.4f} "
                        f"max_gross_lev={policy_sig[5]:.3f} blocked={policy_sig[6]} allowed={policy_sig[7]} "
                        f"daily_loss_abs={policy_sig[8]} daily_loss_pct={policy_sig[9]} "
                        f"overlay_block={policy_sig[10]} overlay_reasons={reason_txt} "
                        f"memory_block={policy_sig[12]} memory_reasons={mem_reason_txt}"
                    )
                    last_policy_sig = policy_sig

            exec_gov = _execution_quality_governor(
                max_trades_per_day=max_trades_cap_runtime,
                max_open_positions=max_open_positions_runtime,
                risk_per_trade=risk_per_trade_runtime,
                max_position_notional_pct=max_position_notional_pct_runtime,
                max_gross_leverage=max_gross_leverage_runtime,
                monitor=monitor,
            )
            max_trades_cap_runtime = int(exec_gov["max_trades_per_day"])
            max_open_positions_runtime = int(exec_gov["max_open_positions"])
            risk_per_trade_runtime = float(exec_gov["risk_per_trade"])
            max_position_notional_pct_runtime = float(exec_gov["max_position_notional_pct"])
            max_gross_leverage_runtime = float(exec_gov["max_gross_leverage"])
            exec_governor_block_new_entries = bool(exec_gov.get("block_new_entries", False))
            exec_governor_state = str(exec_gov.get("state", "off"))
            exec_governor_reasons = [str(x) for x in exec_gov.get("reasons", []) if str(x)]
            exec_governor_recent_executions = int(exec_gov.get("recent_executions", 0))
            exec_governor_exec_rate_per_min = _safe_float(exec_gov.get("exec_rate_per_min"), 0.0)
            exec_governor_avg_slippage_bps = _safe_float(exec_gov.get("avg_slippage_bps"), None)
            exec_governor_p90_slippage_bps = _safe_float(exec_gov.get("p90_slippage_bps"), None)
            exec_sig = (
                exec_governor_state,
                tuple(sorted(exec_governor_reasons)),
                bool(exec_governor_block_new_entries),
                int(max_trades_cap_runtime),
                int(max_open_positions_runtime),
                round(float(risk_per_trade_runtime), 6),
                round(float(max_position_notional_pct_runtime), 6),
                round(float(max_gross_leverage_runtime), 6),
                int(exec_governor_recent_executions),
                round(float(exec_governor_exec_rate_per_min), 4),
                None if exec_governor_avg_slippage_bps is None else round(float(exec_governor_avg_slippage_bps), 4),
                None if exec_governor_p90_slippage_bps is None else round(float(exec_governor_p90_slippage_bps), 4),
            )
            if exec_sig != last_exec_governor_sig:
                reason_txt = ",".join(exec_sig[1]) if exec_sig[1] else "none"
                log_run(
                    "Execution quality governor "
                    f"state={exec_sig[0]} reasons={reason_txt} block_new={exec_sig[2]} "
                    f"recent_exec={exec_sig[8]} rate_per_min={exec_sig[9]:.3f} "
                    f"avg_slip_bps={(f'{exec_sig[10]:.2f}' if isinstance(exec_sig[10], float) else 'na')} "
                    f"p90_slip_bps={(f'{exec_sig[11]:.2f}' if isinstance(exec_sig[11], float) else 'na')} "
                    f"max_trades={exec_sig[3]} max_open={exec_sig[4]} risk_per_trade={exec_sig[5]:.4f} "
                    f"max_notional_pct={exec_sig[6]:.4f} max_gross_lev={exec_sig[7]:.3f}"
                )
                if cfg.MONITORING_ENABLED and exec_governor_state in {"warn", "alert"}:
                    monitor.record_system_event(
                        "execution_quality_governor",
                        f"state={exec_governor_state} reasons={reason_txt} block_new={exec_governor_block_new_entries} "
                        f"avg_slip_bps={(f'{exec_governor_avg_slippage_bps:.2f}' if isinstance(exec_governor_avg_slippage_bps, float) else 'na')} "
                        f"p90_slip_bps={(f'{exec_governor_p90_slippage_bps:.2f}' if isinstance(exec_governor_p90_slippage_bps, float) else 'na')}",
                    )
                last_exec_governor_sig = exec_sig

            policy_loss_hit, policy_daily_loss_abs, policy_daily_loss_pct = _daily_loss_limits_hit(
                policy_caps, day_start_equity, equity
            )
            policy_block_new_entries = bool(
                policy_caps.get("block_new_entries", False)
                or policy_loss_hit
                or overlay_block_new_entries
                or memory_feedback_block_new_entries
                or aion_feedback_block_new_entries
                or exec_governor_block_new_entries
            )
            if policy_loss_hit and (not last_policy_loss_hit):
                log_run(
                    "Risk policy halted new entries "
                    f"(daily_loss_abs={policy_daily_loss_abs:.2f}, daily_loss_pct={policy_daily_loss_pct:.2%})"
                )
            last_policy_loss_hit = bool(policy_loss_hit)

            # Governor hierarchy: escalate critical vetoes into FLATTEN workflow.
            gov_results = []
            if policy_loss_hit:
                gov_results.append({"name": "daily_loss_limit", "score": 0.0, "threshold": 1.0})
            if overlay_block_new_entries:
                gov_results.append({"name": "shock_mask_guard", "score": 0.0, "threshold": 1.0})
            flag_to_governor = {
                "fracture_alert": "crisis_sentinel",
                "drift_alert": "shock_mask_guard",
                "exec_risk_hard": "exposure_gate",
            }
            for fl in ext_runtime_diag.get("flags", []) if isinstance(ext_runtime_diag.get("flags", []), list) else []:
                gname = flag_to_governor.get(str(fl).strip().lower())
                if gname:
                    gov_results.append({"name": gname, "score": 0.0, "threshold": 1.0})

            gov_action = resolve_governor_action(gov_results) if gov_results else GovernorAction.PASS
            try:
                _write_runtime_governor_diagnostics(gov_results, gov_action)
            except Exception as exc:
                log_run(f"governor diagnostics write failed: {exc}")
            if gov_action >= GovernorAction.FLATTEN:
                triggered = [str(x.get("name", "")) for x in gov_results if x.get("name")]
                log_run(f"GOVERNOR FLATTEN: Closing all positions | triggered={','.join(triggered) if triggered else 'unknown'}")
                audit_log(
                    {"event": "GOVERNOR_FLATTEN", "triggered": triggered},
                    log_dir=Path(cfg.STATE_DIR),
                )
                try:
                    (Path(cfg.STATE_DIR) / "KILL_SWITCH").write_text("GOVERNOR_FLATTEN", encoding="utf-8")
                except Exception:
                    pass

                for sym in list(open_positions.keys()):
                    pos = open_positions.get(sym)
                    if not pos:
                        continue
                    qty = int(pos.get("qty", 0))
                    if qty <= 0:
                        continue
                    side = "SELL" if str(pos.get("side", "BUY")).upper() == "BUY" else "BUY"
                    px = _safe_float(last_prices.get(sym), None)
                    if not isinstance(px, float):
                        bars = hist_bars_cached(sym, cfg.HIST_BAR_SIZE, cfg.HIST_DURATION, retries=cfg.IB_RETRY_COUNT)
                        if bars is None or bars.empty:
                            continue
                        px = float(bars["close"].iloc[-1])
                    fill = exe.simulate_fill(
                        side=side,
                        qty=qty,
                        ref_price=px,
                        atr_pct=float(_safe_float(pos.get("atr_pct"), 0.0)),
                        confidence=1.0,
                        allow_partial=False,
                    )
                    fq = int(max(0, min(qty, _safe_int(getattr(fill, "filled_qty", 0), 0))))
                    if fq <= 0:
                        continue
                    monitor.record_execution(float(_safe_float(getattr(fill, "est_slippage_bps", 0.0), 0.0)))
                    cash, closed_pnl = _close_position(
                        open_positions,
                        sym,
                        float(fill.avg_fill),
                        "governor_flatten",
                        cash,
                        closed_pnl,
                        ks,
                        today,
                        fill_ratio=float(_safe_float(getattr(fill, "fill_ratio", 1.0), 1.0)),
                        slippage_bps=float(_safe_float(getattr(fill, "est_slippage_bps", 0.0), 0.0)),
                        monitor=monitor,
                        memory_runtime_context=None,
                        telemetry_sink=telemetry_sink,
                        governor_compound_scalar=None,
                        filled_qty=fq,
                    )
                    if sym not in open_positions:
                        cooldown[sym] = cfg.REENTRY_COOLDOWN_CYCLES

                save_runtime_state(day_key, cash, closed_pnl, trades_today, open_positions, cooldown)
                try:
                    write_system_health(state_dir=Path(cfg.STATE_DIR), log_dir=Path(cfg.LOG_DIR))
                except Exception:
                    pass
                try:
                    _persist_ib_order_state(ib_client)
                except Exception:
                    pass
                send_alert(
                    f"AION governor FLATTEN triggered: {','.join(triggered) if triggered else 'unknown'}",
                    level="CRITICAL",
                )
                return 0

            save_runtime_controls(
                {
                    "ts": dt.datetime.now().isoformat(),
                    "day": day_key,
                    "loop_seconds": int(cfg.LOOP_SECONDS),
                    "watchlist_size": int(len(wl)),
                    "trades_today": int(trades_today),
                    "open_positions": int(len(open_positions)),
                    "max_trades_cap_runtime": int(max_trades_cap_runtime),
                    "max_open_positions_runtime": int(max_open_positions_runtime),
                    "risk_per_trade_runtime": float(risk_per_trade_runtime),
                    "max_position_notional_pct_runtime": float(max_position_notional_pct_runtime),
                    "max_gross_leverage_runtime": float(max_gross_leverage_runtime),
                    "external_runtime_scale": float(ext_runtime_scale),
                    "external_position_risk_scale": float(ext_position_risk_scale),
                    "external_runtime_flags": list(ext_runtime_diag.get("flags", []))
                    if isinstance(ext_runtime_diag.get("flags", []), list)
                    else [],
                    "external_regime": str(ext_runtime_diag.get("regime", "unknown")),
                    "external_degraded": bool(ext_runtime_diag.get("degraded", False)),
                    "external_quality_gate_ok": bool(ext_runtime_diag.get("quality_gate_ok", True)),
                    "external_overlay_stale": bool(ext_runtime_diag.get("overlay_stale", False)),
                    "external_overlay_age_source": str(ext_overlay_age_source),
                    "external_overlay_age_hours": (
                        None if ext_overlay_age_hours is None else float(ext_overlay_age_hours)
                    ),
                    "external_overlay_generated_at_utc": ext_overlay_generated_at_utc,
                    "overlay_block_new_entries": bool(overlay_block_new_entries),
                    "overlay_block_reasons": list(overlay_block_reasons),
                    "memory_feedback_active": bool(memory_feedback_active),
                    "memory_feedback_status": str(memory_feedback_status),
                    "memory_feedback_reasons": list(memory_feedback_reasons),
                    "memory_feedback_risk_scale": float(memory_feedback_risk_scale),
                    "memory_feedback_trades_scale": float(memory_feedback_trades_scale),
                    "memory_feedback_open_scale": float(memory_feedback_open_scale),
                    "memory_feedback_turnover_pressure": (
                        None
                        if memory_feedback_turnover_pressure is None
                        else float(memory_feedback_turnover_pressure)
                    ),
                    "memory_feedback_turnover_dampener": (
                        None
                        if memory_feedback_turnover_dampener is None
                        else float(memory_feedback_turnover_dampener)
                    ),
                    "memory_feedback_block_new_entries": bool(memory_feedback_block_new_entries),
                    "memory_replay_enabled": bool(
                        bool(getattr(cfg, "MEMORY_REPLAY_ENABLED", True))
                        and bool(getattr(cfg, "MEMORY_ENABLE", False))
                    ),
                    "memory_replay_interval_sec": float(max(5.0, float(getattr(cfg, "MEMORY_REPLAY_INTERVAL_SEC", 300.0)))),
                    "memory_replay_last_ts_utc": last_memory_replay_ts_utc,
                    "memory_replay_last_ok": (
                        bool(last_memory_replay_result.get("ok"))
                        if "ok" in last_memory_replay_result
                        else None
                    ),
                    "memory_replay_last_error": (
                        str(last_memory_replay_result.get("error")).strip()
                        if last_memory_replay_result.get("error")
                        else None
                    ),
                    "memory_replay_last_replayed": int(last_memory_replay_result.get("replayed", 0)),
                    "memory_replay_last_failed": int(last_memory_replay_result.get("failed", 0)),
                    "memory_replay_last_processed_files": int(last_memory_replay_result.get("processed_files", 0)),
                    "memory_replay_queued_files": (
                        int(last_memory_replay_result.get("queued_files"))
                        if last_memory_replay_result.get("queued_files") is not None
                        else None
                    ),
                    "memory_replay_remaining_files": (
                        int(last_memory_replay_result.get("remaining_files"))
                        if last_memory_replay_result.get("remaining_files") is not None
                        else None
                    ),
                    "memory_outbox_warn_files": int(max(1, int(getattr(cfg, "MEMORY_OUTBOX_WARN_FILES", 5)))),
                    "memory_outbox_alert_files": int(
                        max(
                            int(max(1, int(getattr(cfg, "MEMORY_OUTBOX_WARN_FILES", 5)))) + 1,
                            int(getattr(cfg, "MEMORY_OUTBOX_ALERT_FILES", 20)),
                        )
                    ),
                    "aion_feedback_active": bool(aion_feedback_active),
                    "aion_feedback_status": str(aion_feedback_status),
                    "aion_feedback_source": str(aion_feedback_source),
                    "aion_feedback_source_selected": str(aion_feedback_source_selected),
                    "aion_feedback_source_preference": str(aion_feedback_source_preference),
                    "aion_feedback_reasons": list(aion_feedback_reasons),
                    "aion_feedback_risk_scale": float(aion_feedback_risk_scale),
                    "aion_feedback_closed_trades": int(aion_feedback_closed_trades),
                    "aion_feedback_hit_rate": (
                        None if aion_feedback_hit_rate is None else float(aion_feedback_hit_rate)
                    ),
                    "aion_feedback_profit_factor": (
                        None if aion_feedback_profit_factor is None else float(aion_feedback_profit_factor)
                    ),
                    "aion_feedback_expectancy": (
                        None if aion_feedback_expectancy is None else float(aion_feedback_expectancy)
                    ),
                    "aion_feedback_drawdown_norm": (
                        None if aion_feedback_drawdown_norm is None else float(aion_feedback_drawdown_norm)
                    ),
                    "aion_feedback_age_hours": (
                        None if aion_feedback_age_hours is None else float(aion_feedback_age_hours)
                    ),
                    "aion_feedback_max_age_hours": (
                        None if aion_feedback_max_age_hours is None else float(aion_feedback_max_age_hours)
                    ),
                    "aion_feedback_stale": bool(aion_feedback_stale),
                    "aion_feedback_last_closed_ts": aion_feedback_last_closed_ts,
                    "aion_feedback_path": str(aion_feedback_path),
                    "aion_feedback_block_new_entries": bool(aion_feedback_block_new_entries),
                    "exec_governor_state": str(exec_governor_state),
                    "exec_governor_reasons": list(exec_governor_reasons),
                    "exec_governor_recent_executions": int(exec_governor_recent_executions),
                    "exec_governor_exec_rate_per_min": float(exec_governor_exec_rate_per_min),
                    "exec_governor_avg_slippage_bps": (
                        None if exec_governor_avg_slippage_bps is None else float(exec_governor_avg_slippage_bps)
                    ),
                    "exec_governor_p90_slippage_bps": (
                        None if exec_governor_p90_slippage_bps is None else float(exec_governor_p90_slippage_bps)
                    ),
                    "exec_governor_block_new_entries": bool(exec_governor_block_new_entries),
                    "killswitch_block_new_entries": bool(killswitch_block_new_entries),
                    "policy_block_new_entries": bool(policy_block_new_entries),
                    "policy_loss_hit": bool(policy_loss_hit),
                    "policy_daily_loss_abs": float(policy_daily_loss_abs),
                    "policy_daily_loss_pct": float(policy_daily_loss_pct),
                }
            )

            memory_runtime_context = _compact_memory_runtime_context(
                ext_runtime_scale=float(ext_runtime_scale),
                ext_position_risk_scale=float(ext_position_risk_scale),
                ext_runtime_diag=ext_runtime_diag,
                ext_overlay_age_hours=ext_overlay_age_hours,
                ext_overlay_age_source=str(ext_overlay_age_source),
                memory_feedback_status=str(memory_feedback_status),
                memory_feedback_risk_scale=float(memory_feedback_risk_scale),
                memory_feedback_turnover_pressure=memory_feedback_turnover_pressure,
                memory_feedback_turnover_dampener=memory_feedback_turnover_dampener,
                memory_feedback_block_new_entries=bool(memory_feedback_block_new_entries),
                aion_feedback_status=str(aion_feedback_status),
                aion_feedback_source=str(aion_feedback_source),
                aion_feedback_source_selected=str(aion_feedback_source_selected),
                aion_feedback_source_preference=str(aion_feedback_source_preference),
                aion_feedback_risk_scale=float(aion_feedback_risk_scale),
                aion_feedback_stale=bool(aion_feedback_stale),
                aion_feedback_block_new_entries=bool(aion_feedback_block_new_entries),
                policy_block_new_entries=bool(policy_block_new_entries),
                killswitch_block_new_entries=bool(killswitch_block_new_entries),
                exec_governor_state=str(exec_governor_state),
                exec_governor_block_new_entries=bool(exec_governor_block_new_entries),
            )
            governor_compound_scalar = float(
                max(
                    0.0,
                    min(
                        2.0,
                        float(ext_runtime_scale)
                        * float(ext_position_risk_scale)
                        * float(memory_feedback_risk_scale)
                        * float(aion_feedback_risk_scale),
                    ),
                )
            )

            for sym in wl:
                try:
                    df = hist_bars_cached(
                        sym,
                        duration=cfg.HIST_DURATION,
                        barSize=cfg.HIST_BAR_SIZE,
                        ttl_seconds=cfg.MAIN_BARS_CACHE_SEC,
                    )
                    if df.empty or len(df) < cfg.MIN_BARS:
                        continue

                    feats = compute_features(df, cfg)
                    last = feats.iloc[-1]

                    price = float(last["close"])
                    atr_val = float(last["atr"])
                    atr_pct = float(last["atr_pct"])
                    adx_val = float(last["adx"])
                    if not (_isfinite(price) and _isfinite(atr_val) and atr_val > 0):
                        continue

                    last_prices[sym] = price
                    lookback = min(cfg.SWING_LOOKBACK, len(df))
                    high = float(df["high"].iloc[-lookback:].max())
                    low = float(df["low"].iloc[-lookback:].min())

                    ext_sig = blend_external_signals(
                        external_signals.get(sym),
                        global_external,
                        max_bias=cfg.EXT_SIGNAL_MAX_BIAS,
                    )
                    ext_sig = _scale_external_signal(ext_sig, ext_runtime_scale, cfg.EXT_SIGNAL_MAX_BIAS)

                    signal = build_trade_signal(
                        last,
                        price,
                        high,
                        low,
                        cfg,
                        profile=profile,
                        external=ext_sig,
                    )
                    base_conf = float(signal.get("confidence", 0.0))
                    intraday_score = 1.0
                    intraday_reasons = []
                    mtf_score = 1.0
                    mtf_reasons = []
                    meta_prob = 1.0
                    intraday_gate = "off" if (not cfg.INTRADAY_CONFIRM_ENABLED) else "skip"
                    mtf_gate = "off" if (not cfg.MTF_CONFIRM_ENABLED) else "skip"
                    meta_gate = "off" if (not cfg.META_LABEL_ENABLED) else "skip"

                    if signal["side"] and cfg.INTRADAY_CONFIRM_ENABLED:
                        intraday_score, intraday_reasons = intraday_entry_alignment(feats, signal["side"], cfg)
                        if intraday_score < float(cfg.INTRADAY_MIN_ALIGNMENT_SCORE):
                            signal["reasons"].append(f"Intraday blocked ({intraday_score:.2f})")
                            signal["reasons"].extend(intraday_reasons)
                            signal["side"] = None
                            intraday_gate = "block"
                        else:
                            mult = float(cfg.INTRADAY_CONF_BASE) + float(cfg.INTRADAY_CONF_GAIN) * float(intraday_score)
                            signal["confidence"] = min(1.0, base_conf * max(0.4, mult))
                            signal["reasons"].append(f"Intraday align {intraday_score:.2f}")
                            signal["reasons"].extend(intraday_reasons)
                            base_conf = float(signal.get("confidence", base_conf))
                            intraday_gate = "pass"

                    if signal["side"] and cfg.MTF_CONFIRM_ENABLED:
                        df_1h = hist_bars_cached(
                            sym,
                            duration=cfg.MTF_1H_DURATION,
                            barSize=cfg.MTF_1H_BAR,
                            ttl_seconds=cfg.MTF_1H_CACHE_SEC,
                        )
                        df_4h = hist_bars_cached(
                            sym,
                            duration=cfg.MTF_4H_DURATION,
                            barSize=cfg.MTF_4H_BAR,
                            ttl_seconds=cfg.MTF_4H_CACHE_SEC,
                        )
                        mtf_score, mtf_reasons = multi_timeframe_alignment(df_1h, df_4h, signal["side"], cfg)
                        if mtf_score < cfg.MTF_MIN_ALIGNMENT_SCORE:
                            signal["reasons"].append(f"MTF blocked ({mtf_score:.2f})")
                            signal["side"] = None
                            mtf_gate = "block"
                        else:
                            signal["confidence"] = min(1.0, base_conf * (0.70 + 0.55 * mtf_score))
                            signal["reasons"].extend(mtf_reasons)
                            mtf_gate = "pass"

                    if signal["side"] and cfg.META_LABEL_ENABLED:
                        meta_prob = meta_model.predict_proba(
                            confidence=float(signal["confidence"]),
                            long_conf=float(signal["long_conf"]),
                            short_conf=float(signal["short_conf"]),
                            adx=adx_val,
                            atr_pct=atr_pct,
                            regime=str(signal["regime"]),
                        )
                        if meta_prob < cfg.META_LABEL_MIN_PROB:
                            signal["reasons"].append(f"Meta-label veto ({meta_prob:.2f})")
                            signal["side"] = None
                            meta_gate = "block"
                        else:
                            signal["confidence"] = min(1.0, float(signal["confidence"]) * (0.60 + 0.70 * meta_prob))
                            meta_gate = "pass"

                    cycle_conf.append(max(float(signal["long_conf"]), float(signal["short_conf"])))
                    log_signal(
                        now(),
                        sym,
                        signal["regime"],
                        float(signal["long_conf"]),
                        float(signal["short_conf"]),
                        signal["side"] or "HOLD",
                        signal["reasons"],
                        meta_prob=meta_prob,
                        mtf_score=mtf_score,
                        intraday_score=intraday_score,
                        intraday_gate=intraday_gate,
                        mtf_gate=mtf_gate,
                        meta_gate=meta_gate,
                        pattern_hits=int(signal.get("pattern_hits", 0)),
                        indicator_hits=int(signal.get("indicator_hits", 0)),
                    )

                    pos = open_positions.get(sym)

                    # Manage open positions
                    if pos:
                        pos["bars_held"] += 1
                        pos["mark_price"] = price

                        if pos["side"] == "LONG":
                            pos["peak_price"] = max(pos["peak_price"], price)
                            if (
                                bool(getattr(cfg, "PARTIAL_PROFIT_ENABLED", True))
                                and (not pos["partial_taken"])
                                and price >= _partial_profit_target_price(
                                    pos["entry"],
                                    pos.get("init_risk", 0.0),
                                    "LONG",
                                    getattr(cfg, "PARTIAL_PROFIT_R_MULTIPLE", cfg.PARTIAL_TAKE_R),
                                )
                            ):
                                cash, closed_pnl, fully_closed = _partial_close(
                                    pos,
                                    sym,
                                    price,
                                    exe,
                                    cash,
                                    closed_pnl,
                                    ks,
                                    today,
                                    monitor=monitor,
                                    memory_runtime_context=memory_runtime_context,
                                    telemetry_sink=telemetry_sink,
                                    governor_compound_scalar=governor_compound_scalar,
                                    ib_client=ib_client,
                                )
                                monitor.record_execution(cfg.SLIPPAGE_BPS)
                                if fully_closed:
                                    del open_positions[sym]
                                    cooldown[sym] = cfg.REENTRY_COOLDOWN_CYCLES
                                    continue
                                pos["stop"] = max(pos["stop"], pos["entry"]) if pos["qty"] > 0 else pos["stop"]

                            if bool(getattr(cfg, "TRAILING_STOP_ENABLED", True)) and pos.get("partial_taken", False):
                                trail_candidate = _trailing_stop_candidate(
                                    "LONG",
                                    pos.get("peak_price", price),
                                    atr_val,
                                    getattr(cfg, "TRAILING_STOP_ATR_MULTIPLE", cfg.TRAIL_ATR_MULT),
                                )
                                pos["trail_stop"] = max(pos["trail_stop"], trail_candidate, pos["entry"])

                            stop_ref, trail_active = _effective_stop_price(pos)
                            trailing_trigger = bool(
                                trail_active and price <= float(pos.get("trail_stop", stop_ref)) and float(pos.get("trail_stop", stop_ref)) >= float(pos.get("stop", stop_ref))
                            )
                            stop_trigger = price <= stop_ref
                            target_trigger = price >= pos["target"]
                        else:
                            pos["trough_price"] = min(pos["trough_price"], price)
                            if (
                                bool(getattr(cfg, "PARTIAL_PROFIT_ENABLED", True))
                                and (not pos["partial_taken"])
                                and price <= _partial_profit_target_price(
                                    pos["entry"],
                                    pos.get("init_risk", 0.0),
                                    "SHORT",
                                    getattr(cfg, "PARTIAL_PROFIT_R_MULTIPLE", cfg.PARTIAL_TAKE_R),
                                )
                            ):
                                cash, closed_pnl, fully_closed = _partial_close(
                                    pos,
                                    sym,
                                    price,
                                    exe,
                                    cash,
                                    closed_pnl,
                                    ks,
                                    today,
                                    monitor=monitor,
                                    memory_runtime_context=memory_runtime_context,
                                    telemetry_sink=telemetry_sink,
                                    governor_compound_scalar=governor_compound_scalar,
                                    ib_client=ib_client,
                                )
                                monitor.record_execution(cfg.SLIPPAGE_BPS)
                                if fully_closed:
                                    del open_positions[sym]
                                    cooldown[sym] = cfg.REENTRY_COOLDOWN_CYCLES
                                    continue
                                pos["stop"] = min(pos["stop"], pos["entry"]) if pos["qty"] > 0 else pos["stop"]

                            if bool(getattr(cfg, "TRAILING_STOP_ENABLED", True)) and pos.get("partial_taken", False):
                                trail_candidate = _trailing_stop_candidate(
                                    "SHORT",
                                    pos.get("trough_price", price),
                                    atr_val,
                                    getattr(cfg, "TRAILING_STOP_ATR_MULTIPLE", cfg.TRAIL_ATR_MULT),
                                )
                                pos["trail_stop"] = min(pos["trail_stop"], trail_candidate, pos["entry"])

                            stop_ref, trail_active = _effective_stop_price(pos)
                            trailing_trigger = bool(
                                trail_active and price >= float(pos.get("trail_stop", stop_ref)) and float(pos.get("trail_stop", stop_ref)) <= float(pos.get("stop", stop_ref))
                            )
                            stop_trigger = price >= stop_ref
                            target_trigger = price <= pos["target"]

                        opposite = opposite_confidence(pos["side"], signal) >= float(signal["opposite_exit_threshold"])
                        timeout = pos["bars_held"] >= cfg.MAX_HOLD_CYCLES

                        if stop_trigger or target_trigger or opposite or timeout:
                            side = "SELL" if pos["side"] == "LONG" else "BUY"
                            if stop_trigger:
                                reason = "trailing_stop" if trailing_trigger else "initial_stop"
                            elif target_trigger:
                                reason = "target_hit"
                            elif opposite:
                                reason = "opposite_high_confidence_signal"
                            else:
                                reason = "time_stop"

                            req_qty = int(max(0, int(pos.get("qty", 0))))
                            audit_log(
                                {
                                    "event": "ORDER_INTENT",
                                    "symbol": str(sym).upper(),
                                    "side": str(side).upper(),
                                    "qty": int(req_qty),
                                    "reason": str(reason),
                                },
                                log_dir=Path(cfg.STATE_DIR),
                            )
                            allowed, gate_reason, gate_diag = _run_exposure_gate(
                                client=ib_client,
                                symbol=sym,
                                side=side,
                                qty=req_qty,
                                price=float(price),
                                fallback_nlv=max(1.0, float(equity)),
                            )
                            audit_log(
                                {
                                    "event": ("EXPOSURE_GATE_PASS" if allowed else "EXPOSURE_GATE_VETO"),
                                    "symbol": str(sym).upper(),
                                    "side": str(side).upper(),
                                    "qty": int(req_qty),
                                    "reason": str(gate_reason),
                                    **gate_diag,
                                },
                                log_dir=Path(cfg.STATE_DIR),
                            )
                            if not allowed:
                                log_run(f"EXPOSURE GATE VETO: {sym} {side} qty={req_qty} reason={gate_reason}")
                                send_alert(f"Exposure gate vetoed {sym} exit: {gate_reason}", level="WARNING")
                                continue

                            audit_log(
                                {
                                    "event": "ORDER_SUBMITTED",
                                    "symbol": str(sym).upper(),
                                    "side": str(side).upper(),
                                    "qty": int(req_qty),
                                    "reason": str(reason),
                                },
                                log_dir=Path(cfg.STATE_DIR),
                            )
                            fill = _execute_fill(
                                exe,
                                side=side,
                                qty=req_qty,
                                ref_price=price,
                                atr_pct=atr_pct,
                                confidence=float(pos.get("confidence", 0.5)),
                                allow_partial=False,
                                symbol=str(sym),
                                ib_client=ib_client,
                            )
                            filled_qty = int(max(0, min(req_qty, int(getattr(fill, "filled_qty", 0)))))
                            if filled_qty <= 0:
                                audit_log(
                                    {
                                        "event": "ORDER_REJECTED",
                                        "symbol": str(sym).upper(),
                                        "side": str(side).upper(),
                                        "qty": int(req_qty),
                                        "reason": str(reason),
                                    },
                                    log_dir=Path(cfg.STATE_DIR),
                                )
                                continue

                            monitor.record_execution(fill.est_slippage_bps)
                            audit_log(
                                {
                                    "event": "ORDER_PARTIAL_FILL" if filled_qty < req_qty else "ORDER_FILLED",
                                    "symbol": str(sym).upper(),
                                    "side": str(side).upper(),
                                    "qty_requested": int(req_qty),
                                    "qty_filled": int(filled_qty),
                                    "avg_fill_price": float(fill.avg_fill),
                                    "reason": str(reason),
                                },
                                log_dir=Path(cfg.STATE_DIR),
                            )
                            cash, closed_pnl = _close_position(
                                open_positions,
                                sym,
                                fill.avg_fill,
                                reason,
                                cash,
                                closed_pnl,
                                ks,
                                today,
                                fill_ratio=fill.fill_ratio,
                                slippage_bps=fill.est_slippage_bps,
                                monitor=monitor,
                                memory_runtime_context=memory_runtime_context,
                                telemetry_sink=telemetry_sink,
                                governor_compound_scalar=governor_compound_scalar,
                                filled_qty=filled_qty,
                                sync_shadow=(str(getattr(fill, "source", "simulator")).strip().lower() != "ib_paper"),
                            )
                            fully_closed = sym not in open_positions
                            if fully_closed:
                                cooldown[sym] = cfg.REENTRY_COOLDOWN_CYCLES
                            continue

                        continue

                    # Candidate collection for portfolio allocator
                    if killswitch_block_new_entries:
                        continue
                    if policy_block_new_entries:
                        continue
                    if not symbol_allowed(sym, policy_caps):
                        continue
                    if sym in cooldown:
                        continue
                    if trades_today >= max_trades_cap_runtime:
                        continue
                    if signal["side"] is None:
                        continue
                    if len(open_positions) + len(entry_candidates) >= max_open_positions_runtime:
                        continue

                    blocked, reason = event_filter.blocked(sym)
                    if blocked:
                        log_run(f"{sym}: event filter blocked entry ({reason})")
                        continue

                    full_rets = df["close"].pct_change().dropna()
                    rets = full_rets.tail(80)
                    volatility = float(rets.std()) if not rets.empty else 0.01
                    margin = abs(float(signal["long_conf"]) - float(signal["short_conf"]))
                    expected_edge = float(signal["confidence"]) * (0.6 + margin)
                    stop_atr_mult, stop_vol_expanded = _entry_stop_atr_multiple(signal["side"], full_rets.values)

                    entry_candidates.append(
                        {
                            "symbol": sym,
                            "side": signal["side"],
                            "signal": signal,
                            "price": price,
                            "atr": atr_val,
                            "atr_pct": atr_pct,
                            "returns": rets,
                            "volatility": max(1e-6, volatility),
                            "confidence": float(signal["confidence"]),
                            "expected_edge": expected_edge,
                            "stop_atr_mult": float(stop_atr_mult),
                            "stop_vol_expanded": bool(stop_vol_expanded),
                            "q_overlay_bias": (
                                _safe_float(ext_sig.get("bias"), 0.0) if isinstance(ext_sig, dict) else 0.0
                            ),
                            "q_overlay_confidence": (
                                _safe_float(ext_sig.get("confidence"), 0.0) if isinstance(ext_sig, dict) else 0.0
                            ),
                            "intraday_score": float(intraday_score),
                        }
                    )

                except Exception as exc:
                    log_run(f"{sym}: loop error: {exc}")
                    continue

            # Portfolio-level entry selection and sizing
            allocations = {}
            if entry_candidates:
                if cfg.PORTFOLIO_ENABLE:
                    allocations = allocate_candidates(entry_candidates, equity, cfg)
                else:
                    gross = equity * max_gross_leverage_runtime
                    per = gross / max(len(entry_candidates), 1)
                    allocations = {
                        c["symbol"]: {"target_notional": per, "weight": 1.0 / len(entry_candidates), "side": c["side"]}
                        for c in entry_candidates
                    }

            for c in sorted(entry_candidates, key=lambda x: x["confidence"], reverse=True):
                if killswitch_block_new_entries or policy_block_new_entries:
                    break
                if trades_today >= max_trades_cap_runtime:
                    break
                if len(open_positions) >= max_open_positions_runtime:
                    break

                sym = c["symbol"]
                if sym in open_positions:
                    continue
                if not symbol_allowed(sym, policy_caps):
                    continue

                alloc = allocations.get(sym, {})
                target_notional = float(alloc.get("target_notional", equity * max_position_notional_pct_runtime))
                qty_port = int(target_notional / max(c["price"], 1e-6))

                qty_risk = risk_qty(
                    equity=equity,
                    risk_per_trade=risk_per_trade_runtime,
                    atr=c["atr"],
                    price=c["price"],
                    confidence=c["confidence"],
                    stop_atr_mult=float(c.get("stop_atr_mult", c["signal"]["stop_atr_mult"])),
                    max_notional_pct=max_position_notional_pct_runtime,
                )
                qty = min(qty_port, qty_risk)
                if qty <= 0:
                    continue

                entry_side = "BUY" if c["side"] == "LONG" else "SELL"
                audit_log(
                    {
                        "event": "ORDER_INTENT",
                        "symbol": str(sym).upper(),
                        "side": str(entry_side).upper(),
                        "qty": int(qty),
                        "reason": "entry_signal",
                        "confidence": float(c.get("confidence", 0.0)),
                    },
                    log_dir=Path(cfg.STATE_DIR),
                )
                allowed, gate_reason, gate_diag = _run_exposure_gate(
                    client=ib_client,
                    symbol=sym,
                    side=entry_side,
                    qty=int(qty),
                    price=float(c["price"]),
                    fallback_nlv=max(1.0, float(equity)),
                )
                audit_log(
                    {
                        "event": ("EXPOSURE_GATE_PASS" if allowed else "EXPOSURE_GATE_VETO"),
                        "symbol": str(sym).upper(),
                        "side": str(entry_side).upper(),
                        "qty": int(qty),
                        "reason": str(gate_reason),
                        **gate_diag,
                    },
                    log_dir=Path(cfg.STATE_DIR),
                )
                if not allowed:
                    log_run(f"EXPOSURE GATE VETO: {sym} {entry_side} qty={qty} reason={gate_reason}")
                    send_alert(f"Exposure gate vetoed {sym} entry: {gate_reason}", level="WARNING")
                    continue
                audit_log(
                    {
                        "event": "ORDER_SUBMITTED",
                        "symbol": str(sym).upper(),
                        "side": str(entry_side).upper(),
                        "qty": int(qty),
                        "reason": "entry_signal",
                    },
                    log_dir=Path(cfg.STATE_DIR),
                )
                fill = _execute_fill(
                    exe,
                    side=entry_side,
                    qty=qty,
                    ref_price=c["price"],
                    atr_pct=c["atr_pct"],
                    confidence=c["confidence"],
                    allow_partial=True,
                    symbol=str(sym),
                    ib_client=ib_client,
                )
                monitor.record_execution(fill.est_slippage_bps)
                if fill.filled_qty <= 0:
                    audit_log(
                        {
                            "event": "ORDER_REJECTED",
                            "symbol": str(sym).upper(),
                            "side": str(entry_side).upper(),
                            "qty": int(qty),
                            "reason": "entry_signal",
                        },
                        log_dir=Path(cfg.STATE_DIR),
                    )
                    continue
                audit_log(
                    {
                        "event": "ORDER_PARTIAL_FILL" if int(fill.filled_qty) < int(qty) else "ORDER_FILLED",
                        "symbol": str(sym).upper(),
                        "side": str(entry_side).upper(),
                        "qty_requested": int(qty),
                        "qty_filled": int(fill.filled_qty),
                        "avg_fill_price": float(fill.avg_fill),
                        "reason": "entry_signal",
                    },
                    log_dir=Path(cfg.STATE_DIR),
                )

                notional = fill.filled_qty * fill.avg_fill
                if not gross_leverage_ok(
                    cash=cash,
                    open_positions=open_positions,
                    next_notional=notional,
                    max_gross_leverage=max_gross_leverage_runtime,
                    equity=max(equity, 1.0),
                ):
                    continue

                if entry_side == "BUY":
                    if cash < notional:
                        continue
                    cash -= notional
                else:
                    cash += notional

                if str(getattr(fill, "source", "simulator")).strip().lower() != "ib_paper":
                    try:
                        apply_shadow_fill(
                            Path(cfg.STATE_DIR) / "shadow_trades.json",
                            symbol=str(sym).upper(),
                            action=str(entry_side).upper(),
                            filled_qty=int(fill.filled_qty),
                            avg_fill_price=float(fill.avg_fill),
                            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
                        )
                    except Exception as exc:
                        log_run(f"shadow update failed on entry {sym}: {exc}")

                init_risk = max(c["atr"] * float(c.get("stop_atr_mult", c["signal"]["stop_atr_mult"])), fill.avg_fill * 0.0035)
                if c["side"] == "LONG":
                    stop = fill.avg_fill - init_risk
                    target = fill.avg_fill + (c["atr"] * float(c["signal"]["target_atr_mult"]))
                    trail_stop = stop
                    peak_price = fill.avg_fill
                    trough_price = fill.avg_fill
                else:
                    stop = fill.avg_fill + init_risk
                    target = fill.avg_fill - (c["atr"] * float(c["signal"]["target_atr_mult"]))
                    trail_stop = stop
                    peak_price = fill.avg_fill
                    trough_price = fill.avg_fill

                open_positions[sym] = {
                    "qty": int(fill.filled_qty),
                    "side": c["side"],
                    "entry": float(fill.avg_fill),
                    "entry_ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "mark_price": float(fill.avg_fill),
                    "stop": float(stop),
                    "target": float(target),
                    "trail_stop": float(trail_stop),
                    "trail_mult": float(getattr(cfg, "TRAILING_STOP_ATR_MULTIPLE", c["signal"]["trail_atr_mult"])),
                    "init_risk": float(init_risk),
                    "bars_held": 0,
                    "partial_taken": False,
                    "peak_price": float(peak_price),
                    "trough_price": float(trough_price),
                    "confidence": float(c["confidence"]),
                    "regime": str(c["signal"]["regime"]),
                    "atr_pct": float(c["atr_pct"]),
                    "stop_atr_mult": float(c.get("stop_atr_mult", c["signal"]["stop_atr_mult"])),
                    "stop_vol_expanded": bool(c.get("stop_vol_expanded", False)),
                    "q_overlay_bias": float(_safe_float(c.get("q_overlay_bias"), 0.0)),
                    "q_overlay_confidence": float(_safe_float(c.get("q_overlay_confidence"), 0.0)),
                    "intraday_score": float(_safe_float(c.get("intraday_score"), 0.0)),
                }
                trades_today += 1

                log_trade(
                    now(),
                    sym,
                    f"ENTRY_{entry_side}",
                    int(fill.filled_qty),
                    float(fill.avg_fill),
                    0.0,
                    0.0,
                    f"{confidence_tag(float(c['confidence']))} conf | {'; '.join(c['signal']['reasons'][:3])}",
                    confidence=float(c["confidence"]),
                    regime=str(c["signal"]["regime"]),
                    stop=float(stop),
                    target=float(target),
                    trail=float(trail_stop),
                    fill_ratio=float(fill.fill_ratio),
                    slippage_bps=float(fill.est_slippage_bps),
                )
                _emit_trade_memory_event(
                    event_type="trade.entry",
                    symbol=sym,
                    side=str(c["side"]),
                    qty=int(fill.filled_qty),
                    entry=float(fill.avg_fill),
                    exit_=0.0,
                    pnl=0.0,
                    reason=f"{confidence_tag(float(c['confidence']))} conf entry",
                    confidence=float(c["confidence"]),
                    regime=str(c["signal"]["regime"]),
                    monitor=monitor,
                    extra={
                        "stop": float(stop),
                        "target": float(target),
                        "trail": float(trail_stop),
                        "fill_ratio": float(fill.fill_ratio),
                        "slippage_bps": float(fill.est_slippage_bps),
                        "atr_pct": float(c["atr_pct"]),
                        "runtime_context": dict(memory_runtime_context),
                    },
                )
                _telemetry_log_trade_decision(
                    telemetry_sink,
                    symbol=sym,
                    decision=("ENTER_LONG" if c["side"] == "LONG" else "ENTER_SHORT"),
                    q_overlay_bias=_safe_float(c.get("q_overlay_bias"), 0.0),
                    q_overlay_confidence=_safe_float(c.get("q_overlay_confidence"), 0.0),
                    confluence_score=float(c["confidence"]),
                    intraday_alignment_score=_safe_float(c.get("intraday_score"), 0.0),
                    regime=str(c["signal"]["regime"]),
                    governor_compound_scalar=governor_compound_scalar,
                    entry_price=float(fill.avg_fill),
                    stop_price=float(stop),
                    risk_distance=float(init_risk),
                    position_size_shares=int(fill.filled_qty),
                    reasons=list(c["signal"]["reasons"][:6]),
                    slippage_bps=float(fill.est_slippage_bps),
                    estimated_slippage_bps=float(fill.est_slippage_bps),
                    extras={
                        "fill_ratio": float(fill.fill_ratio),
                        "target_price": float(target),
                        "trail_price": float(trail_stop),
                    },
                )

            open_pnl = _mark_open_pnl(open_positions, last_prices)
            equity = _equity_from_cash_and_positions(cash, open_positions, last_prices)
            log_equity(now(), equity, cash, open_pnl, closed_pnl)
            save_runtime_state(day_key, cash, closed_pnl, trades_today, open_positions, cooldown)
            last_telemetry_summary_ts = _maybe_update_telemetry_summary(last_telemetry_summary_ts)
            try:
                write_system_health(state_dir=Path(cfg.STATE_DIR), log_dir=Path(cfg.LOG_DIR))
            except Exception:
                pass

            if cfg.MONITORING_ENABLED:
                avg_conf = float(sum(cycle_conf) / len(cycle_conf)) if cycle_conf else 0.0
                monitor.record_cycle(equity=equity, avg_conf=avg_conf)
                alerts = monitor.check_alerts()
                for msg in alerts:
                    log_alert(msg)
                    log_run(f"ALERT: {msg}")

            time.sleep(cfg.LOOP_SECONDS)

    finally:
        try:
            _persist_ib_order_state(ib())
        except Exception:
            pass
        disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
