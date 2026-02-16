from __future__ import annotations


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "on"}


def _to_float(x, default: float | None = None):
    try:
        v = float(x)
    except Exception:
        return default
    if v != v:  # NaN
        return default
    return v


def _uniq(items):
    out = []
    seen = set()
    for raw in items:
        s = str(raw).strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _action(aid: str, priority: int, title: str, why: str, steps: list[str]):
    return {
        "id": str(aid),
        "priority": int(priority),
        "title": str(title),
        "why": str(why),
        "steps": _uniq([str(x) for x in steps if str(x).strip()]),
    }


def _remediation_actions(blocked_reasons: list[str], throttle_reasons: list[str]) -> list[dict]:
    acts = []
    br = [str(x).strip().lower() for x in blocked_reasons if str(x).strip()]
    tr = [str(x).strip().lower() for x in throttle_reasons if str(x).strip()]

    if any(x == "killswitch" for x in br):
        acts.append(
            _action(
                "killswitch_review",
                0,
                "Review kill switch",
                "New entries are blocked by the kill switch.",
                [
                    "Inspect /Users/desmondpottle/Documents/New project/aion/logs/killswitch.json",
                    "Check recent losses in /Users/desmondpottle/Documents/New project/aion/logs/shadow_trades.csv",
                    "Only reset after validating drawdown root cause",
                ],
            )
        )
    if any(x == "risk_policy" for x in br):
        acts.append(
            _action(
                "risk_policy_limits",
                0,
                "Review risk policy caps",
                "Policy caps are actively blocking entries.",
                [
                    "Inspect /Users/desmondpottle/Documents/New project/aion/state/risk_policy.json",
                    "Compare policy caps with /Users/desmondpottle/Documents/New project/aion/state/runtime_controls.json",
                    "Adjust caps only if current market/liquidity regime supports it",
                ],
            )
        )
    if any(x.startswith("external_overlay") for x in br):
        acts.append(
            _action(
                "overlay_refresh",
                1,
                "Refresh external overlay",
                "External Q overlay gating is blocking entries.",
                [
                    "Regenerate Q overlay via q/tools/export_aion_signal_pack.py",
                    "Validate /Users/desmondpottle/Documents/New project/q/runs_plus/q_signal_overlay.json timestamp and risk_flags",
                    "Confirm AION EXT_SIGNAL_FILE points to the active overlay path",
                ],
            )
        )
    if any(x.startswith("memory_feedback") for x in br) or any("memory_feedback" in x for x in tr):
        acts.append(
            _action(
                "novaspine_feedback",
                1,
                "Validate NovaSpine feedback loop",
                "NovaSpine memory feedback indicates degraded confidence.",
                [
                    "Check q/runs_plus/novaspine_context.json and q/runs_plus/novaspine_hive_feedback.json",
                    "If status is unreachable, verify NovaSpine API endpoint and token",
                    "Rerun q/tools/run_novaspine_context.py and q/tools/run_novaspine_hive_feedback.py",
                ],
            )
        )
    if any(x.startswith("memory_outbox") for x in br) or any("memory_outbox" in x for x in tr):
        acts.append(
            _action(
                "memory_outbox_replay",
                1,
                "Drain NovaSpine outbox backlog",
                "Queued memory events are not replaying cleanly to NovaSpine.",
                [
                    "Inspect /Users/desmondpottle/Documents/New project/aion/logs/novaspine_outbox_aion and sent/",
                    "Check memory_replay_* fields in /Users/desmondpottle/Documents/New project/aion/state/runtime_controls.json",
                    "Verify AION_MEMORY_NOVASPINE_URL and token, then restart trade loop if replay remains stalled",
                ],
            )
        )
    if any(x == "execution_governor" for x in br) or any("execution_governor" in x for x in tr):
        acts.append(
            _action(
                "execution_quality",
                1,
                "Reduce execution stress",
                "Execution quality governor is in warn/alert state.",
                [
                    "Inspect slippage trends in /Users/desmondpottle/Documents/New project/aion/logs/runtime_monitor.json",
                    "Reduce trade frequency and max open positions temporarily",
                    "Recheck spread/liquidity conditions and IB route quality",
                ],
            )
        )
    if any("overlay_risk_flag_alert" == x for x in tr):
        acts.append(
            _action(
                "overlay_risk_flags",
                2,
                "Investigate overlay alert flags",
                "Overlay risk flags indicate stressed model regime.",
                [
                    "Inspect runtime risk_flags in q_signal_overlay.json runtime_context",
                    "Review recent Q diagnostics: regime fracture, drift watch, quality snapshot",
                    "Run recalibration pipeline before increasing risk again",
                ],
            )
        )
    if any(x in {"hive_crowding_alert", "hive_crowding_warn"} for x in tr):
        acts.append(
            _action(
                "hive_crowding_rebalance",
                1,
                "Reduce hive crowding risk",
                "Cross-hive crowding signals indicate correlated positioning pressure.",
                [
                    "Inspect /Users/desmondpottle/Documents/New project/q/runs_plus/hive_crowding_penalty.csv",
                    "Review /Users/desmondpottle/Documents/New project/q/runs_plus/cross_hive_summary.json crowding metrics",
                    "Rerun q/tools/run_cross_hive.py with tighter entropy/concentration settings if crowding persists",
                ],
            )
        )
    if any(x in {"hive_entropy_alert", "hive_entropy_warn"} for x in tr):
        acts.append(
            _action(
                "hive_entropy_regime_reset",
                1,
                "Re-evaluate regime diversification pressure",
                "Adaptive entropy pressure indicates structurally unstable cross-hive regime.",
                [
                    "Inspect /Users/desmondpottle/Documents/New project/q/runs_plus/hive_entropy_schedule.csv",
                    "Review /Users/desmondpottle/Documents/New project/q/runs_plus/cross_hive_summary.json entropy_adaptive_diagnostics",
                    "Re-run q/tools/run_cross_hive.py and q/tools/run_regime_switcher.py before restoring risk",
                ],
            )
        )
    if any(x in {"hive_turnover_alert", "hive_turnover_warn"} for x in tr):
        acts.append(
            _action(
                "hive_turnover_throttle",
                1,
                "Stabilize cross-hive turnover",
                "Cross-hive turnover pressure indicates unstable reallocation churn.",
                [
                    "Inspect /Users/desmondpottle/Documents/New project/q/runs_plus/cross_hive_summary.json turnover metrics",
                    "Set tighter CROSS_HIVE_MAX_STEP_TURNOVER and CROSS_HIVE_TURNOVER_LIMIT before next run",
                    "Re-run q/tools/run_cross_hive.py and confirm rolling_turnover_max is below alert threshold",
                ],
            )
        )
    if any(x in {"memory_turnover_alert", "memory_turnover_warn"} for x in tr):
        acts.append(
            _action(
                "memory_turnover_throttle",
                1,
                "Stabilize NovaSpine memory turnover stress",
                "NovaSpine turnover pressure indicates unstable memory-driven regime transitions.",
                [
                    "Inspect /Users/desmondpottle/Documents/New project/q/runs_plus/novaspine_context.json turnover_pressure and turnover_dampener",
                    "Inspect /Users/desmondpottle/Documents/New project/q/runs_plus/novaspine_hive_feedback.json turnover_pressure and per_hive boosts",
                    "Re-run q/tools/run_novaspine_context.py, q/tools/run_novaspine_hive_feedback.py, and q/tools/export_aion_signal_pack.py before restoring risk",
                ],
            )
        )
    if any(x.startswith("aion_feedback") for x in br) or any(
        x in {"aion_outcome_alert", "aion_outcome_warn", "aion_outcome_stale"} for x in tr
    ):
        acts.append(
            _action(
                "aion_outcome_recalibration",
                1,
                "Recalibrate outcome feedback loop",
                "Recent realized AION outcomes are degrading runtime confidence.",
                [
                    "Inspect /Users/desmondpottle/Documents/New project/aion/logs/shadow_trades.csv outcome quality",
                    "Validate Q_AION_SHADOW_TRADES path and lookback/min-trade settings in q/tools/export_aion_signal_pack.py",
                    "Regenerate q/runs_plus/q_signal_overlay.json after confirming closed-trade quality inputs",
                ],
            )
        )

    if not acts:
        acts.append(
            _action(
                "no_action",
                9,
                "No remediation required",
                "Runtime controls are normal.",
                ["Continue monitoring operator/dashboard status."],
            )
        )
    acts.sort(key=lambda a: (int(a.get("priority", 9)), str(a.get("id", ""))))
    return acts


def runtime_decision_summary(
    runtime_controls: dict | None,
    external_overlay_runtime: dict | None = None,
    external_overlay_risk_flags: list[str] | None = None,
) -> dict:
    rc = runtime_controls if isinstance(runtime_controls, dict) else {}
    ext_rt = external_overlay_runtime if isinstance(external_overlay_runtime, dict) else {}
    ext_flags = external_overlay_risk_flags if isinstance(external_overlay_risk_flags, list) else []
    ext_flags = [str(x).strip().lower() for x in ext_flags if str(x).strip()]

    blocked_reasons = []
    if _to_bool(rc.get("killswitch_block_new_entries", False)):
        blocked_reasons.append("killswitch")
    if _to_bool(rc.get("policy_block_new_entries", False)):
        blocked_reasons.append("risk_policy")
    if _to_bool(rc.get("overlay_block_new_entries", False)):
        blocked_reasons.append("external_overlay")
        for r in rc.get("overlay_block_reasons", []) if isinstance(rc.get("overlay_block_reasons", []), list) else []:
            blocked_reasons.append(f"external_overlay:{str(r).strip().lower()}")
    if _to_bool(rc.get("memory_feedback_block_new_entries", False)):
        blocked_reasons.append("memory_feedback")
        for r in rc.get("memory_feedback_reasons", []) if isinstance(rc.get("memory_feedback_reasons", []), list) else []:
            blocked_reasons.append(f"memory_feedback:{str(r).strip().lower()}")
    if _to_bool(rc.get("aion_feedback_block_new_entries", False)):
        blocked_reasons.append("aion_feedback")
        for r in rc.get("aion_feedback_reasons", []) if isinstance(rc.get("aion_feedback_reasons", []), list) else []:
            blocked_reasons.append(f"aion_feedback:{str(r).strip().lower()}")
    if _to_bool(rc.get("exec_governor_block_new_entries", False)):
        blocked_reasons.append("execution_governor")
    if _to_bool(ext_rt.get("stale", False)):
        blocked_reasons.append("external_overlay:stale")

    throttle_reasons = []
    score = 0
    pos_scale = _to_float(rc.get("external_position_risk_scale"), None)
    if pos_scale is not None:
        if pos_scale <= 0.70:
            score += 2
            throttle_reasons.append("position_risk_scale_critical")
        elif pos_scale <= 0.90:
            score += 1
            throttle_reasons.append("position_risk_scale_tight")

    rt_scale = _to_float(rc.get("external_runtime_scale"), None)
    if rt_scale is not None:
        if rt_scale <= 0.75:
            score += 2
            throttle_reasons.append("overlay_runtime_scale_critical")
        elif rt_scale <= 0.90:
            score += 1
            throttle_reasons.append("overlay_runtime_scale_tight")

    exec_state = str(rc.get("exec_governor_state", "unknown")).strip().lower()
    if exec_state == "alert":
        score += 2
        throttle_reasons.append("execution_governor_alert")
    elif exec_state == "warn":
        score += 1
        throttle_reasons.append("execution_governor_warn")

    mem_state = str(rc.get("memory_feedback_status", "unknown")).strip().lower()
    if mem_state == "alert":
        score += 2
        throttle_reasons.append("memory_feedback_alert")
    elif mem_state == "warn":
        score += 1
        throttle_reasons.append("memory_feedback_warn")
    aion_state = str(rc.get("aion_feedback_status", "unknown")).strip().lower()
    if aion_state == "alert":
        score += 2
        throttle_reasons.append("aion_outcome_alert")
    elif aion_state == "warn":
        score += 1
        throttle_reasons.append("aion_outcome_warn")
    aion_stale = _to_bool(rc.get("aion_feedback_stale", False))
    if not aion_stale:
        af_age = _to_float(rc.get("aion_feedback_age_hours"), None)
        af_max_age = _to_float(rc.get("aion_feedback_max_age_hours"), None)
        if af_age is not None and af_max_age is not None and af_max_age > 0 and af_age > af_max_age:
            aion_stale = True
    if aion_stale:
        score += 1
        throttle_reasons.append("aion_outcome_stale")

    if "hive_crowding_alert" in ext_flags:
        score += 2
        throttle_reasons.append("hive_crowding_alert")
    elif "hive_crowding_warn" in ext_flags:
        score += 1
        throttle_reasons.append("hive_crowding_warn")
    if "hive_entropy_alert" in ext_flags:
        score += 2
        throttle_reasons.append("hive_entropy_alert")
    elif "hive_entropy_warn" in ext_flags:
        score += 1
        throttle_reasons.append("hive_entropy_warn")
    if "hive_turnover_alert" in ext_flags:
        score += 2
        throttle_reasons.append("hive_turnover_alert")
    elif "hive_turnover_warn" in ext_flags:
        score += 1
        throttle_reasons.append("hive_turnover_warn")
    if "memory_turnover_alert" in ext_flags:
        score += 2
        throttle_reasons.append("memory_turnover_alert")
    elif "memory_turnover_warn" in ext_flags:
        score += 1
        throttle_reasons.append("memory_turnover_warn")
    if "aion_outcome_alert" in ext_flags:
        score += 2
        throttle_reasons.append("aion_outcome_alert")
    elif "aion_outcome_warn" in ext_flags:
        score += 1
        throttle_reasons.append("aion_outcome_warn")
    if "aion_outcome_stale" in ext_flags:
        score += 1
        throttle_reasons.append("aion_outcome_stale")

    mem_replay_enabled = _to_bool(rc.get("memory_replay_enabled", False))
    mem_replay_last_ok = rc.get("memory_replay_last_ok", None)
    mem_replay_last_error = str(rc.get("memory_replay_last_error", "")).strip().lower()
    mem_replay_failed = _to_float(rc.get("memory_replay_last_failed"), 0.0) or 0.0
    mem_replay_remaining = _to_float(rc.get("memory_replay_remaining_files"), None)
    mem_replay_queued = _to_float(rc.get("memory_replay_queued_files"), None)
    outbox_warn_n = max(1, int((_to_float(rc.get("memory_outbox_warn_files"), 5.0) or 5.0)))
    outbox_alert_n = max(outbox_warn_n + 1, int((_to_float(rc.get("memory_outbox_alert_files"), 20.0) or 20.0)))
    outbox_backlog = max(
        int(mem_replay_remaining) if mem_replay_remaining is not None else 0,
        int(mem_replay_queued) if mem_replay_queued is not None else 0,
    )
    if mem_replay_enabled:
        if mem_replay_failed > 0:
            score += 2
            throttle_reasons.append("memory_outbox_alert")
        elif bool(mem_replay_last_ok is False) and ("unreachable" in mem_replay_last_error or "health_status" in mem_replay_last_error):
            score += 2
            throttle_reasons.append("memory_outbox_alert")
        elif outbox_backlog >= outbox_alert_n:
            score += 2
            throttle_reasons.append("memory_outbox_alert")
        elif outbox_backlog >= outbox_warn_n:
            score += 1
            throttle_reasons.append("memory_outbox_warn")
        elif bool(mem_replay_last_ok is False):
            score += 1
            throttle_reasons.append("memory_outbox_warn")

    if (
        "fracture_alert" in ext_flags
        or "drift_alert" in ext_flags
        or "hive_stress_alert" in ext_flags
        or "hive_crowding_alert" in ext_flags
        or "hive_entropy_alert" in ext_flags
        or "hive_turnover_alert" in ext_flags
        or "memory_turnover_alert" in ext_flags
        or "aion_outcome_alert" in ext_flags
    ):
        score += 1
        throttle_reasons.append("overlay_risk_flag_alert")

    if score >= 3:
        throttle_state = "alert"
    elif score >= 1:
        throttle_state = "warn"
    else:
        throttle_state = "normal"

    blocked_reasons = _uniq(blocked_reasons)
    throttle_reasons = _uniq(throttle_reasons)
    actions = _remediation_actions(blocked_reasons, throttle_reasons)
    return {
        "entry_blocked": bool(len(blocked_reasons) > 0),
        "entry_block_reasons": blocked_reasons,
        "throttle_state": throttle_state,
        "throttle_reasons": throttle_reasons,
        "recommended_actions": actions,
    }
