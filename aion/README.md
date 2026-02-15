# AION (IBKR Paper) - God Mode Brain

AION now includes an advanced paper-trading stack with all 10 requested upgrades:

1. Portfolio optimizer (correlation/volatility-aware allocation)
2. Multi-timeframe confirmation (1H + 4H alignment)
3. Realistic execution simulator (latency, spread, queue impact, partial fills)
4. Event/news risk filter (session blackouts + manual event windows)
5. Meta-label model (probability gate on top of signal engine)
6. Strategy ensemble by regime (trend + mean-reversion + breakout + pattern)
7. Parameter stability testing in walk-forward backtests
8. Paper-to-live promotion gates
9. Real-time monitoring + alerts (confidence drift, slippage drift, drawdown spikes)
10. Continuous recalibration pipeline (backtest + report + tune + gate)

Plus:
- Degraded startup fallback (reuse last watchlist if IB preflight/scan fails)
- IB reconnect-resilient runtime loop (no hard crash on transient outages)
- Adaptive reason learning (per-pattern/per-indicator confidence multipliers from your closed-trade outcomes)

## Prereqs

- IB Gateway or TWS running and logged into PAPER
- API enabled and listening on your configured host/port
- Python with dependencies from `requirements.txt`

## Core run

```bash
cd "/Users/desmondpottle/Documents/New project/aion"
AION_MODE=brain AION_TASK=trade ./run_aion.sh
```

Preflight doctor runs automatically before trading (set `AION_SKIP_DOCTOR=1` to bypass).
`AION_DEGRADED_STARTUP=1` (default) keeps startup alive by falling back to the existing watchlist when needed.
IB connection attempts also scan `AION_IB_PORT_CANDIDATES` (default: `IB_PORT,4002,7497,7496`) and auto-use the first working endpoint.
`AION_AUTO_RESOLVE_IB_CONFLICT=1` (default) auto-resolves duplicate IB Gateway/TWS process conflicts before startup.
`AION_AUTO_RESTART_IB_ON_TIMEOUT=1` (default) can restart a non-responsive IB Gateway process and re-test connectivity.
`AION_IB_APP_PREFERRED` selects the .app used for restart fallback (default prefers `~/Applications/IB Gateway 10.43/IB Gateway 10.43.app` when present).
`AION_IB_APP_CANDIDATES` can define comma-separated IB .app fallback paths for auto-restart.
`AION_AUTO_IB_WARMUP=1` (default) waits up to `AION_IB_WARMUP_SECONDS` for API handshake readiness before startup proceeds.
Signal engine upgrades include live flag/triangle pattern detection, confluence voting, and regime-aware entry thresholds.
Pattern set now also includes live head-and-shoulders and inverse head-and-shoulders detection.

Useful tuning env vars:
- `AION_CONFLUENCE_LONG_MIN`, `AION_CONFLUENCE_SHORT_MIN`
- `AION_CONFLUENCE_BOOST_MAX`, `AION_CONFLUENCE_PENALTY_MAX`
- `AION_SIGNAL_MIN_MARGIN`
- `AION_VWAP_LEN` (rolling VWAP confirmation window)
- `AION_DIVERGENCE_LOOKBACK`, `AION_DIVERGENCE_PRICE_MOVE_MIN`, `AION_DIVERGENCE_RSI_DELTA_MIN`, `AION_DIVERGENCE_OBV_DELTA_MIN`
- `AION_REGIME_TH_SHIFT_TRENDING`, `AION_REGIME_TH_SHIFT_SQUEEZE`, `AION_REGIME_TH_SHIFT_CALM_RANGE`, `AION_REGIME_TH_SHIFT_HIGH_VOL_CHOP`
- `AION_REGIME_OPP_EXIT_SHIFT_TRENDING`, `AION_REGIME_OPP_EXIT_SHIFT_SQUEEZE`, `AION_REGIME_OPP_EXIT_SHIFT_CALM_RANGE`, `AION_REGIME_OPP_EXIT_SHIFT_HIGH_VOL_CHOP`
- `AION_REGIME_MARGIN_SHIFT_TRENDING`, `AION_REGIME_MARGIN_SHIFT_SQUEEZE`, `AION_REGIME_MARGIN_SHIFT_CALM_RANGE`, `AION_REGIME_MARGIN_SHIFT_HIGH_VOL_CHOP`
- External Q runtime-context scaling:
- `AION_EXT_RUNTIME_MIN_SCALE`, `AION_EXT_RUNTIME_MAX_SCALE`
- `AION_EXT_RUNTIME_DEGRADED_SCALE`, `AION_EXT_RUNTIME_QFAIL_SCALE`, `AION_EXT_RUNTIME_FLAG_SCALE`
- `AION_EXT_RUNTIME_DRIFT_WARN_SCALE`, `AION_EXT_RUNTIME_DRIFT_ALERT_SCALE`, `AION_EXT_RUNTIME_QUALITY_STEP_SPIKE_SCALE`
- `AION_EXT_RUNTIME_FRACTURE_WARN_SCALE`, `AION_EXT_RUNTIME_FRACTURE_ALERT_SCALE`
- `AION_EXT_RUNTIME_HIVE_WARN_SCALE`, `AION_EXT_RUNTIME_HIVE_ALERT_SCALE`
- `AION_EXT_RUNTIME_STALE_SCALE` (extra runtime de-risk when Q overlay file is stale)
- Runtime position-risk severity scalers:
- `AION_EXT_RUNTIME_RISK_DRIFT_WARN_SCALE`, `AION_EXT_RUNTIME_RISK_DRIFT_ALERT_SCALE`, `AION_EXT_RUNTIME_RISK_QUALITY_STEP_SCALE`
- `AION_EXT_RUNTIME_RISK_STALE_SCALE`
- External overlay doctor checks:
- `AION_EXT_SIGNAL_MAX_AGE_HOURS`, `AION_EXT_SIGNAL_REQUIRE_RUNTIME_CONTEXT`
- `AION_EXT_SIGNAL_CRITICAL` (fail Doctor if overlay check fails)
By default AION now reads Q overlay from `../q/runs_plus/q_signal_overlay.json`
(`AION_Q_HOME` / `AION_EXT_SIGNAL_FILE` can override this).
When Q runtime risk degrades, AION now automatically scales overlay impact and
reduces effective trades-per-day cap for safer operation.
Fracture flags now also tighten concurrent position caps (`max_open_positions`)
during `fracture_warn` / `fracture_alert` states.
`run_aion.sh` now auto-exports these defaults and prints the active overlay path at startup.
`AION_AUTO_REFRESH_Q_OVERLAY=1` (default) refreshes the Q overlay pack before trade-loop start.
Runtime log lines for external overlay now include Q `source_mode` for traceability.

NovaSpine memory bridge is now wired directly from AION trade events:
- `AION_MEMORY_ENABLE=1` (default via `run_aion.sh`)
- `AION_MEMORY_BACKEND=filesystem|novaspine_api`
- `AION_MEMORY_NOVASPINE_URL` (default `http://127.0.0.1:8420`)
- `AION_MEMORY_NAMESPACE` (default `private/nova/actions`)
- `AION_MEMORY_OUTBOX_DIR` (default `logs/novaspine_outbox_aion`)
- `AION_MEMORY_FAIL_COOLDOWN_SEC` (default `120`; circuit-breaker after API failure)
- `AION_MEMORY_TOKEN` (optional; falls back to `C3AE_API_TOKEN` / `C3_MEMORY_TOKEN`)

When backend is `filesystem`, each ENTRY/PARTIAL/EXIT is queued to local JSONL outbox.
When backend is `novaspine_api`, AION posts events to NovaSpine and safely falls back to outbox on API failure.

## Tasks

- Live paper loop:
```bash
AION_MODE=brain AION_TASK=trade ./run_aion.sh
```

- Walk-forward backtest + stability tests:
```bash
AION_MODE=brain AION_TASK=backtest ./run_aion.sh
```

- Performance report:
```bash
AION_MODE=brain AION_TASK=report ./run_aion.sh
```

- Adaptive tuning:
```bash
AION_MODE=brain AION_TASK=tune ./run_aion.sh
```

- Promotion gate (paper-to-live readiness):
```bash
AION_MODE=brain AION_TASK=gate ./run_aion.sh
```

- Full recalibration pipeline:
```bash
AION_MODE=brain AION_TASK=recalibrate ./run_aion.sh
```

- Environment and schema doctor checks:
```bash
AION_MODE=brain AION_TASK=doctor ./run_aion.sh
```

- Local dashboard:
```bash
AION_MODE=brain AION_TASK=dashboard ./run_aion.sh
```
Then open `http://127.0.0.1:8787` (or your configured `AION_DASHBOARD_HOST`/`AION_DASHBOARD_PORT`).
Dashboard now includes a `Q Overlay` health tile driven by Doctor checks.
Dashboard status payload now surfaces `external_fracture_state` derived from Q overlay risk flags.
Doctor remediation output now includes Q-overlay-specific recovery tips when stale/degraded.

- IB conflict recovery:
```bash
AION_MODE=brain AION_TASK=recover-ib ./run_aion.sh
```

- IB warmup probe:
```bash
AION_MODE=brain AION_TASK=trade AION_AUTO_IB_WARMUP=1 ./run_aion.sh
```

## Event blackout windows

Optional manual file:

- `state/event_blackouts.json`

Format:

```json
{
  "windows": [
    {"start": "2026-02-20T13:20:00", "end": "2026-02-20T14:10:00", "reason": "FOMC"},
    {"start": "2026-02-26T15:50:00", "end": "2026-02-26T16:20:00", "reason": "NVDA earnings"}
  ]
}
```

## Key outputs

- `state/watchlist.txt`
- `state/watchlist.json`
- `state/strategy_profile.json`
- `state/reason_feedback.json`
- `state/meta_model.json`
- `state/live_promotion.json`
- `state/recalibration_state.json`
- `logs/shadow_trades.csv`
- `logs/shadow_equity.csv`
- `logs/signals.csv`
- `logs/alerts.log`
- `logs/runtime_monitor.json`
- `logs/doctor_report.json`
- `logs/ib_recover_report.json`
- `logs/walkforward_results.json`
- `logs/performance_report.json`
- `logs/performance_report.md`

## Legacy runtime

```bash
AION_MODE=legacy AION_TASK=trade ./run_aion.sh
```
