# NovaFinTec Master Handoff

Last updated (UTC): 2026-02-18
Primary repo root: `.`

## 1) What NovaFinTec Is
NovaFinTec is a two-layer quant stack:
- `q/` = research/portfolio brain (walk-forward, councils, governors, overlays, holdout validation, portfolio assembly).
- `aion/` = execution/runtime brain (IBKR paper/live plumbing, intraday gating, policy controls, dashboard, monitoring).

NovaSpine integration is wired as optional but active memory/context feedback through `q/tools/sync_novaspine_memory.py`, `q/tools/run_novaspine_context.py`, `q/tools/run_novaspine_hive_feedback.py`, and AION bridge readers.

## 2) Current Production State (as of this handoff)
Implemented and validated:
- Intraday gate telemetry in AION signals/dashboard.
- Canonical decision telemetry + summary refresh in long-term `paper_loop.py` (`trade_decisions.jsonl`, `telemetry_summary.json`).
- Cross-sectional momentum overlay in Q runtime stack.
- External untouched holdout protocol (`run_external_holdout_validation.py`) integrated into pipeline.
- Complexity penalties in search/tuning (`run_runtime_combo_search.py`, `run_governor_param_sweep.py`).
- Sharpe consistency patch in governor sweep metrics (`run_governor_param_sweep.py` now uses sample std `ddof=1`).
- Opt-in governor walk-forward validator (`run_governor_walkforward.py`) with expanding folds and OOS aggregation.
- Class-aware execution friction in daily cost engine (`make_daily_from_weights.py`).
- Local multi-asset bundle ingestion (`ingest_multi_asset_csv_bundle.py`).

Validation status:
- Latest full local run (`q/tests` + `aion/tests`): `606 passed`.
- Latest targeted run (new governor walk-forward batch): `24 passed`.
- Latest AION suite run after telemetry wiring: `216 passed`.
- GitHub Actions workflow: `.github/workflows/ci.yml`.

## 3) Top-Level File Map
- `.github/workflows/ci.yml`
- `README.md`
- `LICENSE`
- `config/default.yaml`
- `config/aggressive.yaml`
- `config/conservative.yaml`
- `launch_q_aion.sh`
- `q/` (research + portfolio layer)
- `aion/` (execution + live runtime layer)
- `NovaSpine/` (external memory system repo clone)
- `results/` (published snapshot artifacts)

## 4) Q File Map (Core)
### `q/qengine`
- `data.py` data loading/alignment
- `signals.py` base signal construction
- `walkforward.py` walk-forward evaluation primitives
- `risk.py` metrics/risk functions
- `meta.py` macro/meta helpers
- `dna.py`, `dreams.py` experimental regime/latent modules
- `explain.py`, `provenance.py`, `utils.py`

### `q/qmods`
Key modules:
- Councils/meta: `council.py`, `council_train.py`, `meta_council.py`, `meta_stack_v1.py`, `synapses_small.py`
- Risk/governors: `guardrails_bundle.py`, `drawdown_floor.py`, `quality_governor.py`, `regime_fracture.py`, `dream_coherence.py`, `symbolic_governor.py`, `dna_governor.py`
- Hive/ecosystem: `hive.py`, `cross_hive_arb_v1.py`, `hive_transparency.py`, `ecosystem_age.py`
- Memory: `novaspine_adapter.py`, `novaspine_context.py`, `novaspine_hive.py`, `aion_feedback.py`
- Overlays/sleeves: `tail_blender_v1.py`, `risk_parity_sleeve.py`, `adaptive_caps.py`, `feature_neutralizer.py`

### `q/tools` Critical Runtime Scripts
Pipeline orchestration:
- `run_all_in_one_plus.py` (main orchestrator)
- `run_prod_cycle.sh` (strict production cycle)

Portfolio assembly and returns:
- `build_final_portfolio.py`
- `make_daily_from_weights.py`
- `run_execution_constraints.py`

Validation/gates:
- `run_strict_oos_validation.py`
- `run_cost_stress_validation.py`
- `run_external_holdout_validation.py` (new)
- `run_q_promotion_gate.py`

Search/tuning:
- `run_runtime_combo_search.py` (now with complexity penalty)
- `run_governor_param_sweep.py` (now with complexity penalty)
- `run_governor_walkforward.py` (opt-in expanding walk-forward for governor params)
- `tune_legacy_knobs.py`, `tune_micro_all.py`

Alpha overlays:
- `run_credit_leadlag_signal.py`
- `run_cross_sectional_momentum_overlay.py`
- `run_microstructure_proxy_signal.py`
- `run_calendar_event_overlay.py`
- `run_macro_proxy_guard.py`

Data ingestion/expansion:
- `ingest_multi_asset_csv_bundle.py` (new)
- `normalize_new_csvs.py`, `fix_dates_in_data.py`, `fix_close_in_data.py`
- `run_asset_class_diversification.py`

AION bridge and memory:
- `export_aion_signal_pack.py`
- `run_q_aion_integrated_backtest.py`
- `sync_novaspine_memory.py`, `replay_novaspine_outbox.py`

## 5) AION File Map (Core)
### `aion/aion/brain`
- `signals.py` confluence scoring + intraday alignment gate
- `indicators.py` pattern/indicator library
- `external_signals.py` Q overlay ingestion
- `novaspine_bridge.py`

### `aion/aion/exec`
- `paper_loop.py` main loop, signal gating, trade lifecycle
- `dashboard.py` live dashboard + API
- `doctor.py` runtime health checks
- `promotion_gate.py`, `runtime_decision.py`, `runtime_health.py`
- `ops_guard.py`, `operator.py`, `ib_recover.py`, `ib_wait_ready.py`

### `aion/aion/execution`
- `simulator.py` slippage + partial fill model

### `aion/aion/risk`
- `policy.py`, `kill_switch.py`, `event_filter.py`, `position_sizing.py`

### `aion/aion/data`
- `ib_client.py` IBKR data/API integration

### `aion/aion/utils`
- `logging_utils.py` logs/trades/signals schema

## 6) End-to-End Wiring
1. Q builds features/signals/councils/hive/governors.
2. Q assembles final weights (`runs_plus/portfolio_weights_final.csv`).
3. Q applies cost model (`runs_plus/daily_returns.csv`) with class-aware friction.
4. Q runs strict OOS, cost stress, external holdout, and promotion gate.
5. Q exports overlay pack to AION (`q_signal_overlay.json`, mirrored to `aion/state/`).
6. AION reads overlay/runtime context and applies execution/policy/intraday gates.
7. AION writes execution outcomes (`shadow_trades.csv`, monitor JSON).
8. Q calibrates friction + outcome feedback from AION logs and loops.
9. NovaSpine sync/replay adds memory context for future scaling.

## 7) Key Artifacts to Watch
Q artifacts (`q/runs_plus`):
- `portfolio_weights_final.csv` final policy weights
- `daily_returns.csv` costed returns baseline
- `strict_oos_validation.json`
- `cost_stress_validation.json`
- `external_holdout_validation.json` (new)
- `q_promotion_gate.json`
- `runtime_combo_search.json`
- `governor_params_profile.json`
- `governor_walkforward_metrics.json`
- `final_governor_trace.csv`
- `q_signal_overlay.json`

AION artifacts (`aion/logs`, `aion/state`):
- `shadow_trades.csv`
- `signals.csv` (includes `intraday_score`, `intraday_gate`, `mtf_gate`, `meta_gate`)
- `runtime_monitor.json`
- `doctor_report.json`
- `performance_report.json`
- `state/q_signal_overlay.json` (mirror from Q)

Published artifacts (`results/`):
- `walkforward_metrics.json`
- `walkforward_equity.csv`
- `governor_compound_summary.json`
- `README.md`

## 8) Runbooks
### Q full cycle
```bash
cd q
../.venv/bin/python tools/run_all_in_one_plus.py
```

### Strict production cycle
```bash
cd q
./tools/run_prod_cycle.sh
```

### AION paper trade + dashboard
```bash
cd aion
nohup env AION_TASK=trade ./run_aion.sh > logs/live_trade.out 2>&1 &
nohup env AION_TASK=dashboard ./run_aion.sh > logs/dashboard.out 2>&1 &
```

### Multi-asset bundle ingestion (new)
```bash
cd q
export Q_MULTI_ASSET_SOURCE_DIR="/absolute/path/to/csv_bundle"
../.venv/bin/python tools/ingest_multi_asset_csv_bundle.py
```

### External untouched holdout validation (new)
```bash
cd q
../.venv/bin/python tools/run_external_holdout_validation.py
```
Optional direct returns file:
```bash
export Q_EXTERNAL_HOLDOUT_RETURNS_FILE="/absolute/path/holdout_returns.csv"
```

## 9) Known Failure Modes / What to Watch Out For
- Promotion gate fails due hit-rate/cost stress despite high robust Sharpe.
- `q_signal_overlay.json` degraded-safe-mode when gate fails.
- Class map mismatch (`cluster_map.csv` vs asset matrix width) causing weak caps.
- Over-throttling from stacked governors (check `final_governor_trace.csv` and runtime floor).
- IBKR session disconnects/autologoff impacting AION run continuity.

## 10) Roadmap to Shippable Product
### Phase A: Validation hardening
- Enforce external holdout in production gate (`Q_PROMOTION_REQUIRE_EXTERNAL_HOLDOUT=1`).
- Add dedicated holdout dataset governance (frozen time windows + immutable manifests).
- Add pass/fail alerting on holdout drift over time.

### Phase B: Execution realism and capacity
- Expand class-level cost model to include volume/ADV impacts per class.
- Add execution schedule simulator (TWAP/VWAP buckets) in Q integrated backtests.
- Add capacity envelope report (AUM vs decay) per strategy sleeve.

### Phase C: Productization
- Unified ops command that starts Q cycle + AION runtime + health checks.
- Persistent process manager (launchd/systemd/supervisor) for always-on operation.
- User-facing control plane for mode/policy/risk caps and one-click diagnostics.

### Phase D: Commercial readiness
- Immutable run ledger (configs/artifacts/checksums per run).
- Audit-ready model card and risk disclosures.
- SLA-style runtime alerts and recovery automation.

## 11) Handoff Maintenance Contract (MANDATORY)
Every agent working this repo must:
1. Update this file after any non-trivial architecture/runtime change.
2. Update `results/` snapshot after material strategy/governor changes.
3. Include changed artifacts + validation status in commit message or PR notes.
4. Never leave runtime knobs undocumented when added.

Recommended cadence:
- Minor work: update sections 2, 7, 9.
- Major work: update sections 3–11 and refresh roadmap priorities.

## 12) Quick Commands for Future Agents
- Full tests:
```bash
PYTHONPATH="$(pwd)/q:$(pwd)/aion" .venv/bin/python -m pytest -q q/tests aion/tests/test_external_signals_runtime.py aion/tests/test_doctor_external_overlay.py aion/tests/test_dashboard_status.py aion/tests/test_config_external_overlay_path.py aion/tests/test_novaspine_bridge.py aion/tests/test_runtime_risk_caps.py aion/tests/test_risk_policy.py aion/tests/test_ops_guard.py aion/tests/test_promotion_gate.py aion/tests/test_runtime_decision.py aion/tests/test_operator.py aion/tests/test_ib_client.py aion/tests/test_watchlist.py aion/tests/test_backtest_skimmer.py aion/tests/test_telemetry_summary.py aion/tests/test_paper_loop_day_skimmer_dispatch.py aion/tests/test_skimmer_telemetry.py aion/tests/test_paper_loop_telemetry.py
```
- Compile check:
```bash
.venv/bin/python -m compileall -q q/qengine q/qmods q/tools aion/aion
```
- Publish snapshot:
```bash
cd q && ../.venv/bin/python tools/publish_results_snapshot.py
```
