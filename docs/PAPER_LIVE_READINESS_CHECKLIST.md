# Paper-Live Readiness Checklist

Scope: AION paper-live testing with live market data, simulated fills, full safety stack, and hard launch precheck.

## Q Validation Gates

- [x] Strict OOS validation passes (`q/runs_plus/strict_oos_validation.json`)
- [x] Cost-stress validation passes (`q/runs_plus/cost_stress_validation.json`)
- [x] Promotion gate passes (`q/runs_plus/q_promotion_gate.json`)
- [x] Health alerts have zero hard blockers (`q/runs_plus/health_alerts.json`)
- [x] External holdout is required and validated (not skipped)

## External Holdout Integrity

- [x] Added auto-builder: `q/tools/build_external_holdout_from_data.py`
- [x] Pipeline now builds holdout before validation (`q/tools/run_all_in_one_plus.py`)
- [x] Profiles enforce holdout requirement (`config/default.yaml`, `config/aggressive.yaml`, `config/conservative.yaml`)

## AION Execution Safety

- [x] Startup reconciliation in place
- [x] Kill switch watcher active
- [x] Pre-trade exposure gate active
- [x] Persistent order-state save/load active
- [x] Structured audit logging active
- [x] Canary timeout handling active
- [x] Paper-only safety mode in launcher

## Day-Skimmer Wiring

- [x] `paper_loop` dispatches `day_skimmer` mode to `skimmer_loop`
- [x] `skimmer_loop` uses BarEngine + SessionAnalyzer + Patterns + Confluence + IntradayRisk

## Launch Governance

- [x] Added hard readiness precheck script: `aion/tools/paper_live_readiness.py`
- [x] Launcher enforces precheck by default before trade loop (`aion/run_aion.sh`)
- [x] Precheck artifact written to `aion/state/paper_live_readiness.json`
- [x] Trade-loop singleton guard in launcher prevents duplicate `paper_loop` instances unless `AION_FORCE_RESTART_TRADE=1`
- [x] `ops_guard` now launches trade mode with explicit supervisor override to keep orchestration deterministic

## CI and Tests

- [x] CI workflow includes install/lint/type/tests/secrets
- [x] Local full test suite passes

## Optional Next Upgrade (now implemented, opt-in)

- [x] IB paper-account order routing path via `AION_EXECUTION_BACKEND=ib_paper` (trade loops keep simulator default; IB path is opt-in and guarded)
