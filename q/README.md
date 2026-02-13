
# Q v2.5 — Foundations (Runnable)

This is a clean, runnable backbone of Q that you can use **today** to backtest on your CSVs and generate simple dreams. It includes:
- DNA compression + drift (+ drift velocity)
- Signals: DNA drift, Trend (EMA), Momentum
- Simple ensemble council (weighted sign)
- Risk: vol targeting, **max allocation cap 25%**, drawdown brake, flip/entropy budget
- Crisis anchors (VIX-based if provided, or internal drift shock)
- Rolling **walk-forward** backtest (no leakage)
- Explainability cards (per trade)
- Dream image generator (deterministic)
- CLI scripts

## 5-Year-Old Mode: How to Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your CSVs in `./data` (e.g., `SPY.csv`, `VIXCLS.csv`).  
Run:
```bash
python scripts/run_walkforward.py --data ./data --asset SPY.csv --vix VIXCLS.csv --out ./runs/spy_demo
python scripts/make_dreams.py --data ./data --asset SPY.csv --out ./runs/spy_demo
python scripts/make_report.py --run ./runs/spy_demo --out ./runs/spy_demo
```

---

## Roadmap (next coding steps)
- Bandit-tuned council weights
- Anomaly triage + outlier quarantine
- HTML explainability cards with charts
- DVQ tokens
- Provenance ledger


---

## New in v2.5-upd
- **Bandit-tuned council weights** (fit on train → frozen on test)
- **Anomaly triage + outlier quarantine** to skip likely-glitch days
- **HTML explainability** with charts (`scripts/make_report_html.py`)

### Example
```bash
python scripts/run_walkforward.py --data ./data --asset SPY.csv --vix VIXCLS.csv --out ./runs/spy_demo --eta 0.4
python scripts/make_report_html.py --run ./runs/spy_demo --out ./runs/spy_demo
open runs/spy_demo/report.html
```

## Current One-Command Pipeline (v2.5+)
Run the integrated pipeline with all major layers:
```bash
python tools/run_all_in_one_plus.py
```
This now includes:
- symbolic + heartbeat + reflexive generation
- hive build + hive walk-forward diagnostics
- guardrails, governors, councils, meta/synapses, cross-hive, ecosystem
- reliability-aware quality governor (nested WF + hive WF + council diagnostics)
- NovaSpine recall feedback loop (augment -> context boost -> final risk scaling)
- immune drill governance check (synthetic stress scenarios)
- final portfolio assembly + system health + alert gate
- optional NovaSpine bridge (cold/meta memory sync, async-safe)

### Strict production cycle
```bash
./tools/run_prod_cycle.sh
```
This runs strict mode (`Q_STRICT=1`) and fails if critical health alerts trigger.
If your default Python is missing deps, set `Q_PYTHON=/absolute/path/to/venv/bin/python`.

### Health artifacts
- `runs_plus/system_health.json`
- `runs_plus/health_alerts.json`
- `runs_plus/pipeline_status.json`
- `runs_plus/quality_snapshot.json`
- `runs_plus/execution_constraints_info.json`
- `runs_plus/novaspine_sync_status.json`
- `runs_plus/novaspine_last_batch.json`
- `runs_plus/novaspine_context.json`
- `runs_plus/novaspine_context_boost.csv`
- `runs_plus/immune_drill.json`
- `runs_plus/novaspine_replay_status.json`

### Tests
```bash
python -m pytest -q tests
```
GitHub Actions CI is configured under `.github/workflows/ci.yml`.

### Importing large historical CSV sets
If you have many raw CSVs, normalize them into `data/` first:
```bash
python tools/import_history_csvs.py --src "/absolute/path/to/your/csv_folder"
python tools/make_asset_names.py
```

### Execution constraints (live realism)
You can define live execution limits in `config/execution_constraints.json`
(copy from `config/execution_constraints.example.json`), then run:
```bash
python tools/run_execution_constraints.py --replace-final
```

### NovaSpine bridge (optional, recommended for cold/meta memory)
Default is disabled (no runtime risk). Enable with env vars:
```bash
export C3_MEMORY_ENABLE=1
export C3_MEMORY_BACKEND=novaspine_api
export C3_MEMORY_NOVASPINE_URL=http://127.0.0.1:8420
# if NovaSpine auth is enabled:
# export C3AE_API_TOKEN=your_token
# recall feedback loop controls:
# export C3_MEMORY_RECALL_ENABLE=1
# export C3_MEMORY_RECALL_TOPK=6
# export C3_MEMORY_RECALL_MIN_SCORE=0.005
# export C3_MEMORY_RECALL_INCLUDE_ALERTS=0
# immune drill alert gate:
# export Q_REQUIRE_IMMUNE_PASS=1
# optional:
# export C3_MEMORY_NAMESPACE=private/nova/actions
# export C3_MEMORY_DIR=/absolute/path/to/outbox
# export C3_MEMORY_BACKEND=http
# export C3_MEMORY_HTTP_URL=https://your-endpoint.example/v1/events
# export C3_MEMORY_TOKEN=your_token
python tools/sync_novaspine_memory.py
```
The all-in-one pipeline also runs this bridge automatically.
It also replays queued outbox batches with:
```bash
python tools/replay_novaspine_outbox.py
```
