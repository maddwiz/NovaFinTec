
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
