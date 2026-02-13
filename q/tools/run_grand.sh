#!/usr/bin/env bash
set -e
cd /Users/desmondpottle/Desktop/q_v2_5_foundations
source .venv/bin/activate
export PYTHONPATH="$PWD"

# 1) Rebuild sleeves, regime, and governors
python tools/build_min_sleeves.py
python tools/make_regime.py
python tools/apply_regime_governor.py
python tools/patch_regime_weights_with_dna.py
python tools/apply_regime_governor_dna.py

# 2) Rebuild Main portfolio + report
python tools/portfolio_from_runs_plus.py
python tools/build_report_plus.py
python tools/add_regime_final_card_triple.py || true

# 3) Vol-target (uses TARGET_ANN inside tools/portfolio_vol_target.py)
python tools/portfolio_vol_target.py
python tools/add_vol_target_card.py || true

# 4) Costs (Main/Regime/DNA + VT) and costs card
python tools/apply_costs.py
python tools/apply_costs_vt.py
python tools/add_costs_card_vt.py || true

# 5) Robustness + Stress cards
python tools/robust_time_splits.py
python tools/add_robust_card.py || true
python tools/add_stress_card.py || true

# 6) Scoreboard print + CSV
python tools/make_scoreboard.py

open report_best_plus.html
echo "✅ GRAND RUN DONE"
