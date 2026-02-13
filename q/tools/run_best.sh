#!/usr/bin/env bash
set -e
cd /Users/desmondpottle/Desktop/q_v2_5_foundations
source .venv/bin/activate
export PYTHONPATH="$PWD"

# Rebuild sleeves & regime
python tools/build_min_sleeves.py
python tools/make_regime.py
python tools/apply_regime_governor.py
python tools/patch_regime_weights_with_dna.py
python tools/apply_regime_governor_dna.py

# Rebuild main portfolio & report
python tools/portfolio_from_runs_plus.py
python tools/build_report_plus.py
python tools/add_regime_final_card_triple.py
python tools/add_robust_card.py || true
python tools/add_vol_target_card.py || true

# Vol-targeted version (uses TARGET_ANN set in portfolio_vol_target.py)
python tools/portfolio_vol_target.py
python tools/add_vol_target_card.py

open report_best_plus.html
echo "✅ DONE (Main + Regime + DNA + VolTarget card)"
