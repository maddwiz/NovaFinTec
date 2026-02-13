#!/usr/bin/env bash
set -e
mkdir -p runs/IWM runs/RSP runs/LQD_TR runs/HYG_TR

python scripts/run_walkforward_meta.py --data ./data --asset IWM.csv    --out runs/IWM     | tee runs/IWM/stdout.log
python scripts/run_walkforward_meta.py --data ./data --asset RSP.csv    --out runs/RSP     | tee runs/RSP/stdout.log
python scripts/run_walkforward_meta.py --data ./data --asset LQD_TR.csv --out runs/LQD_TR  | tee runs/LQD_TR/stdout.log
python scripts/run_walkforward_meta.py --data ./data --asset HYG_TR.csv --out runs/HYG_TR  | tee runs/HYG_TR/stdout.log
