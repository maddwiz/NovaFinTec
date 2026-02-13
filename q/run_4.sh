#!/usr/bin/env bash
set -e
for a in IWM.csv RSP.csv LQD_TR.csv HYG_TR.csv; do
  out="runs/${a%.csv}_qoverride2"
  mkdir -p "$out"
  echo "== RUN $a (qoverride2) =="
  python scripts/run_walkforward_meta.py --data ./data --asset "$a" --out "$out" --eta 0.7 | tee "$out/stdout.log"
  python tools/save_metrics.py "$out"
done
