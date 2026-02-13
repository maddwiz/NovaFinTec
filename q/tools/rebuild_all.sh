#!/bin/bash
set -euo pipefail
shopt -s nullglob

# 0) discover actual CSVs present
CSVFILES=(data/*.csv)
if [ ${#CSVFILES[@]} -eq 0 ]; then
  echo "No CSVs found in ./data"; exit 1
fi

ASSETS=()
for f in "${CSVFILES[@]}"; do
  bn="$(basename "$f")"
  ASSETS+=( "${bn%.csv}" )
done
echo "Assets: ${ASSETS[*]}"

# 1) prune old runs whose CSV no longer exists (e.g., leftover runs_plus/VIX/)
if [ -d runs_plus ]; then
  for d in runs_plus/*; do
    [ -d "$d" ] || continue
    a="$(basename "$d")"
    if [ ! -f "data/$a.csv" ]; then
      echo "Prune stale run: $a (no matching data/$a.csv)"
      rm -rf "$d"
    fi
  done
fi

# 2) run per-asset pipelines (skip errors; keep going)
for A in "${ASSETS[@]}"; do
  echo ">>> Running pipeline for $A"
  mkdir -p "runs_plus/$A"
  if ! python run_pipeline_cv_plus.py --data ./data --asset "$A.csv" --out "runs_plus/$A" --cost_bps 1.0 --frames 80 | tee "runs_plus/$A/stdout.log"; then
    echo "!! Skipping $A (pipeline error)"; continue
  fi
done

# 3) overlays (best-effort)
python tools/make_overlays_auto.py || echo "Overlays step skipped."

# 4) portfolio (best-effort)
python tools/portfolio_auto.py || echo "Portfolio step skipped."

# 5) family graph + heatmap
python - <<'PY'
from pathlib import Path
from qmods.family_graph import build_family_graph
data = Path("data")
assets = [p.stem for p in data.glob("*.csv")]
build_family_graph(data, assets, step_days=63, min_window=256, out_path=Path("runs_plus/family_graph.json"))
print("wrote runs_plus/family_graph.json")
PY
python tools/family_to_png.py || true

# 6) news (optional)
python tools/ingest_news.py || true

# 7) report (auto-open inside)
python tools/build_report_plus.py || true
open report_plus.html 2>/dev/null || true
