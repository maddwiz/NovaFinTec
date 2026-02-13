#!/usr/bin/env bash
set -euo pipefail
mkdir -p data_new
# ETF proxies for commodities (Stooq, daily)
curl -L -o data_new/USO.csv  "https://stooq.com/q/d/l/?s=uso.us&i=d"
curl -L -o data_new/UNG.csv  "https://stooq.com/q/d/l/?s=ung.us&i=d"
curl -L -o data_new/JJC.csv  "https://stooq.com/q/d/l/?s=jjc.us&i=d"
curl -L -o data_new/WEAT.csv "https://stooq.com/q/d/l/?s=weat.us&i=d"
curl -L -o data_new/CORN.csv "https://stooq.com/q/d/l/?s=corn.us&i=d"
echo "OK"
