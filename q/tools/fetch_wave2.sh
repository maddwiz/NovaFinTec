#!/usr/bin/env bash
set -euo pipefail
mkdir -p data_new
curl -L -o data_new/EURUSD.csv "https://stooq.com/q/d/l/?s=eurusd&d1=20000101&d2=20251231&i=d"
curl -L -o data_new/JPYUSD.csv "https://stooq.com/q/d/l/?s=jpyusd&d1=20000101&d2=20251231&i=d"
curl -L -o data_new/GBPUSD.csv "https://stooq.com/q/d/l/?s=gbpusd&d1=20000101&d2=20251231&i=d"
curl -L -o data_new/AUDUSD.csv "https://stooq.com/q/d/l/?s=audusd&d1=20000101&d2=20251231&i=d"
curl -L -o data_new/WTI.csv    "https://stooq.com/q/d/l/?s=cl.f&d1=20000101&d2=20251231&i=d"
curl -L -o data_new/NATGAS.csv "https://stooq.com/q/d/l/?s=ng.f&d1=20000101&d2=20251231&i=d"
curl -L -o data_new/COPPER.csv "https://stooq.com/q/d/l/?s=hg.f&d1=20000101&d2=20251231&i=d"
curl -L -o data_new/WHEAT.csv  "https://stooq.com/q/d/l/?s=zw.f&d1=20000101&d2=20251231&i=d"
curl -L -o data_new/CORN.csv   "https://stooq.com/q/d/l/?s=zc.f&d1=20000101&d2=20251231&i=d"
curl -L -o data_new/GOLDPM.csv "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GOLDPMGBD228NLBM"
curl -L -o data_new/DGS2.csv   "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2"
curl -L -o data_new/DGS5.csv   "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS5"
curl -L -o data_new/DGS30.csv  "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS30"
echo "OK"
