#!/usr/bin/env bash
set -euo pipefail
mkdir -p data_new

# FRED
curl -L -o data_new/DGS2.csv "https://fred.stlouisfed.org/series/DGS2/downloaddata/DGS2.csv"
curl -L -o data_new/DGS5.csv "https://fred.stlouisfed.org/series/DGS5/downloaddata/DGS5.csv"
curl -L -o data_new/DGS30.csv "https://fred.stlouisfed.org/series/DGS30/downloaddata/DGS30.csv"
curl -L -o data_new/VIXCLS.csv "https://fred.stlouisfed.org/series/VIXCLS/downloaddata/VIXCLS.csv"
curl -L -o data_new/VIX9D.csv "https://fred.stlouisfed.org/series/VIX9D/downloaddata/VIX9D.csv"
curl -L -o data_new/VIX3M.csv "https://fred.stlouisfed.org/series/VIX3M/downloaddata/VIX3M.csv"
curl -L -o data_new/DCOILWTICO.csv "https://fred.stlouisfed.org/series/DCOILWTICO/downloaddata/DCOILWTICO.csv"
curl -L -o data_new/GOLDPMGBD228NLBM.csv "https://fred.stlouisfed.org/series/GOLDPMGBD228NLBM/downloaddata/GOLDPMGBD228NLBM.csv"
curl -L -o data_new/DEXUSEU.csv "https://fred.stlouisfed.org/series/DEXUSEU/downloaddata/DEXUSEU.csv"
curl -L -o data_new/DEXUSUK.csv "https://fred.stlouisfed.org/series/DEXUSUK/downloaddata/DEXUSUK.csv"
curl -L -o data_new/DEXJPUS.csv "https://fred.stlouisfed.org/series/DEXJPUS/downloaddata/DEXJPUS.csv"

# Stooq ETFs
curl -L -o data_new/GLD.csv "https://stooq.com/q/d/l/?s=gld.us&i=d"
curl -L -o data_new/TLT.csv "https://stooq.com/q/d/l/?s=tlt.us&i=d"
curl -L -o data_new/IEF.csv "https://stooq.com/q/d/l/?s=ief.us&i=d"
curl -L -o data_new/EEM.csv "https://stooq.com/q/d/l/?s=eem.us&i=d"
curl -L -o data_new/EFA.csv "https://stooq.com/q/d/l/?s=efa.us&i=d"

# Stooq sectors
curl -L -o data_new/XLK.csv "https://stooq.com/q/d/l/?s=xlk.us&i=d"
curl -L -o data_new/XLF.csv "https://stooq.com/q/d/l/?s=xlf.us&i=d"
curl -L -o data_new/XLE.csv "https://stooq.com/q/d/l/?s=xle.us&i=d"
curl -L -o data_new/XLV.csv "https://stooq.com/q/d/l/?s=xlv.us&i=d"
curl -L -o data_new/XLI.csv "https://stooq.com/q/d/l/?s=xli.us&i=d"
curl -L -o data_new/XLY.csv "https://stooq.com/q/d/l/?s=xly.us&i=d"
curl -L -o data_new/XLU.csv "https://stooq.com/q/d/l/?s=xlu.us&i=d"
curl -L -o data_new/XLB.csv "https://stooq.com/q/d/l/?s=xlb.us&i=d"

echo "OK"
