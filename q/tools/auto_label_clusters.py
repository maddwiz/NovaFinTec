#!/usr/bin/env python3
# Heuristic cluster labeler for your assets.
# Sources (in order):
#  1) runs_plus/asset_names.csv         (preferred)
#  2) data/*.csv + data_new/*.csv       (file stems)
#  3) runs_plus/portfolio_weights.csv   (fallback: A1..AN)
#
# Output:
#  runs_plus/cluster_map.csv  with columns: asset,cluster
#  Appends a report card showing cluster counts.

import csv, re
from pathlib import Path
import numpy as np
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)
DATA_DIRS = [ROOT/"data", ROOT/"data_new"]

# ---- keyword -> cluster rules (edit if you want finer buckets)
RULES = [
    # FX pairs
    (r"^(AUDUSD|EURUSD|GBPUSD|USDJPY|USDCAD|USDCHF|NZDUSD|EURJPY|GBPJPY)$", "FX"),
    (r"FX|FOREX|^DXY$", "FX"),

    # Rates / Treasuries / Yields
    (r"^(DGS\d+|^TNX$|^TYX$|^DTB3$|^DTB6$|^DGS3MO$)$", "Rates"),
    (r"(^IEF$|^TLT$|^SHY$|^IEI$|^ZB$|^ZN$|^ZF$|^UB$)", "Rates"),
    (r"^ED\d+|^SOFR|^FEDFUNDS$", "Rates"),

    # Credit (IG/HY)
    (r"(^LQD$|^IGLB$|IG)", "Credit"),
    (r"(^HYG$|^JNK$|HYG_TR|JNK_TR|^USHY$|^SJNK$)", "Credit"),

    # Equities (broad beta)
    (r"(^SPY$|^ES$|^MES$|^SPX$|^VOO$|^VTI$|^QQQ$|^NDX$|^NQ$|^MNQ$|^IWM$)", "Equities"),
    (r"(^EFA$|^EEM$|^VEA$|^VWO$|^ACWI$|^VXUS$)", "Equities"),
    (r"(XLF|XLE|XLK|XLY|XLI|XLB|XLU|XLV|XLP)", "Equities"),

    # Commodities (broad)
    (r"(^GLD$|^IAU$|^PHYS$|^SLV$|^AG$|^SIL$|^PPLT$|^CPER$|^JJC$)", "Commodities"),
    (r"(^USO$|^BNO$|^DBO$|^XOP$|^XLE$|CL|NG|RB|HO)", "Commodities"),
    (r"(^CORN$|^SOYB$|^WEAT$|^DBA$)", "Commodities"),
    (r"(^DBC$|^GSG$)", "Commodities"),

    # Crypto
    (r"(^BTC$|^BTCUSD$|^XBT$|^ETH$|^ETHUSD$|^WBTC$|^GBTC$|^ETHE$)", "Crypto"),

    # Vol / Tail
    (r"(^VIX$|^VXX$|^UVXY$|^SVIX$|^VVIX$|^MOVE$|VX\d*)", "Vol"),

    # Catch-alls for your earlier list
    (r"(^CPER$)", "Commodities"),
    (r"(^GLD$)", "Commodities"),
    (r"(^CORN$)", "Commodities"),
    (r"(^EEM$|^EFA$)", "Equities"),
    (r"(^HYG(_TR(_\d+)?)?$)", "Credit"),
    (r"(^IEF$)", "Rates"),
    (r"(^AUDUSD$|^EURUSD$|^GBPUSD$)", "FX"),
    (r"(^DGS\d+$|^DGS3MO$)", "Rates"),
]

def read_asset_names_csv(p: Path):
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt: return None
        # header-like single line
        if "," in txt and "\n" not in txt:
            cols = [c.strip() for c in txt.split(",") if c.strip()]
            return cols or None
        # else one per line (or single line with commas)
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if len(lines) == 1 and "," in lines[0]:
            cols = [c.strip() for c in lines[0].split(",") if c.strip()]
            return cols or None
        return lines or None
    except Exception:
        return None

def read_from_data_dirs():
    names = []
    for d in DATA_DIRS:
        if not d.exists(): continue
        for fp in sorted(d.glob("*.csv")):
            names.append(fp.stem)
    # unique, keep order
    out = []
    seen = set()
    for n in names:
        if n not in seen:
            out.append(n); seen.add(n)
    return out

def read_N_from_weights():
    for rel in ["runs_plus/portfolio_weights.csv", "portfolio_weights.csv"]:
        p = ROOT/rel
        if p.exists():
            try:
                a = np.loadtxt(p, delimiter=",")
            except Exception:
                try:
                    a = np.loadtxt(p, delimiter=",", skiprows=1)
                except Exception:
                    continue
            if a.ndim == 1: a = a.reshape(-1,1)
            N = int(a.shape[1])
            return [f"A{i+1}" for i in range(N)]
    return None

def guess_cluster(name: str) -> str:
    u = name.upper()
    for pat, lab in RULES:
        if re.search(pat, u):
            return lab
    return "Unknown"

def append_card(title, html):
    for nm in ["report_all.html","report_best_plus.html","report_plus.html","report.html"]:
        pg = ROOT/nm
        if not pg.exists(): continue
        txt = pg.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        pg.write_text(txt.replace("</body>", card+"</body>") if "</body>" in txt else txt+card, encoding="utf-8")

if __name__ == "__main__":
    # Find names
    names = read_asset_names_csv(RUNS/"asset_names.csv")
    if not names:
        names = read_from_data_dirs()
    if not names:
        names = read_N_from_weights()
    if not names:
        print("(!) Could not infer asset names. Add data/*.csv or runs_plus/asset_names.csv")
        raise SystemExit(0)

    rows = [(nm, guess_cluster(nm)) for nm in names]
    outp = RUNS/"cluster_map.csv"
    with outp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["asset","cluster"])
        w.writerows(rows)

    cnt = Counter([c for _, c in rows])
    bullets = "".join([f"<li>{k}: {v}</li>" for k, v in cnt.most_common()])
    append_card("Cluster Map (Auto) ✔", f"<p>Wrote cluster_map.csv with {len(rows)} assets.</p><ul>{bullets}</ul>")
    print(f"✅ Wrote {outp}  (rows={len(rows)})")
