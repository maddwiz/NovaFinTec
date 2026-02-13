#!/usr/bin/env python3
# tools/sweep_portfolio_fine.py
# Sweeps portfolio TOP_K around 30 and CAP_PER around 0.10–0.12.
# Writes results to runs_plus/sweep_portfolio_knobs.csv (same name as before).

from pathlib import Path
import re, json, subprocess, sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
PORT_BUILDER = ROOT / "tools" / "portfolio_from_runs_plus.py"

TOPK_SET = [28, 30, 32, 34]
CAP_SET  = [0.10, 0.11, 0.12]

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

def read_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}

def metric_or(d, path, default=np.nan):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur: cur = cur[k]
        else: return default
    try: return float(cur)
    except: return default

if __name__ == "__main__":
    original = PORT_BUILDER.read_text()
    rows = []
    try:
        for topk in TOPK_SET:
            for cap in CAP_SET:
                txt = re.sub(r"(TOP_K\s*=\s*)[0-9]+", rf"\g<1>{topk}", original)
                txt = re.sub(r"(CAP_PER\s*=\s*)[0-9.]+", rf"\g<1>{cap}", txt)
                PORT_BUILDER.write_text(txt)

                run(["python", "tools/portfolio_from_runs_plus.py"])
                run(["python", "tools/build_report_plus.py"])
                run(["python", "tools/make_regime.py"])
                run(["python", "tools/apply_regime_governor.py"])
                run(["python", "tools/patch_regime_weights_with_dna.py"])
                run(["python", "tools/apply_regime_governor_dna.py"])

                main = read_json(RUNS/"final_portfolio_summary.json")
                reg  = read_json(RUNS/"final_portfolio_regime_summary.json")
                dna  = read_json(RUNS/"final_portfolio_regime_dna_summary.json")
                rows.append(dict(
                    top_k=topk, cap_per=cap,
                    main_oos=metric_or(main, ["out_sample","sharpe"]),
                    reg_oos =metric_or(reg,  ["out_sample","sharpe"]),
                    dna_oos =metric_or(dna,  ["out_sample","sharpe"]),
                    main_dd =metric_or(main, ["out_sample","maxdd"]),
                    dna_dd  =metric_or(dna,  ["out_sample","maxdd"]),
                ))
        df = pd.DataFrame(rows).sort_values(["dna_oos","main_oos"], ascending=False)
        outp = RUNS/"sweep_portfolio_knobs.csv"
        df.to_csv(outp, index=False)
        print(df.head(12).to_string(index=False))
        print("Saved:", outp)
    finally:
        PORT_BUILDER.write_text(original)
