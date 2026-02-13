#!/usr/bin/env python3
# tools/sweep_portfolio_knobs.py
# Tries a few (TOP_K, CAP_PER) combos by rewriting tools/portfolio_from_runs_plus.py,
# rebuilding portfolio and measuring OOS Sharpe. Saves runs_plus/sweep_portfolio_knobs.csv.

from pathlib import Path
import re
import pandas as pd
import numpy as np
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
PORT_BUILDER = ROOT / "tools" / "portfolio_from_runs_plus.py"

TOPK_SET = [20, 25, 30]
CAP_SET  = [0.10, 0.08, 0.06]

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

def read_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}

def metric_or(d, path, default=np.nan):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    try:
        return float(cur)
    except:
        return default

if __name__ == "__main__":
    # keep original file so we can restore at the end
    original_text = PORT_BUILDER.read_text()

    base = read_json(RUNS / "final_portfolio_summary.json")
    base_oos = metric_or(base, ["out_sample", "sharpe"], np.nan)

    rows = []
    try:
        for topk in TOPK_SET:
            for cap in CAP_SET:
                txt = original_text
                txt = re.sub(r"(TOP_K\s*=\s*)[0-9]+", rf"\g<1>{topk}", txt)
                txt = re.sub(r"(CAP_PER\s*=\s*)[0-9.]+", rf"\g<1>{cap}", txt)
                PORT_BUILDER.write_text(txt)

                # rebuild portfolio + report
                run(["python", "tools/portfolio_from_runs_plus.py"])
                run(["python", "tools/build_report_plus.py"])

                # rebuild regime + dna so the comparison is fair
                run(["python", "tools/make_regime.py"])
                run(["python", "tools/apply_regime_governor.py"])
                run(["python", "tools/patch_regime_weights_with_dna.py"])
                run(["python", "tools/apply_regime_governor_dna.py"])

                main = read_json(RUNS / "final_portfolio_summary.json")
                reg  = read_json(RUNS / "final_portfolio_regime_summary.json")
                dna  = read_json(RUNS / "final_portfolio_regime_dna_summary.json")

                rows.append(dict(
                    top_k=topk,
                    cap_per=cap,
                    base_oos=base_oos,
                    main_oos=metric_or(main, ["out_sample","sharpe"], np.nan),
                    reg_oos=metric_or(reg,  ["out_sample","sharpe"], np.nan),
                    dna_oos=metric_or(dna,  ["out_sample","sharpe"], np.nan),
                ))

        df = pd.DataFrame(rows).sort_values(["dna_oos","reg_oos","main_oos"], ascending=False)
        outp = RUNS / "sweep_portfolio_knobs.csv"
        df.to_csv(outp, index=False)
        print(df.to_string(index=False))
        print("Saved:", outp)
    finally:
        # always restore original file text
        PORT_BUILDER.write_text(original_text)
