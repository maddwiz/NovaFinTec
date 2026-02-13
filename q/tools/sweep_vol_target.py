#!/usr/bin/env python3
# tools/sweep_vol_target.py
# Tries a few TARGET_ANN values and picks the best OOS Sharpe.

from pathlib import Path
import re, subprocess, sys, pandas as pd, json, numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"
SCRIPT = ROOT/"tools"/"portfolio_vol_target.py"
TARGETS = [0.08, 0.10, 0.12]  # 8%, 10%, 12%

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0: sys.exit(r.returncode)

def read_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}

if __name__ == "__main__":
    original = SCRIPT.read_text()
    rows = []
    try:
        for t in TARGETS:
            txt = re.sub(r"(TARGET_ANN\s*=\s*)[0-9.]+", rf"\g<1>{t}", original)
            SCRIPT.write_text(txt)
            run(["python", "tools/portfolio_vol_target.py"])
            summ = read_json(RUNS/"final_portfolio_vt_summary.json")
            rows.append(dict(target=t,
                             vt_oos=summ.get("out_sample",{}).get("sharpe",np.nan),
                             vt_dd=summ.get("out_sample",{}).get("maxdd",np.nan)))
        df = pd.DataFrame(rows).sort_values("vt_oos", ascending=False)
        outp = RUNS/"sweep_vol_target.csv"
        df.to_csv(outp, index=False)
        print(df.to_string(index=False))
        print("Saved:", outp)
    finally:
        SCRIPT.write_text(original)
