#!/usr/bin/env python3
"""
tune_meta.py

Sweeps META knobs and records WF+ averages:
  - META_STRENGTH: 0..1 sizing boost from Meta
  - META_SIGN_THRESH: required |meta| to use meta sign
  - META_REQUIRE_AGREE: 1=only when meta sign agrees with council, 0=meta can override

Writes:
  runs_plus/meta_tuning.csv
  runs_plus/meta_tuning_best.json
"""

import os, sys, json, subprocess, itertools
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNS  = ROOT / "runs_plus"

# Large-enough but still reasonable
META_STRENGTH      = [0.0, 0.3, 0.5, 0.8]
META_SIGN_THRESH   = [0.03, 0.05, 0.08, 0.10]
META_REQUIRE_AGREE = [1, 0]

def run(cmd, env=None):
    print("▶", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, env=env)

def parse_plus_csv():
    p = RUNS / "walk_forward_table_plus.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    def mean(col):
        s = pd.to_numeric(df.get(col), errors="coerce").dropna()
        return float(s.mean()) if len(s) else None
    return {"assets": len(df), "hit": mean("hit"), "sharpe": mean("sharpe"),
            "maxDD": mean("maxDD") if "maxDD" in df.columns else None}

def main():
    results = []
    # ensure artifacts
    run([sys.executable, str(TOOLS / "run_all_plus.py")])
    run([sys.executable, str(TOOLS / "make_hive_council.py")])
    run([sys.executable, str(TOOLS / "make_oos_all.py")])

    for ms, th, agree in itertools.product(META_STRENGTH, META_SIGN_THRESH, META_REQUIRE_AGREE):
        env = os.environ.copy()
        env["META_STRENGTH"] = str(ms)
        env["META_SIGN_THRESH"] = str(th)
        env["META_REQUIRE_AGREE"] = "1" if agree == 1 else "0"
        run([sys.executable, str(TOOLS / "walk_forward_plus.py")], env=env)
        m = parse_plus_csv() or {}
        results.append({
            "META_STRENGTH": ms,
            "META_SIGN_THRESH": th,
            "META_REQUIRE_AGREE": agree,
            "assets": m.get("assets"),
            "hit": m.get("hit"),
            "sharpe": m.get("sharpe"),
            "maxDD": m.get("maxDD"),
        })

    df = pd.DataFrame(results)
    out_csv = RUNS / "meta_tuning.csv"
    df.to_csv(out_csv, index=False)
    dff = df.dropna(subset=["sharpe"]).copy()
    dff["dd_score"] = dff["maxDD"]
    dff = dff.sort_values(["sharpe","dd_score","hit"], ascending=[False, False, False])
    best = dff.iloc[0].to_dict()
    (RUNS / "meta_tuning_best.json").write_text(json.dumps(best, indent=2))
    print("\nBEST META SETTINGS")
    print(json.dumps(best, indent=2))
    print(f"\n✅ Saved {out_csv.as_posix()} and meta_tuning_best.json")

if __name__ == "__main__":
    main()
