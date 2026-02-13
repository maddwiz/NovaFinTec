#!/usr/bin/env python3
"""
eval_addons_plus.py

Runs an ablation over add-on combos using WF+ (flag-aware),
prints a matrix, saves runs_plus/addon_eval_plus.csv,
chooses the best, and writes report_best_plus.html.
"""

import os, sys, json, shutil, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNS  = ROOT / "runs_plus"

CONFIGS = [
    ("baseline (Council only)",             {"dna":False, "heartbeat":False, "symbolic":False, "reflexive":False, "hive":False}),
    ("+ DNA",                               {"dna":True,  "heartbeat":False, "symbolic":False, "reflexive":False, "hive":False}),
    ("+ Heartbeat",                         {"dna":True,  "heartbeat":True,  "symbolic":False, "reflexive":False, "hive":False}),
    ("+ Symbolic",                          {"dna":True,  "heartbeat":True,  "symbolic":True,  "reflexive":False, "hive":False}),
    ("+ Reflexive",                         {"dna":True,  "heartbeat":True,  "symbolic":True,  "reflexive":True,  "hive":False}),
    ("+ Hive (all current)",                {"dna":True,  "heartbeat":True,  "symbolic":True,  "reflexive":True,  "hive":True}),
]

def run(cmd, env=None):
    print("▶", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, env=env)

def flags_env(toggles: dict):
    env = os.environ.copy()
    env["SKIP_DNA"]       = "0" if toggles.get("dna")       else "1"
    env["SKIP_HEARTBEAT"] = "0" if toggles.get("heartbeat") else "1"
    env["SKIP_SYMBOLIC"]  = "0" if toggles.get("symbolic")  else "1"
    env["SKIP_REFLEXIVE"] = "0" if toggles.get("reflexive") else "1"
    env["SKIP_HIVE"]      = "0" if toggles.get("hive")      else "1"
    return env

def parse_plus_csv():
    p = RUNS / "walk_forward_table_plus.csv"
    if not p.exists(): return None
    try: df = pd.read_csv(p)
    except Exception: return None
    def mean(col):
        if col not in df.columns: return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.mean()) if len(s) else None
    return {
        "assets": len(df),
        "hit_avg": mean("hit"),
        "sharpe_avg": mean("sharpe"),
        "maxdd_avg": mean("maxDD") if "maxDD" in df.columns else mean("maxdd"),
    }

def print_table(title, rows, header):
    print("\n" + title)
    print("=" * len(title))
    if not rows:
        print("(no data)"); return
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i,h in enumerate(header)]
    line = " | ".join(str(h).ljust(widths[i]) for i,h in enumerate(header))
    print(line); print("-"*len(line))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(header))))

def main():
    results = []
    # make sure oos.csv exist (idempotent)
    run([sys.executable, str(TOOLS / "make_oos_all.py")])

    for label, toggles in CONFIGS:
        print(f"\n=== Testing (WF+): {label} ===")
        env = flags_env(toggles)
        # build artifacts obeying SKIP_* (for report cards)
        run([sys.executable, str(TOOLS / "run_all_plus.py")], env=env)
        # recompute WF+ under same flags
        run([sys.executable, str(TOOLS / "walk_forward_plus.py")], env=env)
        m = parse_plus_csv() or {"assets":None,"hit_avg":None,"sharpe_avg":None,"maxdd_avg":None}
        results.append({"config": label, **m})

    # matrix
    rows = [[r["config"],
             r["assets"] if r["assets"] is not None else "—",
             f'{r["hit_avg"]:.3f}'    if r["hit_avg"]    is not None else "—",
             f'{r["sharpe_avg"]:.3f}' if r["sharpe_avg"] is not None else "—",
             f'{r["maxdd_avg"]:.3f}'  if r["maxdd_avg"]  is not None else "—"]
            for r in results]
    print_table("ADD-ON ABLATION MATRIX (WF+)", rows,
                ["Config","Assets","Hit(avg)","Sharpe(avg)","MaxDD(avg)"])

    # choose best: higher Sharpe, then higher Hit, then higher MaxDD (less negative)
    def key(r):
        def z(x): return -1e9 if x is None else x
        return (z(r["sharpe_avg"]), z(r["hit_avg"]), z(r["maxdd_avg"]))
    best = max(results, key=key)
    base = results[0]

    # deltas
    def d(a,b): 
        return None if (a is None or b is None) else a-b
    print("\nRECOMMENDATION (WF+)")
    print("====================")
    print(f"Best: {best['config']}")
    dh, ds, dd = d(best["hit_avg"], base["hit_avg"]), d(best["sharpe_avg"], base["sharpe_avg"]), d(best["maxdd_avg"], base["maxdd_avg"])
    if ds is not None: print(f"  ΔSharpe(avg): {ds:+.3f}")
    if dh is not None: print(f"  ΔHit(avg):    {dh:+.3f}")
    if dd is not None: print(f"  ΔMaxDD(avg):  {dd:+.3f}  (less negative is better)")

    # save csv + choice
    pd.DataFrame(results).to_csv(RUNS / "addon_eval_plus.csv", index=False)
    (RUNS / "addon_choice_plus.json").write_text(json.dumps({"baseline": base, "best": best, "ranking": results}, indent=2))
    print(f"\n✅ Saved: { (RUNS/'addon_eval_plus.csv').as_posix() }")
    print(f"✅ Saved: { (RUNS/'addon_choice_plus.json').as_posix() }")

    # also write a best-only report for convenience
    best_label, best_flags = next((c for c in CONFIGS if c[0]==best["config"]), CONFIGS[-1])
    env = flags_env(best_flags)
    run([sys.executable, str(TOOLS / "run_all_plus.py")], env=env)
    # copy the report as report_best_plus.html
    final = ROOT / "report_all.html"
    best_report = ROOT / "report_best_plus.html"
    if final.exists():
        shutil.copy(final, best_report)
        print(f"✅ Wrote best report -> {best_report.as_posix()}")

if __name__ == "__main__":
    main()
