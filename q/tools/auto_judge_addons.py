#!/usr/bin/env python3
import os, sys, json, shutil, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNS  = ROOT / "runs_plus"

# Configs (progressively turning features on)
CONFIGS = [
    ("baseline (Council only)",             {"symbolic":False, "reflexive":False, "dna":False, "heartbeat":False, "hive":False}),
    ("+ DNA",                               {"symbolic":False, "reflexive":False, "dna":True,  "heartbeat":False, "hive":False}),
    ("+ Heartbeat",                         {"symbolic":False, "reflexive":False, "dna":True,  "heartbeat":True,  "hive":False}),
    ("+ Symbolic",                          {"symbolic":True,  "reflexive":False, "dna":True,  "heartbeat":True,  "hive":False}),
    ("+ Reflexive",                         {"symbolic":True,  "reflexive":True,  "dna":True,  "heartbeat":True,  "hive":False}),
    ("+ Hive (all current)",                {"symbolic":True,  "reflexive":True,  "dna":True,  "heartbeat":True,  "hive":True}),
]

def run_with_flags(flags: dict):
    """Call run_all_plus.py with SKIP_* env flags so it only builds what we want."""
    env = os.environ.copy()
    env["SKIP_DNA"]       = "0" if flags.get("dna") else "1"
    env["SKIP_HEARTBEAT"] = "0" if flags.get("heartbeat") else "1"
    env["SKIP_SYMBOLIC"]  = "0" if flags.get("symbolic") else "1"
    env["SKIP_REFLEXIVE"] = "0" if flags.get("reflexive") else "1"
    env["SKIP_HIVE"]      = "0" if flags.get("hive") else "1"
    print("▶ flags:", {k:v for k,v in env.items() if k.startswith("SKIP_")})
    subprocess.check_call([sys.executable, str(TOOLS / "run_all_plus.py")], env=env)

def parse_wf():
    """Read runs_plus/walk_forward_table.csv and return summary metrics."""
    csvp = RUNS / "walk_forward_table.csv"
    if not csvp.exists():
        return None
    try:
        df = pd.read_csv(csvp)
    except Exception:
        return None
    cols = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in cols: return cols[c]
        return None
    hit = pick("hit","oos_hit")
    sh  = pick("sharpe")
    dd  = pick("maxdd","max_dd","drawdown")
    def mean_or_none(colname):
        if not colname: return None
        s = pd.to_numeric(df[colname], errors="coerce").dropna()
        return float(s.mean()) if len(s) else None
    return {
        "assets": len(df),
        "hit_avg": mean_or_none(hit),
        "sharpe_avg": mean_or_none(sh),
        "maxdd_avg": mean_or_none(dd),
    }

def score_row(r):
    """Higher Sharpe, then higher Hit, then less-negative MaxDD (i.e., higher)"""
    def nz(x): return x if x is not None else float("-inf")
    return (nz(r.get("sharpe_avg")), nz(r.get("hit_avg")), nz(r.get("maxdd_avg")))

def print_table(title, rows, header):
    print("\n" + title)
    print("=" * len(title))
    if not rows:
        print("(no data)")
        return
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i,h in enumerate(header)]
    line = " | ".join(str(h).ljust(widths[i]) for i,h in enumerate(header))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(header))))

def main():
    results = []
    for label, toggles in CONFIGS:
        print(f"\n=== Testing: {label} ===")
        run_with_flags(toggles)
        m = parse_wf() or {"assets":None,"hit_avg":None,"sharpe_avg":None,"maxdd_avg":None}
        results.append({"config": label, **m})

    # Print matrix
    rows = []
    for r in results:
        rows.append([
            r["config"],
            r["assets"] if r["assets"] is not None else "—",
            f'{r["hit_avg"]:.3f}'    if r["hit_avg"]    is not None else "—",
            f'{r["sharpe_avg"]:.3f}' if r["sharpe_avg"] is not None else "—",
            f'{r["maxdd_avg"]:.3f}'  if r["maxdd_avg"]  is not None else "—",
        ])
    print_table("ADD-ON ABLATION MATRIX", rows, ["Config","Assets","Hit(avg)","Sharpe(avg)","MaxDD(avg)"])

    # Pick best
    best = max(results, key=score_row)
    baseline = results[0]  # first row is baseline
    d_hit = None if (best["hit_avg"] is None or baseline["hit_avg"] is None) else best["hit_avg"] - baseline["hit_avg"]
    d_sh  = None if (best["sharpe_avg"] is None or baseline["sharpe_avg"] is None) else best["sharpe_avg"] - baseline["sharpe_avg"]
    d_dd  = None if (best["maxdd_avg"] is None or baseline["maxdd_avg"] is None) else best["maxdd_avg"] - baseline["maxdd_avg"]

    print("\nRECOMMENDATION")
    print("==============")
    print(f"Best: {best['config']}")
    if d_sh  is not None: print(f"  ΔSharpe(avg): {d_sh:+.3f}")
    if d_hit is not None: print(f"  ΔHit(avg):    {d_hit:+.3f}")
    if d_dd  is not None: print(f"  ΔMaxDD(avg):  {d_dd:+.3f}  (less negative is better)")

    # Save choice
    choice = {"baseline": baseline, "best": best, "ranking": results}
    (RUNS/"addon_choice.json").write_text(json.dumps(choice, indent=2))
    print(f"\n✅ Saved choice -> { (RUNS/'addon_choice.json').as_posix() }")

    # Build a final "best only" report by setting flags accordingly
    best_label, best_flags = next((c for c in CONFIGS if c[0]==best["config"]), CONFIGS[-1])
    env = os.environ.copy()
    env["SKIP_DNA"]       = "0" if best_flags.get("dna") else "1"
    env["SKIP_HEARTBEAT"] = "0" if best_flags.get("heartbeat") else "1"
    env["SKIP_SYMBOLIC"]  = "0" if best_flags.get("symbolic") else "1"
    env["SKIP_REFLEXIVE"] = "0" if best_flags.get("reflexive") else "1"
    env["SKIP_HIVE"]      = "0" if best_flags.get("hive") else "1"
    subprocess.check_call([sys.executable, str(TOOLS / "run_all_plus.py")], env=env)

    # Copy final to report_best.html
    final = ROOT / "report_all.html"
    best_report = ROOT / "report_best.html"
    if final.exists():
        shutil.copy(final, best_report)
        print(f"✅ Wrote best report -> {best_report.as_posix()}")

if __name__ == "__main__":
    main()
