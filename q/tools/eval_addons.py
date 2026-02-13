#!/usr/bin/env python3
import os, sys, json, shutil, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNS  = ROOT / "runs_plus"

CONFIGS = [
    ("baseline (Council only)",             {"symbolic":False, "reflexive":False, "dna":False, "heartbeat":False, "hive":False}),
    ("+ DNA",                               {"symbolic":False, "reflexive":False, "dna":True,  "heartbeat":False, "hive":False}),
    ("+ Heartbeat",                         {"symbolic":False, "reflexive":False, "dna":True,  "heartbeat":True,  "hive":False}),
    ("+ Symbolic",                          {"symbolic":True,  "reflexive":False, "dna":True,  "heartbeat":True,  "hive":False}),
    ("+ Reflexive",                         {"symbolic":True,  "reflexive":True,  "dna":True,  "heartbeat":True,  "hive":False}),
    ("+ Hive (all current)",                {"symbolic":True,  "reflexive":True,  "dna":True,  "heartbeat":True,  "hive":True}),
]

ARTIFACTS = {
    "dna":        [RUNS/"dna_drift.json", RUNS/"dna_drift.png"],
    "heartbeat":  [RUNS/"heartbeat.json", RUNS/"heartbeat.png"],
    "symbolic":   [RUNS/"symbolic.json", RUNS/"symbolic_scores.csv"],
    "reflexive":  [RUNS/"reflexive.json", RUNS/"reflexive.png"],
    "hive":       [RUNS/"hive.json", RUNS/"hive.png"],
}

def run(cmd):
    print("▶", " ".join(map(str, cmd)))
    subprocess.check_call(cmd)

def exists(p: Path) -> bool:
    try: return p.exists()
    except: return False

def stash(art_paths):
    moved = []
    for p in art_paths:
        if exists(p):
            dest = p.with_suffix(p.suffix + ".off")
            try:
                shutil.move(p, dest)
                moved.append((p, dest))
            except Exception:
                pass
    return moved

def restore(moved):
    for src, dst in moved:
        # note: stash() stored as (original, .off); now we move back
        if exists(dst):
            shutil.move(dst, src)

def make_state(targets: dict):
    """
    Ensure the on/off state of artifacts matches targets.
    If a feature is False -> stash existing artifacts.
    If True -> (re)generate by calling the make_* tool.
    """
    # 1) stash the ones we want OFF
    stashed = []
    for name, want in targets.items():
        if not want:
            stashed += stash(ARTIFACTS.get(name, []))

    # 2) (re)build full report; run_all_plus.py will regenerate any ON artifacts
    run([sys.executable, str(TOOLS / "run_all_plus.py")])
    return stashed

def parse_wf():
    csvp = RUNS / "walk_forward_table.csv"
    if not exists(csvp):
        return None
    try:
        df = pd.read_csv(csvp)
    except Exception:
        return None
    # guess column names
    cols = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in cols: return cols[c]
        return None
    hit = pick("hit","oos_hit")
    sharpe = pick("sharpe")
    dd = pick("maxdd","max_dd","drawdown")
    out = {
        "assets": len(df),
        "hit_avg": float(pd.to_numeric(df[hit], errors="coerce").mean()) if hit else None,
        "sharpe_avg": float(pd.to_numeric(df[sharpe], errors="coerce").mean()) if sharpe else None,
        "maxdd_avg": float(pd.to_numeric(df[dd], errors="coerce").mean()) if dd else None,
    }
    return out

def main():
    results = []
    for label, toggles in CONFIGS:
        print("\n=== EVALUATING:", label, "===")
        # Remember what we stashed to restore after this config
        stashed = make_state(toggles)
        metrics = parse_wf() or {"assets":None,"hit_avg":None,"sharpe_avg":None,"maxdd_avg":None}
        results.append({"config": label, **metrics})
        # restore any artifacts we hid
        restore(stashed)

    # Print table
    print("\nADD-ON ABLATION MATRIX")
    print("=======================")
    rows = []
    for r in results:
        rows.append([
            r["config"],
            r["assets"] if r["assets"] is not None else "—",
            f'{r["hit_avg"]:.3f}' if r["hit_avg"] is not None else "—",
            f'{r["sharpe_avg"]:.3f}' if r["sharpe_avg"] is not None else "—",
            f'{r["maxdd_avg"]:.3f}' if r["maxdd_avg"] is not None else "—",
        ])
    # pretty print
    widths = [max(len(str(x)) for x in col) for col in zip(*([["Config","Assets","Hit(avg)","Sharpe(avg)","MaxDD(avg)"]] + rows))]
    fmt = " | ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format("Config","Assets","Hit(avg)","Sharpe(avg)","MaxDD(avg)"))
    print("-"* (sum(widths) + 3*(len(widths)-1)))
    for row in rows:
        print(fmt.format(*row))

    # Save csv
    df = pd.DataFrame(results)
    out_csv = RUNS / "addon_eval.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved matrix: {out_csv.as_posix()}")

if __name__ == "__main__":
    main()
