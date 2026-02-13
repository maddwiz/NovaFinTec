#!/usr/bin/env python3
# tools/sweep_osc_cost_grid.py
# Runs a small grid over OSC_COST_BPS and OSC_MAX_DPOS,
# reuses your existing tools, and logs results to runs_plus/osc_cost_sweep.csv.

from pathlib import Path
import os, json, subprocess, shlex
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(parents=True, exist_ok=True)

COSTS = [1.0, 2.0, 3.0]          # bps
DPOS  = [0.05, 0.10, 0.20]       # max daily position change

def run(cmd, env=None):
    print("$", cmd)
    res = subprocess.run(shlex.split(cmd), cwd=ROOT, env=env or os.environ.copy(),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    return res.returncode == 0

rows = []
for cbps in COSTS:
    for dpos in DPOS:
        env = os.environ.copy()
        env["OSC_COST_BPS"] = str(cbps)
        env["OSC_MAX_DPOS"] = str(dpos)

        ok = run("python tools/run_osc_portfolio_costed.py", env=env)
        if not ok: 
            rows.append({"cost_bps":cbps, "max_dpos":dpos, "ok":False})
            continue

        summ_p = RUNS / "osc_portfolio_costed_summary.json"
        if not summ_p.exists():
            rows.append({"cost_bps":cbps, "max_dpos":dpos, "ok":False})
            continue
        meta = json.loads(summ_p.read_text())
        g = meta.get("gross", {})
        n = meta.get("net", {})
        params = meta.get("params", {})

        # Blend NET osc with Main
        ok_b = run("python tools/blend_main_with_osc_costed.py", env=env)
        best = {}
        try:
            best = json.loads((RUNS/"blend_main_osc_costed_summary.json").read_text())
        except Exception:
            pass

        rows.append({
            "cost_bps": cbps,
            "max_dpos": dpos,
            "osc_net_sharpe": n.get("sharpe"),
            "osc_net_hit": n.get("hit"),
            "osc_net_maxdd": n.get("maxdd"),
            "blend_alpha": best.get("alpha"),
            "blend_sharpe": best.get("sharpe"),
            "blend_hit": best.get("hit"),
            "blend_maxdd": best.get("maxdd"),
            "assets_used": params.get("assets_used"),
            "ok": True
        })

df = pd.DataFrame(rows)
df.to_csv(RUNS/"osc_cost_sweep.csv", index=False)
print("\nSaved grid results to:", RUNS/"osc_cost_sweep.csv")
if not df.empty:
    best = df[df["blend_sharpe"].notna()].sort_values("blend_sharpe", ascending=False).head(5)
    print("\nTOP 5 by blended Sharpe:")
    print(best[["cost_bps","max_dpos","blend_sharpe","blend_alpha","osc_net_sharpe","assets_used"]].to_string(index=False))
else:
    print("No rows recorded. Check earlier errors.")
