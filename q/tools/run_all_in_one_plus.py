#!/usr/bin/env python3
"""
All-in-One Runner (WF → Guardrails → Hive Brain → Refinements → Report)

What it does, in order (skips missing pieces safely):
  1) Fresh basics: nested WF scaffold, returns/weights, council votes
  2) Guardrails: stability/turnover/gate + disagreement heatmap + DD scaler
  3) Hive Brain: meta-learner, synapses (tiny MLP), cross-hive, tail-blender
  4) Refinements: reflex health, risk-parity sleeve, adaptive caps, neutralizer
  5) Report: tries report_all.html/report_best_plus.html and appends cards already

Outputs go to runs_plus/ when possible.
"""

import os, sys, subprocess, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)

PY = sys.executable  # use current venv python

def exists(fp: Path) -> bool:
    return fp.exists()

def run_script(relpath: str, args=None):
    """Run a Python script if it exists. Returns (ok, returncode)."""
    p = ROOT / relpath
    if not exists(p):
        print(f"… skip (missing): {relpath}")
        return False, None
    cmd = [PY, str(p)] + (args or [])
    print(f"\n▶ {relpath} {' '.join(args or [])}")
    cp = subprocess.run(cmd, cwd=str(ROOT), stdout=sys.stdout, stderr=sys.stderr)
    ok = (cp.returncode == 0)
    if not ok:
        print(f"(!) {relpath} exited with code {cp.returncode} — continuing.")
    return ok, cp.returncode

def try_open_report():
    # prefer best_plus, then all, then plus, then base
    names = ["report_best_plus.html", "report_all.html", "report_plus.html", "report.html"]
    for n in names:
        if exists(ROOT/n):
            # macOS: open in default browser
            try:
                subprocess.run(["open", str(ROOT/n)])
            except Exception:
                pass
            print(f"✅ Report available: {n}")
            return
    print("(!) No report HTML found to open.")

def ensure_env():
    # make sure PYTHONPATH includes project root
    os.environ.setdefault("PYTHONPATH", str(ROOT))
    # friendly echo
    print(f"PYTHON     : {PY}")
    print(f"PROJECT    : {ROOT}")
    print(f"PYTHONPATH : {os.environ.get('PYTHONPATH')}")

if __name__ == "__main__":
    ensure_env()

    # ---------- PHASE 0: Primers / basics ----------
    # (A) Lightweight nested WF summary (scaffold)
    run_script("tools/nested_wf_lite.py")

    # (B) Build minimal returns + base weights (so downstream steps have inputs)
    run_script("tools/make_returns_and_weights.py")

    # (C) Build council votes (real if present, else sleeves or synthetic)
    run_script("tools/make_council_votes.py")
    # (D) Build symbolic/heartbeat/reflexive layers
    run_script("tools/make_symbolic.py")
    run_script("tools/make_heartbeat.py")
    run_script("tools/make_reflexive.py")
    # (E) Build hive signals + per-hive walk-forward diagnostics
    run_script("tools/make_hive.py")
    run_script("tools/run_hive_walkforward.py")

    # ---------- PHASE 1: Guardrails ----------
    # Parameter stability, turnover, disagreement gate + DD scaling + report card
    run_script("tools/run_guardrails.py")
    # Disagreement heatmap (table) → report
    run_script("tools/run_disagreement_heatmap.py")

    # ---------- PHASE 2: Hive Brain ----------
    # Leakage-safe ridge meta over councils → meta_stack_pred.csv
    run_script("tools/run_meta_stack.py")
    # Tiny MLP fusion of councils → synapses_pred.csv
    run_script("tools/run_synapses.py")
    # Confidence-aware blend of meta + synapses
    run_script("tools/run_council_meta_mix.py")
    # Cross-hive arbitration (weights per hive)
    run_script("tools/run_cross_hive.py")
    # Ecosystem age governor (atrophy/split/fusion dynamics on hive weights)
    run_script("tools/run_ecosystem_age.py")
    # Tail-blender over base weights and hedges
    run_script("tools/run_tail_blender.py")

    # ---------- PHASE 3: Refinements ----------
    # Reflex health (and optional gating of reflex signal)
    run_script("tools/run_reflex_health.py")
    # Risk parity sleeve (T x N weights)
    run_script("tools/run_risk_parity.py")
    # Adaptive caps on weights (vol-based clamps)
    run_script("tools/run_adaptive_caps.py")
    # Feature neutralization between two feature sets
    run_script("tools/run_feature_neutralizer.py")
    # Legacy smooth scaler from DNA/heartbeat/symbolic/reflexive layers
    run_script("tools/tune_legacy_knobs.py")
    # Assemble final portfolio weights from available layers
    run_script("tools/build_final_portfolio.py")
    # Emit a health snapshot for unattended operation
    run_script("tools/run_system_health.py")

    # ---------- REPORT ----------
    # Many scripts already append cards; try to open the best report file.
    try_open_report()

    print("\n✅ All-in-one pipeline finished (with safe skips where inputs were missing).")
    print("   Check runs_plus/ for new CSVs and your report HTML for new cards.")
