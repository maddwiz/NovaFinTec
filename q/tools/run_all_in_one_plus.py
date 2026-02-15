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

import os, sys, subprocess, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)

PY = sys.executable  # may be overridden by dependency-aware selector.

def exists(fp: Path) -> bool:
    return fp.exists()


def _python_has_core_deps(py_exec: str) -> bool:
    code = "import importlib.util as u;import sys;sys.exit(0 if u.find_spec('numpy') and u.find_spec('pandas') else 1)"
    try:
        cp = subprocess.run([py_exec, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return cp.returncode == 0
    except Exception:
        return False


def select_python() -> str | None:
    # Prefer current interpreter if deps exist.
    cur = str(sys.executable)
    if _python_has_core_deps(cur):
        return cur

    # Then try explicit override + common venv locations.
    candidates = []
    env_py = os.getenv("Q_PYTHON", "").strip()
    if env_py:
        candidates.append(env_py)
    candidates.extend(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT.parent / ".venv" / "bin" / "python"),
        ]
    )
    for c in candidates:
        p = Path(c)
        if not p.exists() or not p.is_file():
            continue
        if _python_has_core_deps(str(p)):
            return str(p)
    return None

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

def write_pipeline_status(failures, strict_mode: bool):
    payload = {
        "strict_mode": bool(strict_mode),
        "failed_steps": failures,
        "failed_count": int(len(failures)),
        "ok": len(failures) == 0,
    }
    (RUNS / "pipeline_status.json").write_text(json.dumps(payload, indent=2))
    print(f"✅ Wrote {RUNS/'pipeline_status.json'}")

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


def has_any_report(root: Path | None = None) -> bool:
    r = ROOT if root is None else Path(root)
    for n in ["report_best_plus.html", "report_all.html", "report_plus.html", "report.html"]:
        if (r / n).exists():
            return True
    return False


def ensure_report_exists():
    if has_any_report():
        return True
    # Attempt lightweight builders only when report is missing.
    for tool in ["tools/build_report_plus.py", "tools/build_report.py"]:
        ok, _rc = run_script(tool)
        if ok and has_any_report():
            print("✅ Built missing report artifact.")
            return True
    return has_any_report()


def ensure_env():
    # make sure PYTHONPATH includes project root
    os.environ.setdefault("PYTHONPATH", str(ROOT))
    global PY
    selected = select_python()
    if selected is None:
        print("(!) Missing numpy/pandas in current Python and no valid venv found.")
        print("    Set Q_PYTHON to a working interpreter, e.g.:")
        print("    Q_PYTHON=/absolute/path/to/venv/bin/python python tools/run_all_in_one_plus.py")
        return False
    if selected != PY:
        print(f"↺ switching python interpreter to {selected}")
    PY = selected
    # friendly echo
    print(f"PYTHON     : {PY}")
    print(f"PROJECT    : {ROOT}")
    print(f"PYTHONPATH : {os.environ.get('PYTHONPATH')}")
    return True

if __name__ == "__main__":
    strict = str(os.getenv("Q_STRICT", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if not ensure_env():
        write_pipeline_status([{"step": "env_preflight", "code": 2}], strict_mode=strict)
        raise SystemExit(2)
    failures = []
    # Reset status at run start so alert checks do not read stale failures.
    write_pipeline_status([], strict_mode=strict)

    # ---------- PHASE 0: Primers / basics ----------
    # (A) Lightweight nested WF summary (scaffold)
    ok, rc = run_script("tools/nested_wf_lite.py")
    if not ok and rc is not None: failures.append({"step": "tools/nested_wf_lite.py", "code": rc})

    # (B) Build minimal returns + base weights (so downstream steps have inputs)
    ok, rc = run_script("tools/make_returns_and_weights.py")
    if not ok and rc is not None: failures.append({"step": "tools/make_returns_and_weights.py", "code": rc})
    ok, rc = run_script("tools/make_asset_names.py")
    if not ok and rc is not None: failures.append({"step": "tools/make_asset_names.py", "code": rc})

    # (C) Build council votes (real if present, else sleeves or synthetic)
    ok, rc = run_script("tools/make_council_votes.py")
    if not ok and rc is not None: failures.append({"step": "tools/make_council_votes.py", "code": rc})
    # (D) Build symbolic/heartbeat/reflexive layers
    ok, rc = run_script("tools/make_symbolic.py")
    if not ok and rc is not None: failures.append({"step": "tools/make_symbolic.py", "code": rc})
    ok, rc = run_script("tools/run_symbolic_governor.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_symbolic_governor.py", "code": rc})
    # Build shock/news mask early for downstream risk gating.
    ok, rc = run_script("tools/run_news_sentinel.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_news_sentinel.py", "code": rc})
    ok, rc = run_script("tools/make_heartbeat.py")
    if not ok and rc is not None: failures.append({"step": "tools/make_heartbeat.py", "code": rc})
    # DNA drift/stress governor (with fallback from returns).
    ok, rc = run_script("tools/run_dna_governor.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_dna_governor.py", "code": rc})
    ok, rc = run_script("tools/make_reflexive.py")
    if not ok and rc is not None: failures.append({"step": "tools/make_reflexive.py", "code": rc})
    # (E) Build hive signals + per-hive walk-forward diagnostics
    ok, rc = run_script("tools/make_hive.py")
    if not ok and rc is not None: failures.append({"step": "tools/make_hive.py", "code": rc})
    ok, rc = run_script("tools/run_hive_walkforward.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_hive_walkforward.py", "code": rc})

    # ---------- PHASE 1: Guardrails ----------
    # Parameter stability, turnover, disagreement gate + DD scaling + report card
    ok, rc = run_script("tools/run_guardrails.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_guardrails.py", "code": rc})
    # Disagreement heatmap (table) → report
    ok, rc = run_script("tools/run_disagreement_heatmap.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_disagreement_heatmap.py", "code": rc})

    # ---------- PHASE 2: Hive Brain ----------
    # Leakage-safe ridge meta over councils → meta_stack_pred.csv
    ok, rc = run_script("tools/run_meta_stack.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_meta_stack.py", "code": rc})
    # Tiny MLP fusion of councils → synapses_pred.csv
    ok, rc = run_script("tools/run_synapses.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_synapses.py", "code": rc})
    # Confidence-aware blend of meta + synapses
    ok, rc = run_script("tools/run_council_meta_mix.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_council_meta_mix.py", "code": rc})
    # Optional NovaSpine per-hive memory feedback.
    ok, rc = run_script("tools/run_novaspine_hive_feedback.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_novaspine_hive_feedback.py", "code": rc})
    # Cross-hive arbitration (weights per hive)
    ok, rc = run_script("tools/run_cross_hive.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_cross_hive.py", "code": rc})
    # Per-hive transparency artifact + report card.
    ok, rc = run_script("tools/run_hive_transparency.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_hive_transparency.py", "code": rc})
    # Ecosystem age governor (atrophy/split/fusion dynamics on hive weights)
    ok, rc = run_script("tools/run_ecosystem_age.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_ecosystem_age.py", "code": rc})
    # Tail-blender over base weights and hedges
    ok, rc = run_script("tools/run_tail_blender.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_tail_blender.py", "code": rc})

    # ---------- PHASE 3: Refinements ----------
    # Reflex health (and optional gating of reflex signal)
    ok, rc = run_script("tools/run_reflex_health.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_reflex_health.py", "code": rc})
    # Dream/reflex/symbolic coherence governor.
    ok, rc = run_script("tools/run_dream_coherence.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_dream_coherence.py", "code": rc})
    # Risk parity sleeve (T x N weights)
    ok, rc = run_script("tools/run_risk_parity.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_risk_parity.py", "code": rc})
    # Adaptive caps on weights (vol-based clamps)
    ok, rc = run_script("tools/run_adaptive_caps.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_adaptive_caps.py", "code": rc})
    # Feature neutralization between two feature sets
    ok, rc = run_script("tools/run_feature_neutralizer.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_feature_neutralizer.py", "code": rc})
    # Legacy smooth scaler from DNA/heartbeat/symbolic/reflexive layers
    ok, rc = run_script("tools/tune_legacy_knobs.py")
    if not ok and rc is not None: failures.append({"step": "tools/tune_legacy_knobs.py", "code": rc})
    # Pull recall context from NovaSpine and convert to a safe scalar boost.
    ok, rc = run_script("tools/run_novaspine_context.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_novaspine_context.py", "code": rc})
    # Quality governor from nested/hive/council/system diagnostics
    ok, rc = run_script("tools/run_quality_governor.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_quality_governor.py", "code": rc})
    # Assemble final portfolio weights from available layers
    ok, rc = run_script("tools/build_final_portfolio.py")
    if not ok and rc is not None: failures.append({"step": "tools/build_final_portfolio.py", "code": rc})
    # Apply execution constraints for live realism
    ok, rc = run_script("tools/run_execution_constraints.py", ["--replace-final"])
    if not ok and rc is not None: failures.append({"step": "tools/run_execution_constraints.py", "code": rc})
    # Drift watchdog across runs (final weights vs prior snapshot).
    ok, rc = run_script("tools/run_portfolio_drift_watch.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_portfolio_drift_watch.py", "code": rc})
    # Immune drill: synthetic stress test of final constrained weights.
    ok, rc = run_script("tools/run_immune_drill.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_immune_drill.py", "code": rc})
    # Emit a health snapshot for unattended operation
    ok, rc = run_script("tools/run_system_health.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_system_health.py", "code": rc})
    ok, rc = run_script("tools/run_health_alerts.py")
    if not ok and rc is not None: failures.append({"step": "tools/run_health_alerts.py", "code": rc})
    # Export Q overlay pack for AION consumption (safe degraded mode if needed)
    export_args = []
    mirror_json = str(os.getenv("Q_EXPORT_MIRROR_JSON", "")).strip()
    if mirror_json:
        export_args.extend(["--mirror-json", mirror_json])
    ok, rc = run_script("tools/export_aion_signal_pack.py", export_args)
    if not ok and rc is not None: failures.append({"step": "tools/export_aion_signal_pack.py", "code": rc})
    # Replay queued NovaSpine batches before writing the new one.
    ok, rc = run_script("tools/replay_novaspine_outbox.py")
    if not ok and rc is not None: failures.append({"step": "tools/replay_novaspine_outbox.py", "code": rc})
    # Optional cold/meta memory sync bridge (NovaSpine).
    ok, rc = run_script("tools/sync_novaspine_memory.py")
    if not ok and rc is not None: failures.append({"step": "tools/sync_novaspine_memory.py", "code": rc})

    # ---------- REPORT ----------
    # Many scripts already append cards; ensure a report exists then open best file.
    ensure_report_exists()
    try_open_report()

    write_pipeline_status(failures, strict_mode=strict)

    print("\n✅ All-in-one pipeline finished (with safe skips where inputs were missing).")
    print("   Check runs_plus/ for new CSVs and your report HTML for new cards.")
    if strict and failures:
        print("(!) Strict mode enabled and some steps failed.")
        raise SystemExit(2)
