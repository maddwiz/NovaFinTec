#!/usr/bin/env python3
# Alert gate for unattended runs.
#
# Reads:
#   runs_plus/system_health.json
#   runs_plus/guardrails_summary.json (optional)
#
# Exits:
#   0 if healthy enough
#   2 if alert conditions are met

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


if __name__ == "__main__":
    min_health = float(os.getenv("Q_MIN_HEALTH_SCORE", "70"))
    min_global = float(os.getenv("Q_MIN_GLOBAL_GOV_MEAN", "0.45"))
    min_quality_gov = float(os.getenv("Q_MIN_QUALITY_GOV_MEAN", "0.60"))
    min_quality_score = float(os.getenv("Q_MIN_QUALITY_SCORE", "0.45"))
    require_immune_pass = str(os.getenv("Q_REQUIRE_IMMUNE_PASS", "0")).strip().lower() in {"1", "true", "yes", "on"}
    max_issues = int(os.getenv("Q_MAX_HEALTH_ISSUES", "2"))
    min_nested_sharpe = float(os.getenv("Q_MIN_NESTED_SHARPE", "0.20"))
    min_nested_assets = int(os.getenv("Q_MIN_NESTED_ASSETS", "3"))

    health = _load_json(RUNS / "system_health.json") or {}
    guards = _load_json(RUNS / "guardrails_summary.json") or {}
    nested = _load_json(RUNS / "nested_wf_summary.json") or {}
    quality = _load_json(RUNS / "quality_snapshot.json") or {}
    immune = _load_json(RUNS / "immune_drill.json") or {}
    pipeline = _load_json(RUNS / "pipeline_status.json") or {}
    issues = []

    score = float(health.get("health_score", 0.0))
    n_issues = len(health.get("issues", []) or [])
    if score < min_health:
        issues.append(f"health_score<{min_health} ({score:.1f})")
    if n_issues > max_issues:
        issues.append(f"health_issues>{max_issues} ({n_issues})")

    gg = guards.get("global_governor", {}) if isinstance(guards, dict) else {}
    gmean = gg.get("mean", None)
    if gmean is not None:
        try:
            gmean = float(gmean)
            if gmean < min_global:
                issues.append(f"global_governor_mean<{min_global} ({gmean:.3f})")
        except Exception:
            pass

    n_assets = nested.get("assets", None)
    n_sh = nested.get("avg_oos_sharpe", None)
    try:
        n_assets = int(n_assets) if n_assets is not None else None
    except Exception:
        n_assets = None
    try:
        n_sh = float(n_sh) if n_sh is not None else None
    except Exception:
        n_sh = None
    if n_assets is not None and n_sh is not None and n_assets >= min_nested_assets and n_sh < min_nested_sharpe:
        issues.append(f"nested_wf_sharpe<{min_nested_sharpe} ({n_sh:.3f}) over {n_assets} assets")

    q_mean = quality.get("quality_governor_mean", None)
    q_score = quality.get("quality_score", None)
    try:
        q_mean = float(q_mean) if q_mean is not None else None
    except Exception:
        q_mean = None
    try:
        q_score = float(q_score) if q_score is not None else None
    except Exception:
        q_score = None
    if q_mean is not None and q_mean < min_quality_gov:
        issues.append(f"quality_governor_mean<{min_quality_gov} ({q_mean:.3f})")
    if q_score is not None and q_score < min_quality_score:
        issues.append(f"quality_score<{min_quality_score} ({q_score:.3f})")

    immune_ok = bool(immune.get("ok", False)) if isinstance(immune, dict) else False
    immune_pass = bool(immune.get("pass", False)) if isinstance(immune, dict) else False
    if require_immune_pass:
        if not immune_ok:
            issues.append("immune_drill_missing_or_invalid")
        elif not immune_pass:
            issues.append("immune_drill_failed")

    failed_steps = int(pipeline.get("failed_count", 0)) if isinstance(pipeline, dict) else 0
    if failed_steps > 0:
        issues.append(f"pipeline_failed_steps={failed_steps}")

    payload = {
        "ok": len(issues) == 0,
        "thresholds": {
            "min_health_score": min_health,
            "max_health_issues": max_issues,
            "min_global_governor_mean": min_global,
            "min_quality_governor_mean": min_quality_gov,
            "min_quality_score": min_quality_score,
            "require_immune_pass": require_immune_pass,
            "min_nested_sharpe": min_nested_sharpe,
            "min_nested_assets": min_nested_assets,
        },
        "observed": {
            "health_score": score,
            "health_issues": n_issues,
            "global_governor_mean": gmean,
            "quality_governor_mean": q_mean,
            "quality_score": q_score,
            "immune_ok": immune_ok,
            "immune_pass": immune_pass,
            "nested_assets": n_assets,
            "nested_avg_oos_sharpe": n_sh,
            "pipeline_failed_steps": failed_steps,
        },
        "alerts": issues,
    }
    (RUNS / "health_alerts.json").write_text(json.dumps(payload, indent=2))
    print(f"✅ Wrote {RUNS/'health_alerts.json'}")
    if issues:
        print("ALERT:", "; ".join(issues))
        raise SystemExit(2)
    print("✅ Health alerts clear")
