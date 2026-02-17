#!/usr/bin/env python3
"""
Governor ablation runner.

Runs controlled leave-one-out checks for build_final_portfolio governors and
writes a compact scorecard:
  - runs_plus/governor_ablation.csv
  - runs_plus/governor_ablation.json

Notes:
- Requires runs_plus/asset_returns.csv (rebuild via tools/rebuild_asset_matrix.py).
- Always restores baseline outputs at the end.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)
PYTHON = str(Path(sys.executable))

TRACE_GOVS = [
    "rank_sleeve_blend",
    "turnover_governor",
    "meta_execution_gate",
    "council_gate",
    "meta_mix_leverage",
    "meta_mix_reliability",
    "heartbeat_scaler",
    "legacy_scaler",
    "dna_stress_governor",
    "symbolic_governor",
    "dream_coherence",
    "reflex_health_governor",
    "hive_diversification",
    "hive_persistence",
    "global_governor",
    "quality_governor",
    "regime_fracture_governor",
    "regime_moe_governor",
    "uncertainty_sizing",
    "novaspine_context_boost",
    "novaspine_hive_boost",
    "shock_mask_guard",
    "runtime_floor",
]


def _parse_disabled(raw: str) -> set[str]:
    out: set[str] = set()
    for tok in str(raw or "").split(","):
        t = str(tok).strip().lower()
        if t:
            out.add(t)
    return out


def _load_baseline_disabled() -> set[str]:
    p = RUNS / "final_portfolio_info.json"
    if not p.exists():
        return set()
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return set()
    vals = obj.get("disabled_governors", [])
    if not isinstance(vals, list):
        return set()
    return {str(x).strip().lower() for x in vals if str(x).strip()}


def _run(cmd: list[str], env: dict[str, str]) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _build_and_make_daily(extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    _run([PYTHON, str(ROOT / "tools" / "build_final_portfolio.py")], env)
    _run([PYTHON, str(ROOT / "tools" / "make_daily_from_weights.py")], env)


def _metrics_from_daily() -> dict:
    p = RUNS / "daily_returns.csv"
    if not p.exists():
        raise RuntimeError("Missing runs_plus/daily_returns.csv after make_daily_from_weights.")
    r = np.loadtxt(p, delimiter=",")
    r = np.asarray(r, float).ravel()
    if r.size == 0:
        raise RuntimeError("daily_returns.csv is empty.")
    mu = float(np.nanmean(r))
    sd = float(np.nanstd(r) + 1e-12)
    sharpe = float((mu / sd) * np.sqrt(252.0))
    hit = float(np.sum(r > 0.0) / max(1, r.size))
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    mdd = float(np.min(eq - peak))
    return {
        "sharpe": sharpe,
        "hit_rate": hit,
        "max_drawdown": mdd,
        "mean": mu,
        "stdev": sd,
        "n": int(r.size),
    }


def _write_csv(rows: list[dict], outp: Path) -> None:
    cols = [
        "scenario",
        "disabled",
        "sharpe",
        "hit_rate",
        "max_drawdown",
        "delta_sharpe_vs_baseline",
        "delta_hit_vs_baseline",
        "delta_maxdd_vs_baseline",
    ]
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})


def main() -> int:
    if not (RUNS / "asset_returns.csv").exists():
        print("(!) Missing runs_plus/asset_returns.csv. Run tools/rebuild_asset_matrix.py first.")
        return 0

    base_disabled = _load_baseline_disabled()
    candidates = [g for g in TRACE_GOVS if g not in base_disabled]

    rows: list[dict] = []
    baseline_metrics: dict | None = None
    try:
        _build_and_make_daily()
        baseline_metrics = _metrics_from_daily()
        rows.append(
            {
                "scenario": "baseline",
                "disabled": ",".join(sorted(base_disabled)),
                **baseline_metrics,
                "delta_sharpe_vs_baseline": 0.0,
                "delta_hit_vs_baseline": 0.0,
                "delta_maxdd_vs_baseline": 0.0,
            }
        )

        # All governors off (+ concentration off) reference.
        all_disabled = sorted(set(base_disabled).union(set(TRACE_GOVS)))
        _build_and_make_daily(
            {
                "Q_DISABLE_GOVERNORS": ",".join(all_disabled),
                "Q_USE_CONCENTRATION_GOV": "0",
            }
        )
        m_all_off = _metrics_from_daily()
        rows.append(
            {
                "scenario": "all_off",
                "disabled": ",".join(all_disabled + ["concentration_governor"]),
                **m_all_off,
                "delta_sharpe_vs_baseline": float(m_all_off["sharpe"] - baseline_metrics["sharpe"]),
                "delta_hit_vs_baseline": float(m_all_off["hit_rate"] - baseline_metrics["hit_rate"]),
                "delta_maxdd_vs_baseline": float(m_all_off["max_drawdown"] - baseline_metrics["max_drawdown"]),
            }
        )

        # Leave-one-out: disable one governor at a time from baseline profile.
        for gov in candidates:
            disabled = sorted(set(base_disabled).union({gov}))
            _build_and_make_daily({"Q_DISABLE_GOVERNORS": ",".join(disabled)})
            m = _metrics_from_daily()
            rows.append(
                {
                    "scenario": f"drop_{gov}",
                    "disabled": gov,
                    **m,
                    "delta_sharpe_vs_baseline": float(m["sharpe"] - baseline_metrics["sharpe"]),
                    "delta_hit_vs_baseline": float(m["hit_rate"] - baseline_metrics["hit_rate"]),
                    "delta_maxdd_vs_baseline": float(m["max_drawdown"] - baseline_metrics["max_drawdown"]),
                }
            )

        # Concentration governor leave-one-out.
        _build_and_make_daily({"Q_USE_CONCENTRATION_GOV": "0"})
        m_conc_off = _metrics_from_daily()
        rows.append(
            {
                "scenario": "drop_concentration_governor",
                "disabled": "concentration_governor",
                **m_conc_off,
                "delta_sharpe_vs_baseline": float(m_conc_off["sharpe"] - baseline_metrics["sharpe"]),
                "delta_hit_vs_baseline": float(m_conc_off["hit_rate"] - baseline_metrics["hit_rate"]),
                "delta_maxdd_vs_baseline": float(m_conc_off["max_drawdown"] - baseline_metrics["max_drawdown"]),
            }
        )
    finally:
        # Restore baseline so live/export paths are left consistent.
        try:
            _build_and_make_daily()
        except Exception as e:
            print(f"(!) Failed to restore baseline daily/weights: {e}")

    if not rows or baseline_metrics is None:
        print("(!) No ablation rows produced.")
        return 1

    out_csv = RUNS / "governor_ablation.csv"
    out_json = RUNS / "governor_ablation.json"
    _write_csv(rows, out_csv)

    # Rank by Sharpe impact (drop worsens Sharpe => positive keeper score)
    ranked = sorted(
        [r for r in rows if str(r.get("scenario", "")).startswith("drop_")],
        key=lambda x: float(x.get("delta_sharpe_vs_baseline", 0.0)),
    )
    summary = {
        "baseline_disabled": sorted(base_disabled),
        "baseline": baseline_metrics,
        "top_keepers_by_sharpe": ranked[:5],
        "top_neutral_or_harmful_by_sharpe": ranked[-5:],
        "rows": rows,
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"✅ Wrote {out_csv}")
    print(f"✅ Wrote {out_json}")
    print(
        "Baseline:",
        f"Sharpe={baseline_metrics['sharpe']:.3f}",
        f"Hit={baseline_metrics['hit_rate']:.3f}",
        f"MaxDD={baseline_metrics['max_drawdown']:.3f}",
    )
    if ranked:
        best = ranked[0]
        worst = ranked[-1]
        print(
            "Most beneficial governor (by Sharpe when removed):",
            str(best.get("disabled", "")),
            f"delta={float(best.get('delta_sharpe_vs_baseline', 0.0)):+.4f}",
        )
        print(
            "Most neutral/harmful governor (by Sharpe when removed):",
            str(worst.get("disabled", "")),
            f"delta={float(worst.get('delta_sharpe_vs_baseline', 0.0)):+.4f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
