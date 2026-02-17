#!/usr/bin/env python3
# Build a data-driven governor profile from final_governor_trace.csv.
#
# Writes:
#   runs_plus/governor_profile.json
#
# The profile can be consumed by build_final_portfolio.py to disable
# effectively neutral governors and set a runtime floor target.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _safe_float(x, default=None):
    try:
        v = float(x)
    except Exception:
        return default
    if not np.isfinite(v):
        return default
    return float(v)


def _parse_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _ablation_map_from_summary(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = obj.get("rows")
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sc = str(row.get("scenario", ""))
        if not sc.startswith("drop_"):
            continue
        name = sc[len("drop_") :].strip().lower()
        if not name:
            continue
        out[name] = {
            "delta_sharpe": _safe_float(row.get("delta_sharpe_vs_baseline"), None),
            "delta_hit": _safe_float(row.get("delta_hit_vs_baseline"), None),
            "delta_maxdd": _safe_float(row.get("delta_maxdd_vs_baseline"), None),
        }
    return out


def build_governor_profile(
    trace: pd.DataFrame,
    *,
    eps: float = 0.01,
    neutral_mean_abs: float = 0.005,
    neutral_active_share: float = 0.10,
    protected: set[str] | None = None,
    ablation_impacts: dict[str, dict] | None = None,
    ablation_min_sharpe_gain: float = 0.015,
    ablation_max_hit_drop: float = 0.003,
    ablation_max_mdd_worsen: float = 0.003,
):
    prot = {str(x).strip().lower() for x in (protected or set()) if str(x).strip()}
    abl = {str(k).strip().lower(): v for k, v in (ablation_impacts or {}).items() if str(k).strip()}
    if trace.empty:
        return {
            "ok": False,
            "reason": "missing_trace",
            "disable_governors": [],
            "runtime_total_floor": 0.10,
            "governors": [],
        }

    cols = [str(c) for c in trace.columns if str(c) != "runtime_total_scalar"]
    gov_rows = []
    for c in cols:
        s = pd.to_numeric(trace[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        arr = s.values.astype(float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        mad = float(np.mean(np.abs(arr - 1.0)))
        active = float(np.mean(np.abs(arr - 1.0) > float(eps)))
        cname = str(c).strip().lower()

        if cname in prot:
            status = "protected"
        elif (mad <= float(neutral_mean_abs) and active <= float(neutral_active_share)) or (
            std <= 0.002 and abs(mean - 1.0) <= 0.01
        ):
            status = "neutral_redundant"
        else:
            status = "active"
        impact = abl.get(cname)
        delta_sh = _safe_float((impact or {}).get("delta_sharpe"), None)
        delta_hit = _safe_float((impact or {}).get("delta_hit"), None)
        delta_mdd = _safe_float((impact or {}).get("delta_maxdd"), None)
        if cname not in prot and impact and (delta_sh is not None) and (delta_hit is not None) and (delta_mdd is not None):
            if (
                delta_sh >= float(ablation_min_sharpe_gain)
                and delta_hit >= -float(ablation_max_hit_drop)
                and delta_mdd >= -float(ablation_max_mdd_worsen)
            ):
                status = "ablation_harmful"

        gov_rows.append(
            {
                "name": str(c),
                "mean": mean,
                "std": std,
                "min": mn,
                "max": mx,
                "mean_abs_dev": mad,
                "active_share": active,
                "ablation_delta_sharpe": delta_sh,
                "ablation_delta_hit": delta_hit,
                "ablation_delta_maxdd": delta_mdd,
                "status": status,
            }
        )

    gov_rows = sorted(
        gov_rows,
        key=lambda x: (
            x["status"] not in {"ablation_harmful", "neutral_redundant"},
            x["status"] != "ablation_harmful",
            -x["mean_abs_dev"],
            -x["active_share"],
        ),
    )
    disable = sorted([r["name"] for r in gov_rows if r.get("status") in {"neutral_redundant", "ablation_harmful"}])

    rt = None
    if "runtime_total_scalar" in trace.columns:
        rt = pd.to_numeric(trace["runtime_total_scalar"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if rt is None or rt.empty:
        rtv = np.ones(1, dtype=float)
    else:
        rtv = rt.values.astype(float)
    p10 = float(np.percentile(rtv, 10))
    p25 = float(np.percentile(rtv, 25))
    p50 = float(np.percentile(rtv, 50))
    # Keep a light anti-collapse floor by default; aggressive floors can
    # suppress valid exposure and flatten Sharpe.
    floor_target = float(np.clip(max(0.05, min(0.10, p10)), 0.05, 0.15))

    return {
        "ok": True,
        "disable_governors": disable,
        "runtime_total_floor": floor_target,
        "runtime_total": {
            "mean": float(np.mean(rtv)),
            "p10": p10,
            "p25": p25,
            "p50": p50,
            "min": float(np.min(rtv)),
            "max": float(np.max(rtv)),
        },
        "ablation_thresholds": {
            "min_sharpe_gain": float(ablation_min_sharpe_gain),
            "max_hit_drop": float(ablation_max_hit_drop),
            "max_mdd_worsen": float(ablation_max_mdd_worsen),
        },
        "governors": gov_rows,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=str(RUNS / "final_governor_trace.csv"))
    ap.add_argument("--out-json", default=str(RUNS / "governor_profile.json"))
    ap.add_argument("--ablation-json", default=str(RUNS / "governor_ablation.json"))
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--neutral-mean-abs", type=float, default=0.005)
    ap.add_argument("--neutral-active-share", type=float, default=0.10)
    ap.add_argument("--ablation-min-sharpe-gain", type=float, default=0.015)
    ap.add_argument("--ablation-max-hit-drop", type=float, default=0.003)
    ap.add_argument("--ablation-max-mdd-worsen", type=float, default=0.003)
    ap.add_argument(
        "--protect",
        default="global_governor,quality_governor,council_gate,turnover_governor,runtime_floor,shock_mask_guard",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    trace_path = Path(args.trace)
    out_path = Path(args.out_json)
    ablation_path = Path(args.ablation_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _parse_csv(trace_path)
    ablation_impacts = _ablation_map_from_summary(ablation_path)
    protected = {s.strip().lower() for s in str(args.protect).split(",") if s.strip()}
    prof = build_governor_profile(
        df,
        eps=float(max(0.0, args.eps)),
        neutral_mean_abs=float(max(0.0, args.neutral_mean_abs)),
        neutral_active_share=float(np.clip(args.neutral_active_share, 0.0, 1.0)),
        protected=protected,
        ablation_impacts=ablation_impacts,
        ablation_min_sharpe_gain=float(max(0.0, args.ablation_min_sharpe_gain)),
        ablation_max_hit_drop=float(np.clip(args.ablation_max_hit_drop, 0.0, 1.0)),
        ablation_max_mdd_worsen=float(max(0.0, args.ablation_max_mdd_worsen)),
    )

    out_path.write_text(json.dumps(prof, indent=2), encoding="utf-8")
    print(f"✅ Wrote {out_path}")
    if not bool(prof.get("ok", False)):
        print("(!) Governor profile unavailable (missing trace).")
        return 0

    disable = prof.get("disable_governors", [])
    rt = prof.get("runtime_total", {})
    print(
        "Profile: disable=%d | runtime_total(mean/p25/min)=%.3f/%.3f/%.3f | runtime_floor=%.3f"
        % (
            len(disable),
                _safe_float(rt.get("mean"), 0.0),
                _safe_float(rt.get("p25"), 0.0),
                _safe_float(rt.get("min"), 0.0),
                _safe_float(prof.get("runtime_total_floor"), 0.10),
            )
        )
    if disable:
        print("Disable candidates:", ", ".join(disable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
