#!/usr/bin/env python3
"""
Cost-stress validation on strict OOS metrics.

Recomputes net returns under multiple transaction-cost bps scenarios and
evaluates OOS robustness using the same split logic as strict OOS validation.

Writes:
  - runs_plus/cost_stress_validation.json
  - runs_plus/cost_stress_validation_rows.csv
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from tools.make_daily_from_weights import build_costed_daily_returns
from tools import run_strict_oos_validation as so

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _load_mat(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        a = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _first_mat(paths: list[Path]) -> tuple[np.ndarray | None, str]:
    for p in paths:
        a = _load_mat(p)
        if a is not None:
            return a, str(p.relative_to(ROOT))
    return None, ""


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_bps_list(raw: str) -> list[float]:
    vals = []
    for tok in str(raw).split(","):
        t = str(tok).strip()
        if not t:
            continue
        try:
            vals.append(float(np.clip(float(t), 0.0, 200.0)))
        except Exception:
            continue
    out = sorted(set(round(v, 6) for v in vals))
    return [float(v) for v in out]


def _metrics_for_returns(
    r: np.ndarray,
    *,
    train_frac: float,
    min_train: int,
    min_test: int,
    robust_splits: int,
) -> tuple[dict, dict]:
    T = len(r)
    split = so._build_split_index(T, train_frac, min_train, min_test)
    oos = r[split:]
    net = so._metrics(oos)
    split_ix = so._robust_splits(T, min_train=min_train, min_test=min_test, n_splits=robust_splits)
    ms = [so._metrics(r[s:]) for s in split_ix]
    robust = so._aggregate_robust(ms)
    return net, robust


def _append_card(title: str, html: str) -> None:
    if str(os.getenv("Q_DISABLE_REPORT_CARDS", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        return
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


def main() -> int:
    A = _load_mat(RUNS / "asset_returns.csv")
    W, wsrc = _first_mat(
        [
            RUNS / "portfolio_weights_final.csv",
            RUNS / "tune_best_weights.csv",
            RUNS / "weights_regime.csv",
            RUNS / "weights_tail_blend.csv",
            RUNS / "portfolio_weights.csv",
            ROOT / "portfolio_weights.csv",
        ]
    )
    if A is None or W is None:
        out = {
            "ok": False,
            "reason": "missing_asset_returns_or_weights",
            "asset_returns_found": bool(A is not None),
            "weights_found": bool(W is not None),
        }
        (RUNS / "cost_stress_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"✅ Wrote {RUNS/'cost_stress_validation.json'}")
        print("(!) Cost stress skipped: missing inputs")
        return 0

    if A.shape[1] != W.shape[1]:
        out = {
            "ok": False,
            "reason": "shape_mismatch",
            "asset_returns_shape": list(A.shape),
            "weights_shape": list(W.shape),
        }
        (RUNS / "cost_stress_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"✅ Wrote {RUNS/'cost_stress_validation.json'}")
        print("(!) Cost stress skipped: asset/weight shape mismatch")
        return 0

    T = min(A.shape[0], W.shape[0])
    A = A[:T]
    W = W[:T]

    train_frac = float(np.clip(float(os.getenv("Q_STRICT_OOS_TRAIN_FRAC", "0.75")), 0.50, 0.95))
    min_train = int(np.clip(int(float(os.getenv("Q_STRICT_OOS_MIN_TRAIN", "756"))), 100, 100000))
    min_test = int(np.clip(int(float(os.getenv("Q_STRICT_OOS_MIN_TEST", "252"))), 50, 100000))
    robust_splits = int(np.clip(int(float(os.getenv("Q_STRICT_OOS_ROBUST_SPLITS", "5"))), 1, 16))

    cinfo = _load_json(RUNS / "daily_costs_info.json")
    vol_scaled_bps = float(np.clip(float(os.getenv("Q_COST_VOL_SCALED_BPS", cinfo.get("cost_vol_scaled_bps", 0.0))), 0.0, 100.0))
    vol_lookback = int(np.clip(int(float(os.getenv("Q_COST_VOL_LOOKBACK", cinfo.get("cost_vol_lookback", 20)))), 2, 252))
    vol_ref_daily = float(np.clip(float(os.getenv("Q_COST_VOL_REF_DAILY", cinfo.get("cost_vol_ref_daily", 0.0063))), 1e-5, 0.25))
    half_turnover = str(os.getenv("Q_COST_HALF_TURNOVER", str(cinfo.get("cost_half_turnover", True)))).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    fixed_daily_fee = float(np.clip(float(os.getenv("Q_FIXED_DAILY_FEE", cinfo.get("fixed_daily_fee", 0.0))), 0.0, 1.0))
    cash_yield_annual = float(np.clip(float(os.getenv("Q_CASH_YIELD_ANNUAL", cinfo.get("cash_yield_annual", 0.0))), 0.0, 0.20))
    cash_exposure_target = float(np.clip(float(os.getenv("Q_CASH_EXPOSURE_TARGET", cinfo.get("cash_exposure_target", 1.0))), 0.25, 5.0))

    default_bps = float(np.clip(float(cinfo.get("cost_base_bps", 10.0)), 0.0, 100.0))
    raw_bps = str(os.getenv("Q_COST_STRESS_BPS_LIST", "")).strip()
    bps_list = _parse_bps_list(raw_bps) if raw_bps else sorted(set([default_bps, 15.0, 20.0]))
    if not bps_list:
        bps_list = [default_bps]

    rows = []
    for bps in bps_list:
        net, _gross, cost, _turn, _eff, _carry, _cash_frac = build_costed_daily_returns(
            W,
            A,
            base_bps=float(bps),
            vol_scaled_bps=vol_scaled_bps,
            vol_lookback=vol_lookback,
            vol_ref_daily=vol_ref_daily,
            half_turnover=half_turnover,
            fixed_daily_fee=fixed_daily_fee,
            cash_yield_annual=cash_yield_annual,
            cash_exposure_target=cash_exposure_target,
        )
        m_net, m_robust = _metrics_for_returns(
            net,
            train_frac=train_frac,
            min_train=min_train,
            min_test=min_test,
            robust_splits=robust_splits,
        )
        rows.append(
            {
                "base_bps": float(bps),
                "oos_sharpe": float(m_net.get("sharpe", 0.0)),
                "oos_hit_rate": float(m_net.get("hit_rate", 0.0)),
                "oos_max_drawdown": float(m_net.get("max_drawdown", 0.0)),
                "oos_n": int(m_net.get("n", 0)),
                "oos_robust_sharpe": float(m_robust.get("sharpe", 0.0)),
                "oos_robust_hit_rate": float(m_robust.get("hit_rate", 0.0)),
                "oos_robust_max_drawdown": float(m_robust.get("max_drawdown", 0.0)),
                "oos_robust_n": int(m_robust.get("n", 0)),
                "ann_cost_estimate": float(np.mean(cost) * 252.0) if len(cost) else 0.0,
            }
        )

    worst_robust_sharpe = float(min(r["oos_robust_sharpe"] for r in rows))
    worst_robust_hit = float(min(r["oos_robust_hit_rate"] for r in rows))
    worst_robust_mdd = float(min(r["oos_robust_max_drawdown"] for r in rows))

    min_sh = float(np.clip(float(os.getenv("Q_COST_STRESS_MIN_ROBUST_SHARPE", "0.90")), -2.0, 10.0))
    min_hit = float(np.clip(float(os.getenv("Q_COST_STRESS_MIN_ROBUST_HIT", "0.48")), 0.0, 1.0))
    max_abs_mdd = float(np.clip(float(os.getenv("Q_COST_STRESS_MAX_ABS_MDD", "0.12")), 0.001, 2.0))

    reasons = []
    if worst_robust_sharpe < min_sh:
        reasons.append(f"cost_stress_robust_sharpe<{min_sh:.2f} ({worst_robust_sharpe:.3f})")
    if worst_robust_hit < min_hit:
        reasons.append(f"cost_stress_robust_hit<{min_hit:.2f} ({worst_robust_hit:.3f})")
    if abs(worst_robust_mdd) > max_abs_mdd:
        reasons.append(f"cost_stress_robust_abs_mdd>{max_abs_mdd:.3f} ({abs(worst_robust_mdd):.3f})")
    ok = len(reasons) == 0

    cols = [
        "base_bps",
        "oos_sharpe",
        "oos_hit_rate",
        "oos_max_drawdown",
        "oos_n",
        "oos_robust_sharpe",
        "oos_robust_hit_rate",
        "oos_robust_max_drawdown",
        "oos_robust_n",
        "ann_cost_estimate",
    ]
    with (RUNS / "cost_stress_validation_rows.csv").open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=cols)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow({k: r.get(k, "") for k in cols})

    out = {
        "ok": bool(ok),
        "weights_source": wsrc,
        "rows_total": int(T),
        "thresholds": {
            "min_robust_sharpe": float(min_sh),
            "min_robust_hit_rate": float(min_hit),
            "max_abs_robust_mdd": float(max_abs_mdd),
        },
        "settings": {
            "bps_list": [float(x) for x in bps_list],
            "vol_scaled_bps": float(vol_scaled_bps),
            "vol_lookback": int(vol_lookback),
            "vol_ref_daily": float(vol_ref_daily),
            "half_turnover": bool(half_turnover),
            "fixed_daily_fee": float(fixed_daily_fee),
            "cash_yield_annual": float(cash_yield_annual),
            "cash_exposure_target": float(cash_exposure_target),
        },
        "worst_case_robust": {
            "sharpe": float(worst_robust_sharpe),
            "hit_rate": float(worst_robust_hit),
            "max_drawdown": float(worst_robust_mdd),
        },
        "scenarios": rows,
        "reasons": reasons,
    }
    (RUNS / "cost_stress_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    badge = "PASS" if ok else "FAIL"
    color = "#1b8f3a" if ok else "#a91d2b"
    html = (
        f"<p><b style='color:{color}'>Cost stress {badge}</b> across bps={','.join(str(int(x)) for x in bps_list)}.</p>"
        f"<p>Worst robust: Sharpe={worst_robust_sharpe:.3f}, Hit={worst_robust_hit:.3f}, "
        f"MaxDD={worst_robust_mdd:.3f}.</p>"
    )
    if reasons:
        html += f"<p>Reasons: {', '.join(reasons)}</p>"
    _append_card("Cost Stress Validation ✔", html)

    print(f"✅ Wrote {RUNS/'cost_stress_validation_rows.csv'}")
    print(f"✅ Wrote {RUNS/'cost_stress_validation.json'}")
    print(
        f"Cost stress: {'PASS' if ok else 'FAIL'} | "
        f"worst_robust_sh={worst_robust_sharpe:.3f} worst_robust_hit={worst_robust_hit:.3f} "
        f"worst_robust_mdd={worst_robust_mdd:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
