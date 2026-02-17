#!/usr/bin/env python3
"""
Promotion gate for Q -> AION overlay promotion.

Reads:
  - runs_plus/strict_oos_validation.json

Writes:
  - runs_plus/q_promotion_gate.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _append_card(title: str, html: str) -> None:
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


def _select_metrics(val: dict, mode: str) -> tuple[str, dict]:
    net = val.get("metrics_oos_net", {}) if isinstance(val.get("metrics_oos_net"), dict) else {}
    robust = val.get("metrics_oos_robust", {}) if isinstance(val.get("metrics_oos_robust"), dict) else {}
    m = str(mode).strip().lower()
    if m == "robust":
        return "metrics_oos_robust", robust
    if m == "single":
        return "metrics_oos_net", net
    if robust:
        return "metrics_oos_robust", robust
    return "metrics_oos_net", net


def main() -> int:
    src = RUNS / "strict_oos_validation.json"
    val = _load_json(src)
    if not isinstance(val, dict):
        out = {
            "ok": False,
            "source": str(src),
            "reason": "missing_strict_oos_validation",
            "reasons": ["strict_oos_validation_missing"],
        }
        (RUNS / "q_promotion_gate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"✅ Wrote {RUNS/'q_promotion_gate.json'}")
        print("Promotion gate: FAIL (missing strict OOS validation)")
        return 0

    metric_mode = str(os.getenv("Q_PROMOTION_OOS_MODE", "robust_then_single")).strip().lower()
    metric_source, m = _select_metrics(val, metric_mode)
    sharpe = float(m.get("sharpe", 0.0))
    hit = float(m.get("hit_rate", 0.0))
    mdd = float(m.get("max_drawdown", 0.0))
    n = int(m.get("n", 0))

    min_sharpe = float(np.clip(float(os.getenv("Q_PROMOTION_MIN_OOS_SHARPE", "1.00")), -2.0, 10.0))
    min_hit = float(np.clip(float(os.getenv("Q_PROMOTION_MIN_OOS_HIT", "0.49")), 0.0, 1.0))
    max_abs_mdd = float(np.clip(float(os.getenv("Q_PROMOTION_MAX_ABS_MDD", "0.10")), 0.001, 2.0))
    min_n = int(np.clip(int(float(os.getenv("Q_PROMOTION_MIN_OOS_SAMPLES", "252"))), 1, 1000000))

    reasons = []
    if n < min_n:
        reasons.append(f"oos_samples<{min_n} ({n})")
    if sharpe < min_sharpe:
        reasons.append(f"oos_sharpe<{min_sharpe:.2f} ({sharpe:.3f})")
    if hit < min_hit:
        reasons.append(f"oos_hit<{min_hit:.2f} ({hit:.3f})")
    if abs(mdd) > max_abs_mdd:
        reasons.append(f"oos_abs_mdd>{max_abs_mdd:.3f} ({abs(mdd):.3f})")

    ok = len(reasons) == 0
    out = {
        "ok": bool(ok),
        "source": str(src),
        "metric_source": metric_source,
        "metric_mode": metric_mode,
        "thresholds": {
            "min_oos_sharpe": float(min_sharpe),
            "min_oos_hit": float(min_hit),
            "max_abs_mdd": float(max_abs_mdd),
            "min_oos_samples": int(min_n),
        },
        "metrics_oos_net": {
            "sharpe": float(sharpe),
            "hit_rate": float(hit),
            "max_drawdown": float(mdd),
            "n": int(n),
        },
        "reasons": reasons,
    }
    (RUNS / "q_promotion_gate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    badge = "PASS" if ok else "FAIL"
    color = "#1b8f3a" if ok else "#a91d2b"
    html = (
        f"<p><b style='color:{color}'>Promotion {badge}</b> "
        f"(OOS Sharpe={sharpe:.3f}, Hit={hit:.3f}, MaxDD={mdd:.3f}, N={n}, "
        f"source={metric_source}).</p>"
    )
    if reasons:
        html += f"<p>Reasons: {', '.join(reasons)}</p>"
    _append_card("Q Promotion Gate ✔", html)

    print(f"✅ Wrote {RUNS/'q_promotion_gate.json'}")
    print(f"Promotion gate: {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
