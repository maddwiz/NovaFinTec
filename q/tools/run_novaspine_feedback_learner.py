#!/usr/bin/env python3
"""
NovaSpine feedback learner: convert AION telemetry into Bayesian priors.

Writes:
  - runs_plus/novaspine_signal_priors.json
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)
AION_STATE = ROOT.parent / "aion" / "state"


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


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _extract_signal_names(rec: dict) -> list[str]:
    out: list[str] = []
    cs = rec.get("category_scores")
    if isinstance(cs, dict):
        for k, v in cs.items():
            try:
                if float(v) > 0.55:
                    out.append(str(k).strip().lower())
            except Exception:
                continue
    reasons = rec.get("reasons")
    if isinstance(reasons, list):
        for r in reasons:
            s = str(r).strip().lower()
            if not s:
                continue
            if "pattern" in s:
                out.append("pattern_confluence")
            elif "vwap" in s or "value area" in s or "poc" in s:
                out.append("key_levels")
            elif "timeframe" in s:
                out.append("multi_timeframe")
            elif "volume" in s or "momentum" in s:
                out.append("volume_momentum")
            elif "q overlay" in s:
                out.append("q_overlay")
    if not out:
        out.append("price")
    dedup = []
    seen = set()
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


def _extract_outcome(rec: dict) -> int | None:
    for k in ["pnl_realized", "pnl", "realized_pnl", "net_pnl"]:
        if k in rec:
            try:
                v = float(rec.get(k))
            except Exception:
                continue
            if np.isfinite(v):
                return 1 if v > 0 else 0
    if "r_captured" in rec:
        try:
            v = float(rec.get("r_captured"))
            if np.isfinite(v):
                return 1 if v > 0 else 0
        except Exception:
            pass
    return None


def _extract_context(rec: dict) -> tuple[str, str]:
    regime = str(rec.get("regime", rec.get("session_type", "unknown"))).strip().lower() or "unknown"
    phase = str(rec.get("session_phase", rec.get("phase", "unknown"))).strip().lower() or "unknown"
    return regime, phase


def main() -> int:
    paths = [AION_STATE / "skimmer_decisions.jsonl", AION_STATE / "trade_decisions.jsonl"]
    records = []
    for p in paths:
        records.extend(list(_iter_jsonl(p) or []))

    signal_wins = defaultdict(int)
    signal_losses = defaultdict(int)

    context_wins = defaultdict(int)
    context_losses = defaultdict(int)

    usable = 0
    for rec in records:
        outcome = _extract_outcome(rec)
        if outcome is None:
            continue
        sigs = _extract_signal_names(rec)
        regime, phase = _extract_context(rec)

        usable += 1
        for s in sigs:
            if outcome > 0:
                signal_wins[s] += 1
            else:
                signal_losses[s] += 1

            key = f"{s}|{regime}|{phase}"
            if outcome > 0:
                context_wins[key] += 1
            else:
                context_losses[key] += 1

    prior_strength = float(np.clip(float(os.getenv("Q_NOVASPINE_PRIOR_STRENGTH", "1.0")), 0.1, 10.0))

    signal_priors = {}
    for s in sorted(set(signal_wins.keys()) | set(signal_losses.keys())):
        w = int(signal_wins[s])
        l = int(signal_losses[s])
        signal_priors[s] = {
            "alpha": float(2.0 + prior_strength * w),
            "beta": float(2.0 + prior_strength * l),
            "wins": int(w),
            "losses": int(l),
            "hit_rate": float(w / max(1, w + l)),
        }

    context_priors = {}
    for k in sorted(set(context_wins.keys()) | set(context_losses.keys())):
        w = int(context_wins[k])
        l = int(context_losses[k])
        context_priors[k] = {
            "alpha": float(2.0 + prior_strength * w),
            "beta": float(2.0 + prior_strength * l),
            "wins": int(w),
            "losses": int(l),
            "hit_rate": float(w / max(1, w + l)),
        }

    payload = {
        "ok": True,
        "source_files": [str(p) for p in paths if p.exists()],
        "records_total": int(len(records)),
        "records_usable": int(usable),
        "prior_strength": float(prior_strength),
        "signal_priors": signal_priors,
        "context_priors": context_priors,
    }

    out = RUNS / "novaspine_signal_priors.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _append_card(
        "NovaSpine Feedback Priors ✔",
        (
            f"<p>usable_records={usable}, signal_priors={len(signal_priors)}, context_priors={len(context_priors)}</p>"
        ),
    )

    print(f"✅ Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
