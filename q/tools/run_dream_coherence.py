#!/usr/bin/env python3
# Dream/Reflex/Symbolic coherence governor.
#
# Reads:
#   runs_plus/reflex_latent.csv
#   runs_plus/symbolic_latent.csv
#   runs_plus/meta_mix.csv
#   runs_plus/synapses_pred.csv
#   runs_plus/meta_stack_pred.csv
#   runs_plus/heartbeat_exposure_scaler.csv
#   runs_plus/daily_returns.csv
# Writes:
#   runs_plus/dream_coherence_governor.csv
#   runs_plus/dream_coherence_components.csv
#   runs_plus/dream_coherence_info.json

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.dream_coherence import build_dream_coherence_governor

RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _load_series(path: Path, candidates: list[str] | None = None):
    if not path.exists():
        return None
    candidates = candidates or []
    try:
        df = pd.read_csv(path)
    except Exception:
        df = None

    if df is not None and len(df.columns) >= 1:
        cols = {str(c).lower(): str(c) for c in df.columns}
        dcol = cols.get("date") or cols.get("timestamp") or cols.get("time")
        vcol = None
        for c in candidates:
            if c in cols:
                vcol = cols[c]
                break
        if vcol is None:
            nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if nums:
                vcol = str(nums[-1])
        if vcol is not None:
            v = pd.to_numeric(df[vcol], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if dcol is not None:
                d = pd.to_datetime(df[dcol], errors="coerce")
                m = d.notna()
                if int(m.sum()) >= max(5, int(0.5 * len(df))):
                    s = pd.Series(v[m].values.astype(float), index=d[m].values)
                    s = s.sort_index()
                    return s
            return pd.Series(v.values.astype(float))

    try:
        a = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    a = np.asarray(a, float)
    if a.ndim == 2 and a.shape[1] >= 1:
        a = a[:, -1]
    return pd.Series(np.nan_to_num(a.ravel(), nan=0.0, posinf=0.0, neginf=0.0))


def _align_tail(signals: dict[str, pd.Series], returns: pd.Series):
    if returns is None or len(returns) == 0:
        return {}, np.zeros(0, float), None
    keep = {k: v for k, v in signals.items() if v is not None and len(v) > 0}
    if not keep:
        return {}, np.asarray(returns.values, float), returns.index if hasattr(returns, "index") else None

    # If all are date-indexed, align on intersection for best fidelity.
    all_date = hasattr(returns.index, "dtype") and np.issubdtype(returns.index.dtype, np.datetime64)
    all_date = all_date and all(
        hasattr(v.index, "dtype") and np.issubdtype(v.index.dtype, np.datetime64) for v in keep.values()
    )
    if all_date:
        idx = returns.index
        for s in keep.values():
            idx = idx.intersection(s.index)
        idx = idx.sort_values()
        if len(idx) >= 30:
            out = {k: np.asarray(v.reindex(idx).ffill().fillna(0.0).values, float) for k, v in keep.items()}
            r = np.asarray(returns.reindex(idx).ffill().fillna(0.0).values, float)
            return out, r, idx

    # Fallback: tail alignment by shortest length.
    L = min([len(returns)] + [len(v) for v in keep.values()])
    if L <= 0:
        return {}, np.zeros(0, float), None
    out = {k: np.asarray(v.values[-L:], float) for k, v in keep.items()}
    r = np.asarray(returns.values[-L:], float)
    idx = returns.index[-L:] if hasattr(returns, "index") else None
    return out, r, idx


def _append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    returns = _load_series(RUNS / "daily_returns.csv")
    if returns is None or len(returns) == 0:
        print("(!) Missing runs_plus/daily_returns.csv; cannot build dream coherence governor.")
        raise SystemExit(0)

    sig = {
        "reflex_latent": _load_series(RUNS / "reflex_latent.csv", ["reflex_latent"]),
        "symbolic_latent": _load_series(RUNS / "symbolic_latent.csv", ["symbolic_latent"]),
        "meta_mix": _load_series(RUNS / "meta_mix.csv", ["meta_mix"]),
        "synapses_pred": _load_series(RUNS / "synapses_pred.csv", ["synapses_pred"]),
        "meta_stack_pred": _load_series(RUNS / "meta_stack_pred.csv", ["meta_stack_pred"]),
        "heartbeat_exposure": _load_series(RUNS / "heartbeat_exposure_scaler.csv", ["heartbeat_exposure_scaler"]),
    }

    aligned, ret, idx = _align_tail(sig, returns)
    gov, info = build_dream_coherence_governor(aligned, ret, lo=0.70, hi=1.15, smooth=0.88)
    if len(gov) == 0:
        print("(!) Dream coherence alignment failed; skipping.")
        raise SystemExit(0)

    comp = pd.DataFrame(
        {
            "dream_coherence_governor": gov,
        }
    )
    if idx is not None and len(idx) == len(comp):
        comp.insert(0, "DATE", pd.to_datetime(idx).strftime("%Y-%m-%d"))
    comp.to_csv(RUNS / "dream_coherence_governor.csv", index=False)

    comp_info = {
        "status": info.get("status", "na"),
        "signals": info.get("signals", []),
        "length": int(info.get("length", len(gov))),
        "mean_agreement": float(info.get("mean_agreement", 0.0)),
        "mean_efficacy": float(info.get("mean_efficacy", 0.0)),
        "mean_stability": float(info.get("mean_stability", 0.0)),
        "mean_coherence": float(info.get("mean_coherence", 0.0)),
        "mean_governor": float(info.get("mean_governor", float(np.mean(gov)))),
        "min_governor": float(info.get("min_governor", float(np.min(gov)))),
        "max_governor": float(info.get("max_governor", float(np.max(gov)))),
        "per_signal_consensus_corr": info.get("per_signal_consensus_corr", {}),
    }
    (RUNS / "dream_coherence_info.json").write_text(json.dumps(comp_info, indent=2))

    # richer component view (for debugging + report)
    more = pd.DataFrame({"dream_coherence_governor": gov})
    if idx is not None and len(idx) == len(more):
        more.insert(0, "DATE", pd.to_datetime(idx).strftime("%Y-%m-%d"))
    more.to_csv(RUNS / "dream_coherence_components.csv", index=False)

    _append_card(
        "Dream Coherence Governor ✔",
        (
            f"<p>signals={len(comp_info['signals'])}, rows={comp_info['length']}, "
            f"coherence={comp_info['mean_coherence']:.3f}</p>"
            f"<p>governor mean={comp_info['mean_governor']:.3f}, "
            f"min={comp_info['min_governor']:.3f}, max={comp_info['max_governor']:.3f}</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'dream_coherence_governor.csv'}")
    print(f"✅ Wrote {RUNS/'dream_coherence_components.csv'}")
    print(f"✅ Wrote {RUNS/'dream_coherence_info.json'}")
