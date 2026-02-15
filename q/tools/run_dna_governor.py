#!/usr/bin/env python3
# Build DNA stress governor for runtime risk control.
#
# Reads:
#   runs_plus/dna_drift.csv (preferred)
#   runs_plus/daily_returns.csv (fallback)
# Writes:
#   runs_plus/dna_stress_governor.csv
#   runs_plus/dna_stress_info.json
#   runs_plus/dna_drift.csv (fallback build, if missing)

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qengine.dna import drift_regime_flags, drift_velocity, rolling_drift
from qmods.dna_governor import build_dna_stress_governor

RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _load_dna_frame():
    p = RUNS / "dna_drift.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            if len(df) and "dna_drift" in df.columns:
                return df, "dna_drift.csv"
        except Exception:
            pass
    return None, None


def _load_returns():
    p = RUNS / "daily_returns.csv"
    if not p.exists():
        return None
    try:
        a = np.loadtxt(p, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(p, delimiter=",", skiprows=1)
        except Exception:
            return None
    a = np.asarray(a, float)
    if a.ndim == 2 and a.shape[1] >= 1:
        a = a[:, -1]
    a = np.nan_to_num(a.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    return a if len(a) else None


def _build_dna_from_returns():
    r = _load_returns()
    if r is None or len(r) < 64:
        return None, None
    d = rolling_drift(r, window=64, topk=32, smooth_span=5)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    v = drift_velocity(d)
    z, st = drift_regime_flags(d, z_win=63, hi=1.25, lo=-1.25)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(d), freq="B")
    out = pd.DataFrame(
        {
            "DATE": idx.strftime("%Y-%m-%d"),
            "dna_drift": d,
            "dna_velocity": np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0),
            "dna_drift_z": np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0),
            "dna_regime_state": np.nan_to_num(st, nan=0.0, posinf=0.0, neginf=0.0),
        }
    )
    out.to_csv(RUNS / "dna_drift.csv", index=False)
    return out, "daily_returns_fallback"


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
    df, source = _load_dna_frame()
    if df is None:
        df, source = _build_dna_from_returns()
    if df is None:
        print("(!) Missing DNA drift and returns fallback; skipping DNA governor.")
        raise SystemExit(0)

    drift = pd.to_numeric(df.get("dna_drift", 0.0), errors="coerce").fillna(0.0).values
    vel = pd.to_numeric(df.get("dna_velocity", 0.0), errors="coerce").fillna(0.0).values
    acc_col = df["dna_acceleration"] if "dna_acceleration" in df.columns else pd.Series(np.nan, index=df.index)
    acc = pd.to_numeric(acc_col, errors="coerce").values
    if np.isnan(acc).all():
        acc = np.gradient(vel) if len(vel) else np.zeros_like(vel)
    acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
    z = pd.to_numeric(df.get("dna_drift_z", 0.0), errors="coerce").fillna(0.0).values
    st = pd.to_numeric(df.get("dna_regime_state", 0.0), errors="coerce").fillna(0.0).values

    ret = _load_returns()
    stress, gov, info = build_dna_stress_governor(
        drift,
        vel,
        acc,
        z,
        st,
        returns=ret,
        lo=0.72,
        hi=1.12,
        smooth=0.88,
    )
    if len(gov) == 0:
        print("(!) DNA governor empty; skipping.")
        raise SystemExit(0)

    out = pd.DataFrame({"dna_stress": stress, "dna_stress_governor": gov})
    if "DATE" in df.columns and len(df["DATE"]) >= len(out):
        out.insert(0, "DATE", pd.to_datetime(df["DATE"], errors="coerce").astype(str).values[: len(out)])
    out.to_csv(RUNS / "dna_stress_governor.csv", index=False)
    comp = pd.DataFrame(
        {
            "dna_drift": drift[: len(stress)],
            "dna_velocity": vel[: len(stress)],
            "dna_acceleration": acc[: len(stress)],
            "dna_drift_z": z[: len(stress)],
            "dna_regime_state": st[: len(stress)],
            "dna_stress": stress,
            "dna_stress_governor": gov,
        }
    )
    if "DATE" in df.columns and len(df["DATE"]) >= len(comp):
        comp.insert(0, "DATE", pd.to_datetime(df["DATE"], errors="coerce").astype(str).values[: len(comp)])
    comp.to_csv(RUNS / "dna_stress_components.csv", index=False)

    meta = {
        **info,
        "source": source,
        "drift_rows": int(len(df)),
        "governor_file": str(RUNS / "dna_stress_governor.csv"),
        "components_file": str(RUNS / "dna_stress_components.csv"),
    }
    (RUNS / "dna_stress_info.json").write_text(json.dumps(meta, indent=2))

    _append_card(
        "DNA Stress Governor ✔",
        (
            f"<p>source={source}, rows={meta['length']}, mean stress={meta['mean_stress']:.3f}</p>"
            f"<p>governor mean={meta['mean_governor']:.3f}, "
            f"min={meta['min_governor']:.3f}, max={meta['max_governor']:.3f}</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'dna_stress_governor.csv'}")
    print(f"✅ Wrote {RUNS/'dna_stress_info.json'}")
