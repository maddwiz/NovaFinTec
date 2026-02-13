#!/usr/bin/env python3
# Reflex Health Gating

import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.reflex_health_index import reflex_health, gate_reflex

RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def load_series(rel):
    p = ROOT / rel
    if not p.exists():
        return None
    try:
        a = np.loadtxt(p, delimiter=",").ravel()
    except Exception:
        try:
            a = np.loadtxt(p, delimiter=",", skiprows=1).ravel()
        except Exception:
            return None
    a = np.asarray(a, float).ravel()
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def first_series(paths):
    for rel in paths:
        a = load_series(rel)
        if a is not None:
            return a
    return None


def load_reflex_signal():
    p = RUNS / "reflexive_signal.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            if "reflex_signal" in df.columns:
                a = pd.to_numeric(df["reflex_signal"], errors="coerce").fillna(0.0).values
                if len(a):
                    return np.asarray(a, float)
        except Exception:
            pass
    p2 = RUNS / "reflex_signal.csv"
    if p2.exists():
        try:
            df = pd.read_csv(p2)
            if "reflex_signal" in df.columns:
                a = pd.to_numeric(df["reflex_signal"], errors="coerce").fillna(0.0).values
                if len(a):
                    return np.asarray(a, float)
        except Exception:
            pass
    return None


def append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        f = ROOT / name
        if not f.exists():
            continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        f.write_text(txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card, encoding="utf-8")


if __name__ == "__main__":
    r = first_series(["runs_plus/reflex_returns.csv"])
    if r is None:
        pred = first_series(["runs_plus/meta_stack_pred.csv", "runs_plus/synapses_pred.csv", "runs_plus/meta_mix.csv"])
        y = first_series(["runs_plus/daily_returns.csv", "daily_returns.csv"])
        if pred is not None and y is not None:
            T = min(len(pred), len(y))
            r = np.asarray(pred[:T] * y[:T], float)
        else:
            r = np.zeros(252, float)

    H = reflex_health(r, lookback=126)
    np.savetxt(RUNS / "reflex_health.csv", H, delimiter=",")

    sig = load_reflex_signal()
    wrote_gate = False
    if sig is not None:
        L = min(len(sig), len(H))
        gated = gate_reflex(sig[:L], H[:L], min_h=0.5)
        np.savetxt(RUNS / "reflex_signal_gated.csv", gated, delimiter=",")
        wrote_gate = True

    msg = "Saved reflex_health.csv" + (" + reflex_signal_gated.csv" if wrote_gate else "")
    append_card("Reflex Health Gating ✔", f"<p>{msg}</p>")
    print("✅", msg)
