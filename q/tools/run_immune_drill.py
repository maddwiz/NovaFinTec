#!/usr/bin/env python3
# Governance immune drill: stress-test current portfolio against synthetic shock regimes.
#
# Writes:
#   runs_plus/immune_drill.json

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.immune_drill import run_scenarios  # noqa: E402

RUNS = ROOT / "runs_plus"
DATA = ROOT / "data"
RUNS.mkdir(exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_weights(path: Path):
    if not path.exists():
        return None
    try:
        w = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            w = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    w = np.asarray(w, float)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    return w


def _load_asset_returns_from_data():
    frames = []
    for p in sorted(DATA.glob("*.csv")):
        sym = p.stem.replace("_prices", "").upper().strip()
        if not sym:
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        dcol = None
        for c in ["DATE", "Date", "date", "timestamp", "Timestamp"]:
            if c in df.columns:
                dcol = c
                break
        pcol = None
        for c in ["Adj Close", "adj_close", "AdjClose", "Close", "close", "price", "Price"]:
            if c in df.columns:
                pcol = c
                break
        if dcol is None or pcol is None:
            continue
        d = pd.DataFrame({"DATE": pd.to_datetime(df[dcol], errors="coerce"), sym: pd.to_numeric(df[pcol], errors="coerce")})
        d = d.dropna(subset=["DATE"]).sort_values("DATE")
        d[sym] = d[sym].pct_change()
        d["DATE"] = d["DATE"].dt.normalize()
        frames.append(d[["DATE", sym]])
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="DATE", how="outer")
    out = out.sort_values("DATE").reset_index(drop=True).fillna(0.0)
    return out.drop(columns=["DATE"], errors="ignore").values


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
    W = _load_weights(RUNS / "portfolio_weights_final.csv")
    A = _load_asset_returns_from_data()
    if W is None or A is None:
        out = {
            "timestamp_utc": _now_iso(),
            "ok": False,
            "error": "missing_weights_or_asset_returns",
        }
        (RUNS / "immune_drill.json").write_text(json.dumps(out, indent=2))
        print(f"✅ Wrote {RUNS/'immune_drill.json'}")
        raise SystemExit(0)

    result = run_scenarios(W=W, A=A)
    out = {
        "timestamp_utc": _now_iso(),
        "ok": True,
        "pass": bool(result["summary"]["pass"]),
        "summary": result["summary"],
        "scenarios": result["scenarios"],
        "shape": {
            "weights_rows": int(W.shape[0]),
            "weights_cols": int(W.shape[1]),
            "asset_returns_rows": int(A.shape[0]),
            "asset_returns_cols": int(A.shape[1]),
        },
    }
    (RUNS / "immune_drill.json").write_text(json.dumps(out, indent=2))

    s = out["summary"]
    _append_card(
        "Immune Drill ✔",
        (
            f"<p>pass={out['pass']}, worst DD={s['worst_max_dd']:.3f}, "
            f"worst tail p01={s['worst_tail_p01']:.3f}, worst Sharpe={s['worst_sharpe']:.3f}</p>"
        ),
    )
    print(f"✅ Wrote {RUNS/'immune_drill.json'}")
