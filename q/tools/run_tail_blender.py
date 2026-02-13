#!/usr/bin/env python3
# Tail-Blender: blend hedges into base weights
# Reads:  portfolio_weights.csv  OR  runs_plus/portfolio_weights.csv
#         runs_plus/hedge_*.csv  (each [T,N] hedge path)  OR uses DD-scaled as fallback
# Writes: runs_plus/weights_tail_blend.csv
# Appends a small card to report_*.

import glob
import numpy as np
from pathlib import Path
from qmods.tail_blender_v1 import tail_blend

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"; RUNS.mkdir(exist_ok=True)

def maybe_load(path: Path):
    if path.exists():
        # try no-header first, then with header
        try:
            arr = np.loadtxt(path, delimiter=",")
            if arr.ndim == 1: arr = arr.reshape(-1, 1)
            return arr
        except Exception:
            try:
                arr = np.loadtxt(path, delimiter=",", skiprows=1)
                if arr.ndim == 1: arr = arr.reshape(-1, 1)
                return arr
            except Exception:
                return None
    return None

def load_first_non_none(paths):
    for p in paths:
        a = maybe_load(p)
        if a is not None:
            return a
    return None

def append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html"]:
        p = ROOT / name
        if not p.exists(): 
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")
        print(f"✅ Appended card to {name}")

if __name__ == "__main__":
    # ---- 1) Load BASE weights safely (no "or" on arrays) ----
    base = load_first_non_none([
        ROOT / "portfolio_weights.csv",
        RUNS / "portfolio_weights.csv",
    ])
    if base is None:
        print("(!) No portfolio_weights found. Skipping Tail-Blender.")
        raise SystemExit(0)

    # ---- 2) Load hedges (if any) ----
    hedge_files = sorted(glob.glob(str(RUNS / "hedge_*.csv")))
    hedges = []
    for hf in hedge_files:
        h = maybe_load(Path(hf))
        if h is not None:
            hedges.append(h)

    # Fallback: if no hedges, try DD-scaled as a “hedge”
    if not hedges:
        dd = maybe_load(RUNS / "weights_dd_scaled.csv")
        if dd is not None and dd.shape == base.shape:
            hedges = [dd]
            weights = [0.20]  # 20% budget to DD-scaled
        else:
            # Nothing to blend → just write base back so pipeline continues
            outp = RUNS / "weights_tail_blend.csv"
            np.savetxt(outp, base, delimiter=",")
            append_card("Tail-Blender (no hedges)", "<p>No hedges found; wrote base to weights_tail_blend.csv</p>")
            print(f"✅ Wrote {outp} (base only)")
            raise SystemExit(0)
    else:
        # Default hedge budget: up to 0.30 total, split equally
        H = len(hedges)
        total = min(0.30, 0.10 * H)
        weights = [total / H] * H

    # ---- 3) Align shapes (truncate to shortest T) ----
    T = min([base.shape[0]] + [h.shape[0] for h in hedges])
    N = base.shape[1]
    base = base[:T]
    hedges = [h[:T] for h in hedges]
    # Ensure hedge shapes match N
    hedges = [h if h.shape[1] == N else np.tile(h[:, :1], (1, N)) for h in hedges]

    # ---- 4) Blend ----
    blended = tail_blend(base, hedges, weights)

    # ---- 5) Save + report ----
    outp = RUNS / "weights_tail_blend.csv"
    np.savetxt(outp, blended, delimiter=",")
    html = f"<p>Blended {len(hedges)} hedge(s) into base. Budget={sum(weights):.2f}. Saved weights_tail_blend.csv (T={T}, N={N}).</p>"
    append_card("Tail-Blender ✔", html)
    print(f"✅ Wrote {outp}")
