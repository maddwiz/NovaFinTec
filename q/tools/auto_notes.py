from pathlib import Path
import numpy as np, pandas as pd
from qmods.io import load_close
from qmods.drift import rolling_dna_drift

DATA = Path("data")
OUT  = Path("runs_plus")
OUT.mkdir(exist_ok=True)

for p in DATA.glob("*.csv"):
    a = p.stem
    try:
        s = load_close(p)
    except Exception:
        continue
    d = rolling_dna_drift(s, 126).dropna()
    if d.empty: continue
    spikes = d[d > (d.mean() + 2.5*d.std())]
    note = "None"
    if len(spikes):
        when = spikes.index[-1].strftime("%Y-%m-%d")
        note = f"Recent DNA-drift spike on {when} (z~{((spikes.iloc[-1]-d.mean())/d.std()):.2f})."
    ndir = OUT/a
    ndir.mkdir(exist_ok=True, parents=True)
    (ndir/"notes.txt").write_text(note)
    print(a, "→", note)
