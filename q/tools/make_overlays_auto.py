import pathlib, itertools
from qmods.io import load_close

DATA = pathlib.Path("data")
OUT  = pathlib.Path("runs_plus")

assets = [p.stem for p in DATA.glob("*.csv")]
assets = sorted(set(assets))
pairs = list(itertools.combinations(assets[:6], 2))[:4]

from qmods.overlay import save_overlay_gif

for a, b in pairs:
    try:
        ca = load_close(DATA/f"{a}.csv").to_numpy()
        cb = load_close(DATA/f"{b}.csv").to_numpy()
    except Exception as e:
        print(f"skip {a}__{b} ({e})"); continue
    out_dir = OUT/f"{a}__{b}"
    save_overlay_gif(ca, cb, out_dir, name="overlay", frames=80, step=5, fps=12)
    print(f"Wrote overlay for {a}__{b} → {out_dir}/overlay.gif")
