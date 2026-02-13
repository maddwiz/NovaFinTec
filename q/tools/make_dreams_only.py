# tools/make_dreams_only.py — force-create dream.gif and dream.mp4 for each price asset
from pathlib import Path
from qmods.io import load_close
from qmods.dreams import save_dream_gif, save_dream_mp4

DATA = Path("data")
RUNS = Path("runs_plus")
RUNS.mkdir(exist_ok=True)

assets = sorted([p.stem for p in DATA.glob("*.csv")])
for a in assets:
    # skip obvious non-price files by name
    if a.lower() in {"news"}:
        print("· skipping non-price CSV:", a)
        continue
    try:
        s = load_close(DATA/f"{a}.csv")
        if s is None or len(s) == 0:
            print("· skipping (no price series):", a)
            continue
        outdir = RUNS/a
        outdir.mkdir(exist_ok=True, parents=True)
        save_dream_gif(s.values, outdir/"dream.gif", frames=120)
        save_dream_mp4(s.values, outdir/"dream.mp4", frames=180)
        print("✓", a, "dreams written")
    except Exception:
        # silently ignore non-price formats
        print("· skipping (not a price CSV):", a)
        continue
