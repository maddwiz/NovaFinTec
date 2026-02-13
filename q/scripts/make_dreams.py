
#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import pandas as pd
from qengine.data import load_csv
from qengine.signals import dna_signal
from qengine.dreams import dream_from_dna

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    df = load_csv(os.path.join(args.data, args.asset))
    _, drift, _ = dna_signal(df["Close"])
    img = dream_from_dna(drift.values, width=640, height=320, seed=0)
    img.save(outdir/f"{Path(args.asset).stem}_dream.png")
    print(f"Saved dream to {outdir}/{Path(args.asset).stem}_dream.png")

if __name__ == "__main__":
    main()
