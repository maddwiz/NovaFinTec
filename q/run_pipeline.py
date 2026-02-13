import argparse, pathlib, json
import pandas as pd, numpy as np
from qmods.meta_council import meta_council
from qmods.dna import fft_topk_dna
from qmods.heartbeat import heartbeat_bpm
from qmods.drift import rolling_dna_drift
from qmods.dreams import save_dream_png
from qmods.log import append_growth_log

def base_metrics(close: pd.Series):
    r = close.pct_change().dropna()
    hit = (np.sign(r) == np.sign(r.shift(-1))).mean()
    sh = (r.mean() / (r.std() + 1e-12)) * np.sqrt(252) if r.std() else 0.0
    eq = (1 + r).cumprod()
    mdd = float((eq/eq.cummax() - 1).min())
    return float(hit), float(sh), mdd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pathlib.Path(args.data)/args.asset, parse_dates=["Date"], index_col="Date")
    close = df["Close"].astype(float)

    # 1) Meta council signal (stub)
    meta = meta_council(close)

    # 2) DNA + drift + heartbeat
    dna = fft_topk_dna(close.values)
    drift = rolling_dna_drift(close, 126)
    bpm = heartbeat_bpm(close)

    # 3) Base metrics (for now: price-only baseline)
    hit, sh, mdd = base_metrics(close)

    # 4) Dream (stub image)
    save_dream_png(close.values, outdir/"dream.png")

    # 5) Save artifacts
    result = {
        "asset": args.asset,
        "hit_rate": hit,
        "sharpe": sh,
        "max_dd": mdd,
        "dna": dna,
        "dna_drift_pct": float(drift.ffill().iloc[-1]) if drift.notna().any() else None,
        "heartbeat_bpm_latest": float(bpm.ffill().iloc[-1]) if bpm.notna().any() else None,
    }
    (outdir/"summary.json").write_text(json.dumps(result, indent=2))

    # 6) Growth log append
    append_growth_log(result, pathlib.Path("GROWTH_LOG.md"))

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
