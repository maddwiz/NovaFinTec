#!/usr/bin/env python3
import argparse, os, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    summary = json.load(open(run/"summary.json"))
    metrics = pd.read_csv(run/"metrics_per_fold.csv")
    eq = pd.read_csv(run/"equity_curve.csv")
    cards = None
    try:
        cards = pd.read_csv(run/"explain_cards.csv")
    except Exception:
        cards = None

    plt.figure()
    plt.plot(eq["equity"].values)
    plt.title("Equity Curve")
    plt.savefig(out/"equity_curve.png", bbox_inches="tight")
    plt.close()

    if "dna_drift" in cards.columns:
        plt.figure()
        cards["dna_drift"].plot()
        plt.title("DNA Drift (Test Period)")
        plt.savefig(out/"dna_drift.png", bbox_inches="tight")
        plt.close()

    html = []
    html.append("<html><head><meta charset='utf-8'><title>Q v2.5 Foundations Report</title></head><body>")
    html.append("<h1>Q v2.5 Foundations — Report</h1>")
    html.append(f"<p><b>Asset:</b> {summary['asset']}<br>")
    html.append(f"<b>Hit Rate:</b> {summary['hit_rate']:.3f} &nbsp; ")
    html.append(f"<b>Sharpe:</b> {summary['sharpe']:.3f} &nbsp; ")
    html.append(f"<b>Max Drawdown:</b> {summary['max_dd']:.3f}</p>")

    html.append("<h2>Equity Curve</h2>")
    html.append("<img src='equity_curve.png' width='640'>")

    if (out/"dna_drift.png").exists():
        html.append("<h2>DNA Drift</h2>")
        html.append("<img src='dna_drift.png' width='640'>")

    html.append("<h2>First 15 Explainability Rows</h2>")
    html.append(cards.head(15).to_html(index=False) if cards is not None else "<p>(no explain cards)</p>")

    html.append("</body></html>")
    (out/"report.html").write_text("\n".join(html))
    print(f"Wrote HTML report to {out/'report.html'}")

if __name__ == "__main__":
    main()
