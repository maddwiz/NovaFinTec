
#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Folder with summary.json, metrics_per_fold.csv, equity_curve.csv, explain_cards.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    summary = json.load(open(run/"summary.json"))
    metrics = pd.read_csv(run/"metrics_per_fold.csv")
    eq = pd.read_csv(run/"equity_curve.csv")
    cards = pd.read_csv(run/"explain_cards.csv")

    md = []
    md.append(f"# Q v2.5 Foundations Report")
    md.append("")
    md.append(f"**Asset:** {summary['asset']}")
    md.append(f"**Hit Rate:** {summary['hit_rate']:.3f}")
    md.append(f"**Sharpe:** {summary['sharpe']:.3f}")
    md.append(f"**Max Drawdown:** {summary['max_dd']:.3f}")
    md.append("")
    md.append("## Fold Metrics (first 10 rows)")
    md.append(metrics.head(10).to_markdown(index=False))
    md.append("")
    md.append("## Explainability Cards (first 10 rows)")
    md.append(cards.head(10).to_markdown(index=False))

    with open(out/"report.md","w") as f:
        f.write("\n".join(md))
    print(f"Wrote report to {out/'report.md'}")

if __name__ == "__main__":
    main()
