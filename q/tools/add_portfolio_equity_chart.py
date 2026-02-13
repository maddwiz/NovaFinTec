#!/usr/bin/env python3
"""
add_portfolio_equity_chart.py

- Reads runs_plus/portfolio_plus.csv (date, port_equity)
- Renders a PNG equity curve at runs_plus/portfolio_equity.png
- Embeds the chart into both report_all.html and report_best_plus.html
  as an inline <img> (base64), with the header "Portfolio Equity Curve (WF+)".
"""

from pathlib import Path
import base64
import io

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_all.html", ROOT / "report_best_plus.html"]

def exists(p: Path):
    try:
        return p.exists()
    except:
        return False

def insert_before_body(html_in: str, block: str) -> str:
    return html_in.replace("</body>", block + "\n</body>") if "</body>" in html_in else html_in + block

def make_chart_png() -> bytes:
    csvp = RUNS / "portfolio_plus.csv"
    if not exists(csvp):
        raise SystemExit("portfolio_plus.csv not found. Run build_portfolio_plus.py first.")
    df = pd.read_csv(csvp, parse_dates=["date"])
    if not {"date","port_equity"}.issubset(df.columns):
        raise SystemExit("portfolio_plus.csv missing required columns.")

    # Plot (single figure, no custom colors/styles)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["date"], df["port_equity"])
    ax.set_title("Portfolio Equity Curve (WF+)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (index = 1.0)")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()

    # Also save a copy to disk for reference
    out_png = RUNS / "portfolio_equity.png"
    try:
        out_png.write_bytes(png_bytes)
        print(f"✅ Wrote {out_png.as_posix()}")
    except Exception:
        pass

    return png_bytes

def main():
    png = make_chart_png()
    b64 = base64.b64encode(png).decode("ascii")
    img_tag = f'<img alt="Portfolio Equity Curve (WF+)" style="width:100%;max-width:960px;display:block;margin:8px auto;" src="data:image/png;base64,{b64}"/>'
    block = f"""
<section style="border:1px solid #ddd;padding:12px;margin:16px 0;border-radius:8px">
  <h2 style="margin:0 0 8px 0">Portfolio Equity Curve (WF+)</h2>
  {img_tag}
  <p style="color:#666;margin:6px 0 0">Source: runs_plus/portfolio_plus.csv</p>
</section>
""".strip()

    for fp in FILES:
        if not exists(fp):
            print(f"skip {fp.name}: not found")
            continue
        html_in = fp.read_text()
        if "Portfolio Equity Curve (WF+)" in html_in:
            print(f"already has Portfolio Equity Curve: {fp.name}")
            continue
        fp.write_text(insert_before_body(html_in, block))
        print(f"✅ Injected Portfolio Equity Curve into {fp.name}")

if __name__ == "__main__":
    main()
