#!/usr/bin/env python3
# tools/add_dna_card.py
# Adds a DNA Drift card to HTML reports. Works with multiple possible filenames.

from pathlib import Path
import json as _json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
CARD_START="<!--DNA_CARD_START-->"
CARD_END="<!--DNA_CARD_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]; post = html.split(CARD_END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

def _find_csv():
    for name in ["dna_drift.csv", "drift_series.csv", "dna_series.csv"]:
        p = RUNS / name
        if p.exists(): return p
    return None

def _find_json():
    for name in ["dna_summary.json", "dna_drift_summary.json", "drift_summary.json"]:
        p = RUNS / name
        if p.exists(): return p
    return None

if __name__ == "__main__":
    csvp = _find_csv()
    jsnp = _find_json()
    if not csvp or not jsnp:
        raise SystemExit("Run tools/make_dna_drift.py first (so dna_drift.csv + dna_summary.json exist).")

    # small preview chart
    png = RUNS / "dna_drift_preview.png"
    try:
        import matplotlib.pyplot as plt
        df = pd.read_csv(csvp, parse_dates=["DATE"])
        # try to find a column named 'drift' or similar
        drift_col = None
        for c in df.columns:
            lc = c.lower()
            if lc in ("drift","dna_drift","drift_pct","drift_z"):
                drift_col = c; break
        if drift_col is None:
            # pick the first numeric column after DATE
            for c in df.columns:
                if c != "DATE" and pd.api.types.is_numeric_dtype(df[c]):
                    drift_col = c; break
        if drift_col:
            plt.figure()
            plt.plot(df["DATE"], df[drift_col])
            plt.title(f"DNA Drift ({drift_col})")
            plt.xlabel("Date"); plt.ylabel(drift_col)
            plt.tight_layout()
            plt.savefig(png, dpi=120)
    except Exception:
        png = None

    meta = _json.loads((jsnp).read_text())
    last = meta.get("last", {})
    last_txt = ""
    if last:
        last_txt = f"<p><b>Latest:</b> {last.get('date','?')} &nbsp; drift={last.get('drift','?')}</p>"

    block = f"""{CARD_START}
<section style="border:2px solid #444;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">DNA Drift</h2>
  <p style="color:#666;margin:4px 0 8px 0">Compressed model state movement over time. Higher = bigger internal change (possible regime shift risk).</p>
  {last_txt if last_txt else ""}
  {"<img src='runs_plus/dna_drift_preview.png' style='max-width:100%'>" if png and png.exists() else "<p>No preview image available.</p>"}
</section>
{CARD_END}"""
    for f in FILES:
        if not f.exists():
            print("skip", f.name); 
            continue
        f.write_text(upsert(f.read_text(), block))
        print(f"✅ Upserted DNA card in", f.name)
