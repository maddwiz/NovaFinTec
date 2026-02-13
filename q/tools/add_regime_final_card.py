#!/usr/bin/env python3
# tools/add_regime_final_card.py
# Adds a side-by-side card comparing Main final vs Regime-Governed final.

from pathlib import Path
import json as _json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
CARD_START="<!--REGIME_FINAL_CARD_START-->"
CARD_END="<!--REGIME_FINAL_CARD_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]; post = html.split(CARD_END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

def _num(x):
    try:
        return float(x)
    except Exception:
        return None

def f3(x):
    v = _num(x)
    return f"{v:.3f}" if v is not None else "?"

def f2(x):
    v = _num(x)
    return f"{v:.2f}" if v is not None else "?"

if __name__ == "__main__":
    base = RUNS / "final_portfolio_summary.json"
    govj = RUNS / "final_portfolio_regime_summary.json"
    gcsv = RUNS / "final_portfolio_regime.csv"

    if not base.exists():
        raise SystemExit("Missing runs_plus/final_portfolio_summary.json – run your normal pipeline first.")
    if not govj.exists() or not gcsv.exists():
        raise SystemExit("Run tools/apply_regime_governor.py first.")

    basej = _json.loads(base.read_text())
    gov   = _json.loads(govj.read_text())

    # tiny preview: compare equity curves
    png = RUNS / "regime_vs_main.png"
    try:
        import matplotlib.pyplot as plt
        df = pd.read_csv(gcsv, parse_dates=["DATE"])
        if not df.empty:
            plt.figure()
            plt.plot(df["DATE"], df.get("eq_main", (1.0 + df["ret_main"]).cumprod()), label="Main")
            plt.plot(df["DATE"], df.get("eq_governed", (1.0 + df["ret_governed"]).cumprod()), label="Regime-Governed")
            plt.legend()
            plt.title("Equity: Main vs Regime-Governed")
            plt.xlabel("Date"); plt.ylabel("Index from 1.0")
            plt.tight_layout()
            plt.savefig(png, dpi=120)
    except Exception:
        png = None

    bi = basej.get("in_sample",  {})
    bo = basej.get("out_sample", {})
    gi = gov.get("in_sample",    {})
    go = gov.get("out_sample",   {})

    main_block = (
        f"IS Sharpe: {f3(bi.get('sharpe'))}<br>"
        f"OOS Sharpe: {f3(bo.get('sharpe'))}<br>"
        f"IS Hit: {f2(bi.get('hit'))} | OOS Hit: {f2(bo.get('hit'))}<br>"
        f"IS MaxDD: {f2(bi.get('maxdd'))} | OOS MaxDD: {f2(bo.get('maxdd'))}"
    )

    gov_block = (
        f"IS Sharpe: {f3(gi.get('sharpe'))}<br>"
        f"OOS Sharpe: {f3(go.get('sharpe'))}<br>"
        f"IS Hit: {f2(gi.get('hit'))} | OOS Hit: {f2(go.get('hit'))}<br>"
        f"IS MaxDD: {f2(gi.get('maxdd'))} | OOS MaxDD: {f2(go.get('maxdd'))}"
    )

    note = gov.get("weights_note", "Effective weights reallocate missing sleeves back to Main.")

    block = f"""{CARD_START}
<section style="border:2px solid #6a4;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Final Portfolio — Main vs Regime-Governed</h2>
  <div style="display:flex;gap:24px;flex-wrap:wrap">
    <div style="min-width:260px">
      <h3 style="margin:6px 0">Main (current)</h3>
      <p>{main_block}</p>
    </div>
    <div style="min-width:260px">
      <h3 style="margin:6px 0">Regime-Governed (observer → applied)</h3>
      <p>{gov_block}</p>
    </div>
  </div>
  {"<img src='runs_plus/regime_vs_main.png' style='max-width:100%'>" if png and png.exists() else ""}
  <p style="color:#666;margin-top:6px">Note: {note} Symbolic/Reflexive are mapped from signals with conservative daily caps.</p>
</section>
{CARD_END}"""

    for f in FILES:
        if not f.exists():
            print("skip", f.name)
            continue
        f.write_text(upsert(f.read_text(), block))
        print(f"✅ Upserted REGIME FINAL card in", f.name)
