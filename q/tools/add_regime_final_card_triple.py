#!/usr/bin/env python3
# tools/add_regime_final_card_triple.py
# Shows Main, Regime-Governed, and Regime+DNA side-by-side.

from pathlib import Path
import json as _json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
CARD_START="<!--REGIME_FINAL_TRIPLE_START-->"
CARD_END="<!--REGIME_FINAL_TRIPLE_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]; post = html.split(CARD_END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

def _num(x):
    try: return float(x)
    except: return None
def f3(x):
    v = _num(x); return f"{v:.3f}" if v is not None else "?"
def f2(x):
    v = _num(x); return f"{v:.2f}" if v is not None else "?"

if __name__ == "__main__":
    base = RUNS / "final_portfolio_summary.json"
    regj = RUNS / "final_portfolio_regime_summary.json"
    dnap = RUNS / "final_portfolio_regime_dna_summary.json"
    gcsv = RUNS / "final_portfolio_regime.csv"
    dcsv = RUNS / "final_portfolio_regime_dna.csv"

    if not base.exists():
        raise SystemExit("Missing final_portfolio_summary.json")
    if not regj.exists():
        raise SystemExit("Missing final_portfolio_regime_summary.json (run apply_regime_governor.py)")
    if not dnap.exists():
        raise SystemExit("Missing final_portfolio_regime_dna_summary.json (run apply_regime_governor_dna.py)")

    bj = _json.loads(base.read_text())
    rj = _json.loads(regj.read_text())
    dj = _json.loads(dnap.read_text())

    main_block = (
        f"IS Sharpe: {f3(bj.get('in_sample',{}).get('sharpe'))}<br>"
        f"OOS Sharpe: {f3(bj.get('out_sample',{}).get('sharpe'))}<br>"
        f"IS Hit: {f2(bj.get('in_sample',{}).get('hit'))} | OOS Hit: {f2(bj.get('out_sample',{}).get('hit'))}<br>"
        f"IS MaxDD: {f2(bj.get('in_sample',{}).get('maxdd'))} | OOS MaxDD: {f2(bj.get('out_sample',{}).get('maxdd'))}"
    )
    reg_block = (
        f"IS Sharpe: {f3(rj.get('in_sample',{}).get('sharpe'))}<br>"
        f"OOS Sharpe: {f3(rj.get('out_sample',{}).get('sharpe'))}<br>"
        f"IS Hit: {f2(rj.get('in_sample',{}).get('hit'))} | OOS Hit: {f2(rj.get('out_sample',{}).get('hit'))}<br>"
        f"IS MaxDD: {f2(rj.get('in_sample',{}).get('maxdd'))} | OOS MaxDD: {f2(rj.get('out_sample',{}).get('maxdd'))}"
    )
    dna_block = (
        f"IS Sharpe: {f3(dj.get('in_sample',{}).get('sharpe'))}<br>"
        f"OOS Sharpe: {f3(dj.get('out_sample',{}).get('sharpe'))}<br>"
        f"IS Hit: {f2(dj.get('in_sample',{}).get('hit'))} | OOS Hit: {f2(dj.get('out_sample',{}).get('hit'))}<br>"
        f"IS MaxDD: {f2(dj.get('in_sample',{}).get('maxdd'))} | OOS MaxDD: {f2(dj.get('out_sample',{}).get('maxdd'))}"
    )

    # preview chart: all three curves if available
    png = RUNS / "regime_triple.png"
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        if gcsv.exists():
            g = pd.read_csv(gcsv, parse_dates=["DATE"])
            # Main
            if "eq_main" in g.columns:
                plt.plot(g["DATE"], g["eq_main"], label="Main")
            else:
                plt.plot(g["DATE"], (1.0 + g["ret_main"]).cumprod(), label="Main")
            # Regime
            if "eq_governed" in g.columns:
                plt.plot(g["DATE"], g["eq_governed"], label="Regime")
            else:
                plt.plot(g["DATE"], (1.0 + g["ret_governed"]).cumprod(), label="Regime")
        if dcsv.exists():
            d = pd.read_csv(dcsv, parse_dates=["DATE"])
            # Regime+DNA
            if "eq_governed_dna" in d.columns:
                plt.plot(d["DATE"], d["eq_governed_dna"], label="Regime+DNA")
            else:
                plt.plot(d["DATE"], (1.0 + d["ret_governed_dna"]).cumprod(), label="Regime+DNA")
        plt.legend()
        plt.title("Equity: Main vs Regime vs Regime+DNA")
        plt.xlabel("Date"); plt.ylabel("Index from 1.0")
        plt.tight_layout()
        plt.savefig(png, dpi=120)
    except Exception as e:
        print("chart warn:", e)
        png = None

    block = f"""{CARD_START}
<section style="border:2px solid #58c;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Final Portfolio — Main vs Regime vs Regime+DNA</h2>
  <div style="display:flex;gap:24px;flex-wrap:wrap">
    <div style="min-width:240px"><h3>Main</h3><p>{main_block}</p></div>
    <div style="min-width:240px"><h3>Regime</h3><p>{reg_block}</p></div>
    <div style="min-width:240px"><h3>Regime+DNA</h3><p>{dna_block}</p></div>
  </div>
  {"<img src='runs_plus/regime_triple.png' style='max-width:100%'>" if png and png.exists() else ""}
  <p style="color:#666;margin-top:6px">DNA drift attenuates add-ons in unstable periods; weight flows back to Main.</p>
</section>
{CARD_END}"""

    for f in FILES:
        if not f.exists():
            print("skip", f.name)
            continue
        f.write_text(upsert(f.read_text(), block))
        print(f"✅ Upserted REGIME TRIPLE card in", f.name)
