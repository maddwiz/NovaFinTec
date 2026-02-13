#!/usr/bin/env python3
# tools/add_stress_card.py
# Computes window returns for Main, Regime+DNA, and Vol-Target across stress periods,
# then inserts a card with a small table.

from pathlib import Path
import pandas as pd, numpy as np, json as _json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"
FILES = [ROOT/"report_best_plus.html", ROOT/"report_all.html"]
START="<!--STRESS_CARD_START-->"
END  ="<!--STRESS_CARD_END-->"

# Date windows (adjust if your data starts later)
WINDOWS = [
    ("GFC (2008-09 to 2009-03)", "2008-09-01", "2009-03-31"),
    ("COVID Crash (2020-02 to 2020-04)", "2020-02-01", "2020-04-30"),
    ("Rates Spike (2022-06 to 2023-10)", "2022-06-01", "2023-10-31"),
]

def _load_csv(rel, ret_cols, eq_cols):
    p = RUNS/rel
    if not p.exists(): return None
    df = pd.read_csv(p)
    dcol=None
    for c in df.columns:
        if str(c).lower() in ("date","timestamp"): dcol=c; break
    if dcol is None and "DATE" in df.columns: dcol="DATE"
    if dcol:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol]).sort_values(dcol)
    for c in ret_cols:
        if c in df.columns:
            r = pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
            return df[[dcol]].rename(columns={dcol:"DATE"}) if dcol else None, r.values
    for c in eq_cols:
        if c in df.columns:
            eq = pd.to_numeric(df[c], errors="coerce").replace([np.inf,-np.inf], np.nan)
            r = eq.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
            return df[[dcol]].rename(columns={dcol:"DATE"}) if dcol else None, r.values
    return None, None

def _cum_return_between(dates, r, start, end):
    if dates is None: return np.nan
    s = pd.to_datetime(start); e = pd.to_datetime(end)
    m = (dates["DATE"] >= s) & (dates["DATE"] <= e)
    if not m.any(): return np.nan
    rr = pd.Series(r)[m.values]
    eq = (1.0 + rr).cumprod()
    return float(eq.iloc[-1]-1.0)

def _fmt_pct(x):
    if pd.isna(x): return "—"
    return f"{x*100:0.1f}%"

def upsert(html, block):
    if START in html and END in html:
        pre = html.split(START)[0]; post = html.split(END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

if __name__ == "__main__":
    m_dates, m_r = _load_csv("portfolio_plus.csv",
                             ["ret","ret_net","ret_plus","return","daily_ret","port_ret","portfolio_ret"],
                             ["eq","eq_net","equity","equity_curve","equity_index","portfolio_eq"])
    d_dates, d_r = _load_csv("final_portfolio_regime_dna.csv",
                             ["ret_governed_dna","ret"],
                             ["eq","equity","equity_curve"])
    v_dates, v_r = _load_csv("portfolio_plus_vt.csv",
                             ["ret_vt","ret"], ["eq_vt","eq"])

    rows=[]
    for name, s, e in WINDOWS:
        rows.append({
            "window": name,
            "Main":  _cum_return_between(m_dates, m_r, s, e),
            "Regime+DNA": _cum_return_between(d_dates, d_r, s, e),
            "Vol-Target": _cum_return_between(v_dates, v_r, s, e),
        })
    df = pd.DataFrame(rows)
    # Build HTML table
    show = df.copy()
    for c in ["Main","Regime+DNA","Vol-Target"]:
        show[c] = show[c].map(_fmt_pct)
    html_table = show.to_html(index=False, border=0, justify="left")

    block = f"""{START}
<section style="border:2px solid #946;padding:14px;margin:18px 0;border-radius:10px">
  <h2 style="margin:0 0 8px 0">Stress Windows (Cumulative Return)</h2>
  <p style="color:#666;margin:6px 0 12px 0">How each track behaved during major shocks.</p>
  {html_table}
</section>
{END}"""

    for f in FILES:
        if not f.exists():
            print("skip", f.name); 
            continue
        f.write_text(upsert(f.read_text(), block))
        print("✅ Upserted STRESS card in", f.name)
