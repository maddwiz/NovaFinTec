#!/usr/bin/env python3
# tools/osc_guard_auto.py
# If blended Sharpe <= Main Sharpe + margin, force alpha=1.0 (ignore oscillator) and update a small badge card.

from pathlib import Path
import json, pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
FILES = [ROOT / "report_best_plus.html", ROOT / "report_all.html"]
MARGIN = 0.02  # require at least +0.02 Sharpe improvement

CARD_START="<!--OSC_GUARD_CARD_START-->"
CARD_END="<!--OSC_GUARD_CARD_END-->"

def upsert(html, block):
    if CARD_START in html and CARD_END in html:
        pre = html.split(CARD_START)[0]; post = html.split(CARD_END,1)[1]
        return pre + block + post
    return html.replace("</body>", block + "\n</body>") if "</body>" in html else html + block

def sharpe_of_csv(p: Path):
    if not p.exists(): return None
    df = pd.read_csv(p)
    ret_col = None
    for c in ["ret_net","ret","return","ret_gross","pnl","daily_ret","portfolio_ret","port_ret"]:
        if c in df.columns:
            ret_col = c; break
    if ret_col is None:
        for c in ["eq_net","eq","equity","equity_curve"]:
            if c in df.columns:
                r = pd.to_numeric(df[c], errors="coerce").pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0)
                s=r.std(); 
                return 0.0 if s==0 or np.isnan(s) else float((r.mean()/s)*np.sqrt(252))
        return None
    r = pd.to_numeric(df[ret_col], errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0)
    s=r.std(); 
    return 0.0 if s==0 or np.isnan(s) else float((r.mean()/s)*np.sqrt(252))

if __name__ == "__main__":
    main_sh = sharpe_of_csv(RUNS/"portfolio_plus.csv")
    blend_meta_p = RUNS/"blend_main_osc_costed_summary.json"
    if not blend_meta_p.exists():
        raise SystemExit("Run blend_main_with_osc_costed.py first.")
    meta = json.loads(blend_meta_p.read_text())
    blend_sh = float(meta.get("sharpe", 0.0))
    alpha    = float(meta.get("alpha", 1.0))

    action = "KEEP"
    if main_sh is not None and blend_sh <= (main_sh + MARGIN):
        # Force alpha=1.0
        meta["alpha"] = 1.0
        (RUNS/"blend_main_osc_costed_summary.json").write_text(json.dumps(meta, indent=2))
        action = "DISABLE"

    badge = f"""{CARD_START}
<section style="border:1px dashed #aaa;padding:8px;margin:12px 0;border-radius:6px;background:#fafafa">
  <b>Oscillator Guard:</b> {action}. 
  <span style="color:#666">Main Sharpe≈{(main_sh if main_sh is not None else 0):.3f} | Blend Sharpe≈{blend_sh:.3f} | Margin={MARGIN:.2f}</span>
</section>
{CARD_END}"""

    for f in FILES:
        if not f.exists(): continue
        f.write_text(upsert(f.read_text(), badge))
        print(f"✅ Guard badge updated in {f.name}")
