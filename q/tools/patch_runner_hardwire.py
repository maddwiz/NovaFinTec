from pathlib import Path
import re

p = Path("scripts/run_walkforward_meta.py")
txt = p.read_text()
changed = False

# 1) insert helpers if missing
if "def apply_postprocess(" not in txt:
    txt = txt.replace(
        "import numpy as np",
        "import numpy as np\n\n"
        "def _vol_scale(series, window=20, eps=1e-8):\n"
        "    r = series.pct_change().fillna(0.0)\n"
        "    vol = r.rolling(window).std().fillna(method='bfill').replace(0, eps)\n"
        "    return vol\n\n"
        "def apply_postprocess(meta_signal, price_series, vix_series=None, k=2.2):\n"
        "    # shrink on storm days if VIX provided\n"
        "    if vix_series is not None:\n"
        "        storm = (vix_series > 25).astype(float)\n"
        "        storm = storm.reindex(meta_signal.index).fillna(0.0)\n"
        "        meta_signal = meta_signal * (1.0 - 0.15*storm)\n"
        "    # volatility targeting + tanh + cap\n"
        "    vol = _vol_scale(price_series)\n"
        "    meta_signal = np.tanh(k * (meta_signal / (vol + 1e-8)))\n"
        "    meta_signal = np.clip(meta_signal, -1.0, 1.0)\n"
        "    return meta_signal\n"
    )
    changed = True

# 2) ensure VIX series exists once (harmless if VIX missing)
if "## Q_HOOK_VIX_SERIES" not in txt and "sig_train = {'price': (0.8 * price_score_train)}" in txt:
    txt = txt.replace(
        "sig_train = {'price': (0.8 * price_score_train)}",
        "sig_train = {'price': (0.8 * price_score_train)}\n"
        "        ## Q_HOOK_VIX_SERIES\n"
        "        vix_train_series = VIX.reindex(train_df.index)['Close'] if 'VIX' in locals() and VIX is not None else None\n"
        "        vix_test_series  = VIX.reindex(test_df.index)['Close'] if 'VIX' in locals() and VIX is not None else None"
    )
    changed = True

# 3) locate where meta_signal_* are built and hard-wire postprocess right after
pat = re.compile(
    r"(meta_signal_train\s*=\s*apply_meta\(.*?\)\s*[\r\n]+[\s\S]*?meta_signal_test\s*=\s*apply_meta\(.*?\))",
    re.M
)
m = pat.search(txt)
if m and "HARDWIRED FINAL POSTPROCESS" not in txt:
    insert_after = m.end()
    inj = (
        "\n        # === HARDWIRED FINAL POSTPROCESS ===\n"
        "        meta_signal_train = apply_postprocess(meta_signal_train, train_df['Close'], vix_train_series, k=2.2)\n"
        "        meta_signal_test  = apply_postprocess(meta_signal_test,  test_df['Close'],  vix_test_series,  k=2.2)\n"
        "        # === END HARDWIRE ===\n"
    )
    txt = txt[:insert_after] + inj + txt[insert_after:]
    changed = True

# 4) ensure positions come from meta_signal_* (not old signal)
txt2 = re.sub(r"pos_train\s*=\s*[^\n]+", "pos_train = np.sign(meta_signal_train)", txt, count=1)
txt2 = re.sub(r"pos_test\s*=\s*[^\n]+",  "pos_test  = np.sign(meta_signal_test)",  txt2, count=1)
if txt2 != txt:
    txt = txt2
    changed = True

if changed:
    p.write_text(txt)
    print("hardwired: postprocess installed and positions use meta_signal_*")
else:
    print("no changes (already hardwired).")
