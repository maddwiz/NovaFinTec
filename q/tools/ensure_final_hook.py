from pathlib import Path
import re

p = Path("scripts/run_walkforward_meta.py")
txt = p.read_text()

# 1) inject apply_postprocess if missing
if "def apply_postprocess(" not in txt:
    inject = """
def apply_postprocess(sig, close, vix=None, k=2.0):
    import numpy as np, pandas as pd
    def _vol(x, w=20):
        r = pd.Series(x).pct_change().rolling(w).std().fillna(method='bfill').replace(0, 1e-8)
        return r.values
    vol = _vol(close)
    s = np.tanh(k * (sig / (vol + 1e-8)))
    if vix is not None:
        # calm → a bit softer, storm → keep size
        v = pd.Series(vix).pct_change().abs().rolling(5).mean().fillna(0.0)
        gate = (0.8 + 0.2 * (v / (v.quantile(0.95)+1e-8))).clip(0.7, 1.0).values
        s = s * gate
    return s
"""
    # put it near the top, after imports
    m = re.search(r"(\nif __name__ == ['\"]__main__['\"]:)", txt)
    if m:
        txt = txt.replace(m.group(1), inject + m.group(1))
    else:
        txt = inject + "\n" + txt

# 2) ensure the helper is actually called before positions
if "apply_postprocess(meta_signal_test" not in txt:
    txt = re.sub(
        r"(meta_signal_test\s*=\s*apply_meta\(.*?\)\s*)",
        r"\1\n"
        r"        # final postprocess (vol targeting + tanh + VIX softener)\n"
        r"        vix_train_series = None\n"
        r"        vix_test_series  = None\n"
        r"        try:\n"
        r"            from qengine.data import load_csv as _lc\n"
        r"            import os\n"
        r"            vix_path = os.path.join(args.data, 'VIX.csv') if os.path.exists(os.path.join(args.data,'VIX.csv')) else os.path.join(args.data,'VIXCLS.csv')\n"
        r"            if os.path.exists(vix_path):\n"
        r"                _v = _lc(vix_path)['Close']\n"
        r"                vix_train_series = _v.reindex(train_df.index)\n"
        r"                vix_test_series  = _v.reindex(test_df.index)\n"
        r"        except Exception:\n"
        r"            pass\n"
        r"        meta_signal_train = apply_postprocess(meta_signal_train, train_df['Close'].values, vix_train_series, k=2.0)\n"
        r"        meta_signal_test  = apply_postprocess(meta_signal_test,  test_df['Close'].values,  vix_test_series,  k=2.0)\n",
        txt, flags=re.S
    )

# 3) save
p.write_text(txt)
print("✅ final hook installed: apply_postprocess + call sites")
