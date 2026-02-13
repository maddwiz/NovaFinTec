from pathlib import Path
import re

p = Path("scripts/run_walkforward_meta.py")
txt = p.read_text()

# make sure the final hook exists (you installed it already)
if "def apply_postprocess(" not in txt:
    raise SystemExit("apply_postprocess not found. Run tools/ensure_final_hook.py first.")

# switch positions from sign(meta_signal) → continuous meta_signal in [-1,1]
new = re.sub(r"pos_train\s*=\s*[^\n]+",
             "pos_train = meta_signal_train.astype(float).clip(-1.0, 1.0)", txt, count=1)
new = re.sub(r"pos_test\s*=\s*[^\n]+",
             "pos_test  = meta_signal_test.astype(float).clip(-1.0, 1.0)",  new, count=1)

if new != txt:
    p.write_text(new)
    print("✅ positions now use continuous meta_signal (clipped to [-1,1])")
else:
    print("ℹ️ positions already continuous (no change)")
