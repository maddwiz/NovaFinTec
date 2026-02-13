#!/usr/bin/env python3
"""
walk_forward_plus.py v7.0 (meta-council SAFE + turnover/cost robust)

Applies add-ons and recomputes per-asset Hit/Sharpe/MaxDD on net returns.
Saves per-asset daily series to runs_plus/<asset>/oos_plus.csv.
"""

import os, json, math, warnings
from pathlib import Path
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

# flags
SKIP_HEARTBEAT = os.environ.get("SKIP_HEARTBEAT","0") == "1"
SKIP_DNA       = os.environ.get("SKIP_DNA","0") == "1"
SKIP_SYMBOLIC  = os.environ.get("SKIP_SYMBOLIC","0") == "1"
SKIP_REFLEXIVE = os.environ.get("SKIP_REFLEXIVE","0") == "1"

# tunables
DNA_THRESH       = float(os.environ.get("DNA_THRESH", "0.05"))
REFLEXIVE_CLIP   = float(os.environ.get("REFLEXIVE_CLIP", "0.2"))
SYMBOLIC_TILT    = float(os.environ.get("SYMBOLIC_TILT", "0.1"))
HB_MULT          = float(os.environ.get("HB_MULT", "1.0"))
META_STRENGTH    = float(os.environ.get("META_STRENGTH", "0.5"))
META_SIGN_THRESH = float(os.environ.get("META_SIGN_THRESH", "0.05"))
META_REQUIRE_AGREE = os.environ.get("META_REQUIRE_AGREE","1") == "1"

# robust scoring / execution realism
COST_BPS = float(os.environ.get("WF_COST_BPS", "1.0"))
WINSOR_PCT = float(os.environ.get("WF_WINSOR_PCT", "0.005"))
TURNOVER_STEP_CAP = float(os.environ.get("WF_TURNOVER_STEP_CAP", "0.35"))

def read_json(path, default):
    try: return json.loads(Path(path).read_text())
    except Exception: return default

def safe_max_drawdown(equity: pd.Series) -> float:
    eq = pd.to_numeric(equity, errors="coerce").ffill().fillna(1.0).clip(lower=1e-9)
    log_eq = np.log(eq)
    peak = np.maximum.accumulate(log_eq.values)
    dd = np.exp(log_eq.values - peak) - 1.0
    return float(np.nanmin(dd)) if dd.size else float("nan")

def sharpe_ratio(pnl: pd.Series) -> float:
    s = pd.to_numeric(pnl, errors="coerce").dropna()
    if s.size == 0: return float("nan")
    return float(s.mean() / (s.std() + 1e-9) * math.sqrt(252))

def hit_ratio(pnl: pd.Series, ret: pd.Series) -> float:
    a = np.sign(pd.to_numeric(pnl, errors="coerce").fillna(0).values)
    b = np.sign(pd.to_numeric(ret, errors="coerce").fillna(0).values)
    if a.size == 0 or b.size == 0: return float("nan")
    return float((a == b).mean())

def safe_mean(x):
    s = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(s.mean()) if not s.empty else None


def winsorize_series(x: pd.Series, p: float) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").fillna(0.0)
    p = max(0.0, min(0.49, float(p)))
    if p <= 0:
        return s
    lo = float(s.quantile(p))
    hi = float(s.quantile(1.0 - p))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return s
    return s.clip(lo, hi)


def throttle_position(pos: pd.Series, max_step: float) -> pd.Series:
    cap = float(max(0.0, max_step))
    if cap <= 0:
        return pd.to_numeric(pos, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    p = pd.to_numeric(pos, errors="coerce").fillna(0.0).values.astype(float)
    out = np.zeros_like(p)
    if p.size == 0:
        return pd.Series(out, index=pos.index)
    out[0] = np.clip(p[0], -1.0, 1.0)
    for i in range(1, p.size):
        target = float(np.clip(p[i], -1.0, 1.0))
        delta = target - out[i - 1]
        if abs(delta) > cap:
            target = out[i - 1] + np.sign(delta) * cap
        out[i] = float(np.clip(target, -1.0, 1.0))
    return pd.Series(out, index=pos.index)

# artifacts
council      = read_json(RUNS / "council.json", {"final_weights": {}}).get("final_weights", {})
meta_council = read_json(RUNS / "meta_council.json", {"asset_weights": {}}).get("asset_weights", {})
heartbeat    = read_json(RUNS / "heartbeat.json", {"exposure_scaler": 1.0})
dna_map      = read_json(RUNS / "dna_drift.json", {"dna_drift": {}}).get("dna_drift", {})
sym_map      = read_json(RUNS / "symbolic.json", {"symbolic": {}}).get("symbolic", {})
refl_map     = read_json(RUNS / "reflexive.json", {"weights": {}}).get("weights", {})

hb_scaler_raw = heartbeat.get("exposure_scaler", None)
if hb_scaler_raw is None:
    hb_curve = heartbeat.get("heartbeat", {}) if isinstance(heartbeat, dict) else {}
    try:
        if hb_curve:
            last_bpm = float(hb_curve[max(hb_curve.keys())])
            ratio = (last_bpm - 60.0) / 120.0
            hb_scaler_raw = float(np.clip(1.15 - 0.60 * ratio, 0.50, 1.20))
        else:
            hb_scaler_raw = 1.0
    except Exception:
        hb_scaler_raw = 1.0
hb_scaler = (1.0 if SKIP_HEARTBEAT else hb_scaler_raw) * HB_MULT

dna_market_last = 0.0
try:
    dna_csv = RUNS / "dna_drift.csv"
    if dna_csv.exists():
        ddf = pd.read_csv(dna_csv)
        if "dna_drift" in ddf.columns and len(ddf):
            dna_market_last = float(pd.to_numeric(ddf["dna_drift"], errors="coerce").dropna().iloc[-1])
except Exception:
    dna_market_last = 0.0

rows = []
for asset_dir in RUNS.iterdir():
    if not asset_dir.is_dir(): continue
    oos_file = asset_dir / "oos.csv"
    if not oos_file.exists(): continue

    sym = asset_dir.name
    df = pd.read_csv(oos_file)
    if not {"date","ret","pos"}.issubset(df.columns): continue

    df["ret"] = pd.to_numeric(df["ret"], errors="coerce").fillna(0.0).clip(-0.20, 0.20)
    base_pos  = pd.to_numeric(df["pos"], errors="coerce").fillna(0.0)

    # ----- Meta/Council SIGN (safe) -----
    meta_w = float(meta_council.get(sym, 0.0))
    council_w = float(council.get(sym, 0.0))
    council_sign = 0.0 if council_w == 0.0 else (1.0 if council_w > 0 else -1.0)
    meta_sign = 0.0
    if abs(meta_w) >= META_SIGN_THRESH:
        meta_sign = 1.0 if meta_w > 0 else -1.0

    if META_REQUIRE_AGREE:
        if meta_sign != 0.0 and council_sign != 0.0 and meta_sign == council_sign:
            base_sign = meta_sign
        else:
            base_sign = council_sign
    else:
        base_sign = meta_sign if meta_sign != 0.0 else council_sign

    pos = base_pos * (1.0 if base_sign == 0.0 else base_sign)

    # meta sizing (mild)
    if META_STRENGTH > 0 and meta_sign != 0.0:
        if (not META_REQUIRE_AGREE) or (meta_sign == council_sign and council_sign != 0.0):
            amp = 1.0 + (min(1.0, abs(meta_w)) * min(1.0, META_STRENGTH))
            pos = pos * amp

    # Heartbeat
    pos = pos * hb_scaler

    # DNA gate
    drift_val = 0.0
    if (not SKIP_DNA) and sym in dna_map and dna_map[sym]:
        try: drift_val = float(dna_map[sym][max(dna_map[sym].keys())])
        except Exception: drift_val = 0.0
    elif not SKIP_DNA:
        drift_val = dna_market_last
    if (not SKIP_DNA) and (drift_val > DNA_THRESH):
        pos = pos * 0.5

    # Symbolic tilt
    sym_score = 0.0 if SKIP_SYMBOLIC else float(sym_map.get(sym, 0.0))
    if sym_score >  +0.5: pos = pos + SYMBOLIC_TILT
    if sym_score <  -0.5: pos = pos - SYMBOLIC_TILT

    # Reflexive overlay
    refl = 0.0 if SKIP_REFLEXIVE else float(refl_map.get(sym, 0.0))
    pos = pos + np.clip(refl * REFLEXIVE_CLIP, -REFLEXIVE_CLIP, REFLEXIVE_CLIP)

    # clip + turnover throttle + compute gross/net pnl
    pos = pd.Series(np.clip(pos, -1.0, 1.0), index=df.index)
    pos = throttle_position(pos, max_step=TURNOVER_STEP_CAP)
    pos_lag = pos.shift(1).fillna(0.0)
    turnover = pos.diff().abs().fillna(0.0)
    cost = turnover * (COST_BPS / 10000.0)

    pnl_gross = (pos_lag * df["ret"]).clip(-0.95, 0.95)
    pnl_net = (pnl_gross - cost).clip(-0.95, 0.95)
    pnl_eval = winsorize_series(pnl_net, p=WINSOR_PCT)
    pnl_eval_gross = winsorize_series(pnl_gross, p=WINSOR_PCT)
    equity_plus = (1.0 + pnl_net).cumprod()

    # metrics
    hit = hit_ratio(pnl_net, df["ret"])
    sh  = sharpe_ratio(pnl_eval)
    sh_gross = sharpe_ratio(pnl_eval_gross)
    mdd = safe_max_drawdown(equity_plus)
    turnover_ann = float(turnover.mean() * 252.0)
    cost_ann = float(cost.mean() * 252.0)
    avg_abs_pos = float(pos.abs().mean())

    # save per-asset daily series for portfolio
    outp = asset_dir / "oos_plus.csv"
    pd.DataFrame({
        "date": df["date"],
        "ret": df["ret"],
        "pos_plus": pos,
        "turnover": turnover,
        "cost": cost,
        "pnl_gross": pnl_gross,
        "pnl_plus": pnl_net,
        "equity_plus": equity_plus,
    }).to_csv(outp, index=False)

    rows.append({
        "asset": sym,
        "hit": hit,
        "sharpe": sh,
        "sharpe_gross": sh_gross,
        "maxDD": mdd,
        "turnover_ann": turnover_ann,
        "cost_ann": cost_ann,
        "avg_abs_pos": avg_abs_pos,
        "dna_drift": drift_val,
        "symbolic_score": sym_score,
        "reflexive_raw": refl,
        "heartbeat_scaler": hb_scaler,
        "meta_weight": meta_w,
        "council_weight": council_w,
    })

out = pd.DataFrame(rows)
out_csv = RUNS / "walk_forward_table_plus.csv"
out.to_csv(out_csv, index=False)
print(f"✅ Wrote {out_csv.as_posix()}")

avg_hit = safe_mean(out["hit"])
avg_sh  = safe_mean(out["sharpe"])
avg_dd  = safe_mean(out["maxDD"])
print("WF+ SUMMARY (meta-council, saved oos_plus)")
print(f"  Assets: {len(out)}")
print(f"  Hit(avg):    {avg_hit:.3f}" if avg_hit is not None else "  Hit(avg): —")
print(f"  Sharpe(avg): {avg_sh:.3f}" if avg_sh is not None else "  Sharpe(avg): —")
print(f"  MaxDD(avg):  {avg_dd:.3f}" if avg_dd is not None else "  MaxDD(avg): —")
