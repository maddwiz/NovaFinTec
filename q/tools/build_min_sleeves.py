#!/usr/bin/env python3
# tools/build_min_sleeves.py
# Build minimal sleeve streams so the Regime governor has inputs.
# Writes in runs_plus/:
#   sleeve_vol.csv        (DATE, ret)
#   sleeve_osc.csv        (DATE, ret)
#   symbolic_signal.csv   (DATE, sym_signal)
#   reflexive_signal.csv  (DATE, reflexive_signal)

from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(parents=True, exist_ok=True)

P_PORT = RUNS / "portfolio_plus.csv"

# ==== KNOBS (tiny nudges) ====
VOL_SCALE = 0.0014    # was 0.0015
OSC_SCALE = 0.0022    # was 0.0012
SYM_DEFAULT = 0.0
RFX_DEFAULT = 0.0
# =============================

def _safe_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf], np.nan)

def _load_main(portfolio_path: Path = P_PORT):
    if not portfolio_path.exists():
        raise SystemExit("Missing runs_plus/portfolio_plus.csv (run portfolio_from_runs_plus.py first).")
    df = pd.read_csv(portfolio_path)
    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or lowers.get("timestamp") or list(df.columns)[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).rename(columns={dcol:"DATE"}).sort_values("DATE")
    for c in ["ret","ret_net","ret_plus","return","daily_ret","port_ret","portfolio_ret","pnl","pnl_plus"]:
        if c in df.columns:
            r = _safe_num(df[c]).fillna(0.0).clip(-0.5,0.5)
            return pd.DataFrame({"DATE": df["DATE"], "ret_main": r})
    for c in ["eq","eq_net","equity","equity_curve","equity_index","portfolio_eq","port_equity"]:
        if c in df.columns:
            eq = _safe_num(df[c])
            r = eq.pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)
            return pd.DataFrame({"DATE": df["DATE"], "ret_main": r})
    raise SystemExit("portfolio_plus.csv has no returns/equity columns I recognize.")

def _zscore(x, w):
    s = pd.Series(x)
    m = s.rolling(w, min_periods=max(5, w//3)).mean()
    v = s.rolling(w, min_periods=max(5, w//3)).std()
    z = (s - m) / v.replace(0, np.nan)
    return z.fillna(0.0).clip(-5,5)

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowers = {c.lower(): c for c in df.columns}
    for c in candidates:
        hit = lowers.get(c.lower())
        if hit is not None:
            return hit
    return None


def _load_signal_existing(path: Path, dates: pd.Series, signal_candidates: list[str], conf_candidates: list[str], out_col: str):
    if not path.exists():
        return None, "missing"
    try:
        src = pd.read_csv(path)
    except Exception:
        return None, "read_error"
    if src.empty:
        return None, "empty"

    dcol = _pick_col(src, ["DATE", "date", "timestamp", "time"])
    scol = _pick_col(src, signal_candidates)
    if dcol is None or scol is None:
        return None, "schema_missing"

    src = src.rename(columns={dcol: "DATE"})
    src["DATE"] = pd.to_datetime(src["DATE"], errors="coerce")
    src = src.dropna(subset=["DATE"]).copy()
    if src.empty:
        return None, "no_dates"

    src["_sig"] = _safe_num(src[scol]).clip(-1.0, 1.0)
    ccol = _pick_col(src, conf_candidates)
    if ccol is not None:
        src["_w"] = _safe_num(src[ccol]).fillna(0.5).clip(0.0, 1.0)
        src["_wx"] = src["_sig"] * src["_w"]
        g = src.groupby("DATE", as_index=False).agg(_wx=("_wx", "sum"), _w=("_w", "sum"))
        src = pd.DataFrame({"DATE": g["DATE"], "_sig": g["_wx"] / (g["_w"] + 1e-12)})
    else:
        src = src.groupby("DATE", as_index=False)["_sig"].mean()

    base = pd.DataFrame({"DATE": pd.to_datetime(dates, errors="coerce")}).dropna()
    out = base.merge(src, on="DATE", how="left")
    out["_sig"] = _safe_num(out["_sig"]).ffill().fillna(0.0).clip(-1.0, 1.0)
    return pd.DataFrame({"DATE": out["DATE"], out_col: out["_sig"]}), "existing"


def _fallback_symbolic(r: np.ndarray) -> np.ndarray:
    rr = pd.Series(np.asarray(r, float))
    trend = np.tanh(_zscore(rr.rolling(12, min_periods=4).mean().fillna(0.0), 42))
    shock = np.tanh(_zscore(rr, 8))
    sym = np.clip(0.65 * trend - 0.35 * shock, -1.0, 1.0)
    return np.asarray(sym, float)


def _fallback_reflexive(r: np.ndarray, sym: np.ndarray) -> np.ndarray:
    rr = pd.Series(np.asarray(r, float))
    s = pd.Series(np.asarray(sym, float)).shift(1).fillna(0.0)
    vol = pd.Series(rr).rolling(21, min_periods=7).std().fillna(0.0)
    vol_z = np.tanh(_zscore(vol, 42))
    feedback = np.sign(rr) * np.clip(np.abs(rr) / (vol + 1e-6), 0.0, 2.0)
    reflex = np.tanh(0.55 * s + 0.30 * feedback - 0.15 * vol_z)
    return np.asarray(np.clip(reflex, -1.0, 1.0), float)


def build_min_sleeves(runs_dir: Path | None = None) -> dict:
    runs = Path(runs_dir) if runs_dir is not None else RUNS
    runs.mkdir(parents=True, exist_ok=True)
    df = _load_main(runs / "portfolio_plus.csv")  # DATE, ret_main

    r = df["ret_main"].values

    # VOL sleeve: low realized vol -> +, high vol -> -
    vol20 = pd.Series(r).rolling(20, min_periods=10).std()
    vol100 = pd.Series(r).rolling(100, min_periods=30).std()
    vol_rel = (vol20 / vol100.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    sig_vol = (1.0 - vol_rel).clip(-1.0, 1.0).fillna(0.0)
    ret_vol = (sig_vol * VOL_SCALE).clip(-0.05, 0.05)
    pd.DataFrame({"DATE": df["DATE"], "ret": ret_vol}).to_csv(runs / "sleeve_vol.csv", index=False)

    # OSC sleeve: fade short-term extremes
    z10 = _zscore(r, 10)
    sig_osc = (-z10).clip(-1.0, 1.0)
    ret_osc = (sig_osc * OSC_SCALE).clip(-0.05, 0.05)
    pd.DataFrame({"DATE": df["DATE"], "ret": ret_osc}).to_csv(runs / "sleeve_osc.csv", index=False)

    # Symbolic stream: preserve existing module output, fallback only if missing.
    sym_df, sym_src = _load_signal_existing(
        runs / "symbolic_signal.csv",
        dates=df["DATE"],
        signal_candidates=["sym_signal", "symbolic_signal", "signal"],
        conf_candidates=["confidence", "sym_confidence"],
        out_col="sym_signal",
    )
    if sym_df is None:
        sym_vals = _fallback_symbolic(r)
        sym_df = pd.DataFrame({"DATE": df["DATE"], "sym_signal": sym_vals})
        sym_src = f"fallback({sym_src})"
    sym_df["sym_signal"] = _safe_num(sym_df["sym_signal"]).fillna(SYM_DEFAULT).clip(-1.0, 1.0)
    sym_df.to_csv(runs / "symbolic_signal.csv", index=False)

    # Reflexive stream: preserve existing module output, fallback only if missing.
    ref_df, ref_src = _load_signal_existing(
        runs / "reflexive_signal.csv",
        dates=df["DATE"],
        signal_candidates=["reflexive_signal", "reflex_signal", "signal"],
        conf_candidates=["reflex_confidence", "confidence"],
        out_col="reflexive_signal",
    )
    if ref_df is None:
        ref_vals = _fallback_reflexive(r, sym_df["sym_signal"].values)
        ref_df = pd.DataFrame({"DATE": df["DATE"], "reflexive_signal": ref_vals})
        ref_src = f"fallback({ref_src})"
    ref_df["reflexive_signal"] = _safe_num(ref_df["reflexive_signal"]).fillna(RFX_DEFAULT).clip(-1.0, 1.0)
    ref_df.to_csv(runs / "reflexive_signal.csv", index=False)

    info = {
        "rows": int(len(df)),
        "symbolic_source": sym_src,
        "reflexive_source": ref_src,
        "vol_scale": float(VOL_SCALE),
        "osc_scale": float(OSC_SCALE),
    }
    (runs / "min_sleeves_info.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    info = build_min_sleeves()
    print(
        "✅ Wrote: sleeve_vol.csv, sleeve_osc.csv, symbolic_signal.csv, reflexive_signal.csv "
        f"(sym={info['symbolic_source']}, reflex={info['reflexive_source']})"
    )
