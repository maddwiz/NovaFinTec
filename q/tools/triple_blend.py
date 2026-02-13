#!/usr/bin/env python3
# tools/triple_blend.py
# v8: time-series CV + tighter caps + stronger cost + dispersion & realism guards

from pathlib import Path
import pandas as pd, numpy as np, json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"

MAIN_P = RUNS / "portfolio_plus.csv"
VOL_P  = RUNS / "vol_overlay_costed.csv"
OSC_P  = RUNS / "osc_portfolio_costed.csv"

# ---- knobs ----
GRID_STEP        = 0.05
W_VOL_MAX        = 0.20       # tighter cap
W_OSC_MAX        = 0.15       # tighter cap
RIDGE_LAMBDA     = 0.15       # stronger pull to Main=[1,0,0]
N_FOLDS          = 4
TEST_FRAC        = 0.15
ROLL_STD_WIN     = 252
ROLL_STD_MIN     = 0.0002
TARGET_FLOOR     = 0.005
SCALE_MIN        = 0.20
SCALE_MAX        = 5.00
COST_FINAL_BPS   = 3.0        # stronger daily cost on final blend
# guards:
OOS_SPIKE_MULT   = 1.25       # OOS must be <= 1.25 * IS
OOS_SPIKE_PLUS   = 0.30       # and also <= IS + 0.30
OOS_DD_MIN       = 0.03       # require at least 3% drawdown magnitude
CV_STD_MAX       = 0.60       # std of fold OOS Sharpes must be <= 0.60

def safe(s):
    return pd.Series(pd.to_numeric(s, errors="coerce")).replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(-0.5,0.5)

def ann_sharpe(r):
    r = pd.Series(r).replace([np.inf,-np.inf], np.nan).dropna()
    s = r.std()
    if s == 0 or np.isnan(s): return 0.0
    return float((r.mean()/s)*(252.0**0.5))

def dd_min(r):
    eq = (1.0 + r).cumprod()
    peak = pd.concat([eq, pd.Series(1.0, index=eq.index)], axis=1).max(axis=1).cummax()
    dd = (eq/peak - 1.0)
    return float(dd.min())

def smart_load(path: Path, prefer="auto"):
    if not path.exists(): raise SystemExit(f"Missing {path}")
    df = pd.read_csv(path)
    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or lowers.get("timestamp") or lowers.get("time") or df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol)

    cand = None
    if prefer == "net" and "ret_net" in df.columns:
        cand = "ret_net"
    if cand is None:
        for c in ["ret","ret_plus","ret_gross","return","pnl","pnl_plus","daily_ret","portfolio_ret","port_ret"]:
            if c in df.columns: cand = c; break
    if cand:
        r = safe(df[cand])
        return pd.DataFrame({"DATE": df[dcol], "ret": r})

    for c in ["eq_net","eq","equity","equity_curve","portfolio_eq","equity_index","port_equity"]:
        if c in df.columns:
            eq = pd.to_numeric(df[c], errors="coerce")
            r = safe(eq.pct_change())
            return pd.DataFrame({"DATE": df[dcol], "ret": r})
    raise SystemExit(f"No returns/equity columns found in {path.name}")

def first_active_idx(r: pd.Series, win=ROLL_STD_WIN, thresh=ROLL_STD_MIN):
    rs = pd.Series(r).rolling(win).std()
    idx = rs[rs > thresh].index
    return int(idx[0]) if len(idx) else int(len(r) * 0.5)

def choose_target_std(std_list):
    vals = [float(v) for v in std_list if np.isfinite(v) and v > 0]
    return float(max(np.median(vals), TARGET_FLOOR)) if vals else TARGET_FLOOR

def weight_grid():
    step = GRID_STEP
    for wm in np.arange(0.0, 1.0 + 1e-9, step):
        # cap vol weight
        vmax = min(1.0 - wm, W_VOL_MAX)
        for wv in np.arange(0.0, vmax + 1e-9, step):
            wo = 1.0 - wm - wv
            if wo < -1e-9: continue
            if wo > W_OSC_MAX + 1e-9: continue
            yield float(wm), float(wv), float(wo)

def cv_folds(n_rows, n_folds=N_FOLDS, test_frac=TEST_FRAC):
    test = max(50, int(n_rows * test_frac))
    train_min = max(200, int(n_rows * (1 - n_folds*test_frac)))
    starts = []
    i = train_min
    for _ in range(n_folds):
        if i + test > n_rows: break
        starts.append((0, i, i, i+test))
        i += test
    return starts

if __name__ == "__main__":
    main = smart_load(MAIN_P, prefer="auto").rename(columns={"ret":"ret_main"})
    vol  = smart_load(VOL_P,  prefer="net").rename(columns={"ret":"ret_vol"})
    osc  = smart_load(OSC_P,  prefer="net").rename(columns={"ret":"ret_osc"})

    df = main.merge(vol, on="DATE", how="inner").merge(osc, on="DATE", how="inner").sort_values("DATE")
    if df.empty: raise SystemExit("No overlapping dates across Main, Vol NET, Osc NET.")

    # Auto-trim early flat era
    start_idx = first_active_idx(df["ret_main"])
    if start_idx > 0:
        df = df.iloc[start_idx:].reset_index(drop=True)

    n = len(df)
    if n < 1000: raise SystemExit("Not enough rows after trimming; lower ROLL_STD_MIN or check data.")
    folds = cv_folds(n)
    if not folds: raise SystemExit("Could not form CV folds; increase data or adjust TEST_FRAC/N_FOLDS.")

    # CV search
    cv_rows = []
    w0 = np.array([1.0,0.0,0.0])
    for (tr_s, tr_e, te_s, te_e) in folds:
        tr = df.iloc[tr_s:tr_e]; te = df.iloc[te_s:te_e]
        rm_tr, rv_tr, ro_tr = safe(tr["ret_main"]), safe(tr["ret_vol"]), safe(tr["ret_osc"])
        rm_te, rv_te, ro_te = safe(te["ret_main"]), safe(te["ret_vol"]), safe(te["ret_osc"])

        # scale add-ons toward target (search only)
        target = choose_target_std([rm_tr.std(), rv_tr.std(), ro_tr.std()])
        eps = 1e-12
        kv = target / max(rv_tr.std(), eps); kv = float(np.clip(kv, SCALE_MIN, SCALE_MAX))
        ko = target / max(ro_tr.std(), eps); ko = float(np.clip(ko, SCALE_MIN, SCALE_MAX))

        rm_tr_s, rv_tr_s, ro_tr_s = rm_tr, rv_tr*kv, ro_tr*ko
        rm_te_s, rv_te_s, ro_te_s = rm_te, rv_te*kv, ro_te*ko

        # pick best by ridge-penalized IS Sharpe
        best_score = -9
        best = (1.0,0.0,0.0)
        for wm,wv,wo in weight_grid():
            r = wm*rm_tr_s + wv*rv_tr_s + wo*ro_tr_s
            sh = ann_sharpe(r)
            w = np.array([wm,wv,wo])
            score = sh - RIDGE_LAMBDA*float(((w - w0)**2).sum())
            if score > best_score:
                best_score = score
                best = (wm,wv,wo)

        wm,wv,wo = best
        r_te = wm*rm_te_s + wv*rv_te_s + wo*ro_te_s
        sh_te = ann_sharpe(r_te); hit_te=float((r_te>0).mean()); dd_te=dd_min(r_te)

        cv_rows.append({
            "train_rows": len(tr), "test_rows": len(te),
            "train_end": str(tr["DATE"].iloc[-1].date()), "test_end": str(te["DATE"].iloc[-1].date()),
            "w_main": wm, "w_vol": wv, "w_osc": wo,
            "oos_sharpe": sh_te, "oos_hit": hit_te, "oos_maxdd": dd_te
        })

    cv_df = pd.DataFrame(cv_rows)

    # Use trimmed-mean weights (drop best & worst fold by oos_sharpe)
    if len(cv_df) >= 3:
        order = cv_df.sort_values("oos_sharpe").index.tolist()
        keep = order[1:-1]
        use = cv_df.loc[keep]
    else:
        use = cv_df
    wm = float(use["w_main"].mean())
    wv = float(use["w_vol"].mean())
    wo = float(use["w_osc"].mean())

    # Final series (UNSCALED) with conservative daily cost
    rm_fu, rv_fu, ro_fu = safe(df["ret_main"]), safe(df["ret_vol"]), safe(df["ret_osc"])
    ret_cost = COST_FINAL_BPS / 10000.0
    r_full_gross = wm*rm_fu + wv*rv_fu + wo*ro_fu
    r_full = r_full_gross - ret_cost
    eq_full = (1.0 + r_full).cumprod()
    out = pd.DataFrame({
        "DATE": df["DATE"], "ret": r_full, "eq": eq_full,
        "ret_gross": r_full_gross, "ret_cost_per_day": ret_cost,
        "w_main": wm, "w_vol": wv, "w_osc": wo
    })
    out.to_csv(RUNS/"final_portfolio.csv", index=False)

    # Report IS/OOS on last fold (closest to today) with FINAL weights
    last = cv_df.iloc[-1]
    te_end_date = last["test_end"]
    te_end_idx = df.index[df["DATE"] <= pd.to_datetime(te_end_date)].max()
    fold_len = int(last["test_rows"])
    te_slice = slice(te_end_idx - fold_len + 1, te_end_idx + 1)
    tr_slice = slice(0, te_end_idx - fold_len + 1)

    r_is_g = wm*rm_fu.iloc[tr_slice] + wv*rv_fu.iloc[tr_slice] + wo*ro_fu.iloc[tr_slice]
    r_oos_g = wm*rm_fu.iloc[te_slice] + wv*rv_fu.iloc[te_slice] + wo*ro_fu.iloc[te_slice]
    r_is  = r_is_g - ret_cost
    r_oos = r_oos_g - ret_cost

    is_sh, is_hit, is_dd = ann_sharpe(r_is), float((r_is>0).mean()), dd_min(r_is)
    oos_sh, oos_hit, oos_dd = ann_sharpe(r_oos), float((r_oos>0).mean()), dd_min(r_oos)

    # extra realism guards
    guard_reason = ""
    cv_std = float(cv_df["oos_sharpe"].std()) if len(cv_df) > 1 else 0.0
    if (oos_sh > min(OOS_SPIKE_MULT*max(is_sh,1e-6), is_sh + OOS_SPIKE_PLUS)) or (abs(oos_dd) < OOS_DD_MIN) or (cv_std > CV_STD_MAX):
        wm,wv,wo = 1.0,0.0,0.0
        r_full_gross = rm_fu
        r_full = r_full_gross - ret_cost
        eq_full = (1.0 + r_full).cumprod()
        out = pd.DataFrame({
            "DATE": df["DATE"], "ret": r_full, "eq": eq_full,
            "ret_gross": r_full_gross, "ret_cost_per_day": ret_cost,
            "w_main": wm, "w_vol": wv, "w_osc": wo
        })
        out.to_csv(RUNS/"final_portfolio.csv", index=False)
        r_is_g = rm_fu.iloc[tr_slice]; r_oos_g = rm_fu.iloc[te_slice]
        r_is, r_oos = r_is_g - ret_cost, r_oos_g - ret_cost
        is_sh, is_hit, is_dd = ann_sharpe(r_is), float((r_is>0).mean()), dd_min(r_is)
        oos_sh, oos_hit, oos_dd = ann_sharpe(r_oos), float((r_oos>0).mean()), dd_min(r_oos)
        guard_reason = "FALLBACK_TO_MAIN (dispersion/spike/DD guard)"

    summary = {
        "weights": {"w_main": wm, "w_vol": wv, "w_osc": wo},
        "cv": {
            "folds": cv_rows,
            "median_oos_sharpe": float(cv_df["oos_sharpe"].median()) if not cv_df.empty else None,
            "mean_oos_sharpe":   float(cv_df["oos_sharpe"].mean())   if not cv_df.empty else None,
            "std_oos_sharpe":    float(cv_std)
        },
        "costs": {"final_cost_bps_per_day": COST_FINAL_BPS},
        "in_sample":  {"sharpe": is_sh,  "hit": is_hit,  "maxdd": is_dd},
        "out_sample": {"sharpe": oos_sh, "hit": oos_hit, "maxdd": oos_dd},
        "trim": {"start_index": int(start_idx), "start_date": str(df['DATE'].iloc[0].date())},
        "guards": {"reason": guard_reason,
                   "oos_limit_mult": OOS_SPIKE_MULT, "oos_limit_plus": OOS_SPIKE_PLUS,
                   "oos_dd_min": OOS_DD_MIN, "cv_std_max": CV_STD_MAX}
    }
    RUNS.joinpath("final_portfolio_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nTRIPLE BLEND v8 (CV + tighter caps + stronger cost + dispersion guard)")
    print("=======================================================================")
    print(f"Rows after trim: {n} | Folds: {len(folds)} (test≈{int(n*TEST_FRAC)} each)")
    print(f"Chosen weights (trimmed-mean across folds): w_main={wm:.2f} w_vol={wv:.2f} w_osc={wo:.2f}")
    if guard_reason:
        print("GUARD:", guard_reason, "→ final weights: w_main=1.00 w_vol=0.00 w_osc=0.00")
    print(f"IS : Sharpe={is_sh:.3f} | Hit={is_hit:.3f} | MaxDD={is_dd:.3f}")
    print(f"OOS: Sharpe={oos_sh:.3f} | Hit={oos_hit:.3f} | MaxDD={oos_dd:.3f}")
    print(f"Conservative daily cost applied: {COST_FINAL_BPS:.2f} bps/day")
    print("Saved: runs_plus/final_portfolio.csv and final_portfolio_summary.json")
