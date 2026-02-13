#!/usr/bin/env python3
# Apply execution/live constraints to portfolio weights.
#
# Reads:
#   runs_plus/portfolio_weights_final.csv
#   runs_plus/asset_names.csv (optional)
#   config/execution_constraints.json (optional)
#
# Writes:
#   runs_plus/portfolio_weights_exec.csv
#   runs_plus/execution_constraints_info.json
# Optionally overwrites portfolio_weights_final.csv with --replace-final.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
CFG = ROOT / "config" / "execution_constraints.json"
RUNS.mkdir(exist_ok=True)


def _load_weights():
    p = RUNS / "portfolio_weights_final.csv"
    if not p.exists():
        return None
    try:
        a = np.loadtxt(p, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(p, delimiter=",", skiprows=1)
        except Exception:
            return None
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _load_assets(n_cols: int):
    p = RUNS / "asset_names.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            if len(df.columns):
                c0 = df.columns[0]
                syms = [str(x).upper().strip() for x in df[c0].tolist() if str(x).strip()]
                if len(syms) >= n_cols:
                    return syms[:n_cols]
        except Exception:
            pass
    return [f"ASSET_{i+1}" for i in range(n_cols)]


def _load_config():
    cfg = {
        "allow_shorts": True,
        "max_abs_weight": 0.25,
        "short_blocklist": [],
        "long_blocklist": [],
        "symbol_caps": {},
        "session_scales": {"regular": 1.0, "after_hours": 0.60, "closed": 0.0},
        "renormalize_to_gross": True,
    }
    if CFG.exists():
        try:
            raw = json.loads(CFG.read_text())
            if isinstance(raw, dict):
                cfg.update(raw)
        except Exception:
            pass
    return cfg


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace-final", action="store_true", help="Overwrite portfolio_weights_final.csv with constrained weights.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    W = _load_weights()
    if W is None:
        raise SystemExit("Missing runs_plus/portfolio_weights_final.csv")

    cfg = _load_config()
    assets = _load_assets(W.shape[1])
    idx = {a: i for i, a in enumerate(assets)}

    out = np.asarray(W, float).copy()
    gross0 = np.sum(np.abs(out), axis=1)

    max_abs = float(np.clip(float(cfg.get("max_abs_weight", 0.25)), 0.01, 2.0))
    out = np.clip(out, -max_abs, max_abs)

    allow_shorts = bool(cfg.get("allow_shorts", True))
    if not allow_shorts:
        out = np.clip(out, 0.0, None)

    for sym in [str(x).upper() for x in cfg.get("short_blocklist", [])]:
        if sym in idx:
            out[:, idx[sym]] = np.clip(out[:, idx[sym]], 0.0, None)
    for sym in [str(x).upper() for x in cfg.get("long_blocklist", [])]:
        if sym in idx:
            out[:, idx[sym]] = np.clip(out[:, idx[sym]], None, 0.0)

    sym_caps = cfg.get("symbol_caps", {}) or {}
    if isinstance(sym_caps, dict):
        for sym, cap in sym_caps.items():
            s = str(sym).upper()
            if s not in idx:
                continue
            c = float(np.clip(float(cap), 0.0, 2.0))
            out[:, idx[s]] = np.clip(out[:, idx[s]], -c, c)

    session = str(os.getenv("Q_SESSION_MODE", "regular")).strip().lower()
    session_scales = cfg.get("session_scales", {}) or {}
    sess_scale = float(session_scales.get(session, session_scales.get("regular", 1.0)))
    sess_scale = float(np.clip(sess_scale, 0.0, 2.0))
    out *= sess_scale

    if bool(cfg.get("renormalize_to_gross", True)):
        gross1 = np.sum(np.abs(out), axis=1)
        target = np.minimum(gross0, np.full_like(gross0, max_abs * out.shape[1]))
        scale = np.divide(target, gross1 + 1e-12)
        scale = np.clip(scale, 0.0, 3.0)
        out = out * scale.reshape(-1, 1)

    outp = RUNS / "portfolio_weights_exec.csv"
    np.savetxt(outp, out, delimiter=",")

    if args.replace_final:
        np.savetxt(RUNS / "portfolio_weights_final.csv", out, delimiter=",")

    info = {
        "rows": int(out.shape[0]),
        "cols": int(out.shape[1]),
        "allow_shorts": allow_shorts,
        "max_abs_weight": max_abs,
        "session_mode": session,
        "session_scale": sess_scale,
        "replace_final": bool(args.replace_final),
        "gross_before_mean": float(np.mean(gross0)),
        "gross_after_mean": float(np.mean(np.sum(np.abs(out), axis=1))),
    }
    (RUNS / "execution_constraints_info.json").write_text(json.dumps(info, indent=2))
    print(f"✅ Wrote {outp}")
    print(f"✅ Wrote {RUNS/'execution_constraints_info.json'}")
    if args.replace_final:
        print(f"✅ Replaced {RUNS/'portfolio_weights_final.csv'}")
