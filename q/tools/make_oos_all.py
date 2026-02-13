#!/usr/bin/env python3
"""
make_oos_all.py v3 (walk-forward council)

Builds per-asset OOS files (`runs_plus/<asset>/oos.csv`) using the same
momentum/mean-reversion/carry council logic as the walk-forward evaluators.
This replaces the synthetic EMA/z-score placeholder baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from qmods.io import load_close
from qmods.meta_council import carry_signal, meanrev_signal, momentum_signal

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
np.seterr(all="ignore")

DATA = ROOT / "data"
RUNS = ROOT / "runs_plus"

GRID = [
    (0.9, 0.1, 0.0),
    (0.8, 0.2, 0.0),
    (0.7, 0.2, 0.1),
    (0.6, 0.3, 0.1),
    (0.5, 0.4, 0.1),
    (0.4, 0.4, 0.2),
    (0.3, 0.5, 0.2),
    (0.2, 0.6, 0.2),
    (0.1, 0.7, 0.2),
    (0.0, 0.8, 0.2),
]


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_max_drawdown(equity: pd.Series) -> float:
    eq = pd.to_numeric(equity, errors="coerce").ffill().fillna(1.0).clip(lower=1e-9)
    log_eq = np.log(eq)
    peak = np.maximum.accumulate(log_eq.values)
    dd = np.exp(log_eq.values - peak) - 1.0
    return float(np.nanmin(dd)) if dd.size else float("nan")


def sharpe_ratio(pnl: pd.Series) -> float:
    s = pd.to_numeric(pnl, errors="coerce").dropna()
    if s.size == 0:
        return float("nan")
    return float(s.mean() / (s.std() + 1e-9) * math.sqrt(252.0))


def hit_ratio(pnl: pd.Series, ret: pd.Series) -> float:
    a = np.sign(pd.to_numeric(pnl, errors="coerce").fillna(0.0).values)
    b = np.sign(pd.to_numeric(ret, errors="coerce").fillna(0.0).values)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float((a == b).mean())


def build_returns(close: pd.Series) -> pd.Series:
    # Use log returns to match walk-forward evaluators.
    r = np.diff(np.log(np.maximum(close.values.astype(float), 1e-12)))
    ret = pd.Series(np.r_[0.0, r], index=close.index)
    return ret.clip(-0.20, 0.20)


def eval_on_prefix(ret: np.ndarray, pos: np.ndarray, cost_bps: float) -> float:
    if len(ret) <= 2:
        return -1e9
    pos_lag = np.roll(pos, 1)
    pos_lag[0] = 0.0
    turnover = np.abs(np.diff(np.r_[0.0, pos]))
    cost = cost_bps / 10000.0
    pnl = pos_lag * ret - turnover * cost
    sd = float(np.std(pnl, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(pnl) / sd * np.sqrt(252.0))


def build_oos_positions(close: pd.Series, train_min: int, step: int, cost_bps: float) -> tuple[pd.Series, pd.Series]:
    ret = build_returns(close)

    mom = np.asarray(momentum_signal(close), float)
    mr = np.asarray(meanrev_signal(close), float)
    car = np.asarray(carry_signal(close), float)

    n = len(close)
    pos = np.zeros(n, dtype=float)

    for t in range(train_min, n, step):
        best_w = None
        best_sh = -1e12

        for wm, wr, wc in GRID:
            meta_train = wm * mom[:t] + wr * mr[:t] + wc * car[:t]
            pos_train = np.tanh(meta_train)
            sh = eval_on_prefix(ret.values[:t], pos_train, cost_bps)
            if sh > best_sh:
                best_sh = sh
                best_w = (wm, wr, wc)

        if best_w is None:
            continue

        wm, wr, wc = best_w
        end = min(t + step, n)
        meta_test = wm * mom[t:end] + wr * mr[t:end] + wc * car[t:end]
        pos[t:end] = np.tanh(meta_test)

    # Keep pre-train window flat to avoid accidental in-sample leakage.
    pos[: min(train_min, n)] = 0.0
    return pd.Series(pos, index=close.index), ret


def eval_and_write(sym: str, close: pd.Series, pos: pd.Series, ret: pd.Series, cost_bps: float) -> None:
    df = pd.DataFrame(
        {
            "date": close.index,
            "price": close.values,
            "ret": ret.values,
            "pos": pos.values,
        }
    )
    turnover = df["pos"].diff().abs().fillna(0.0)
    df["pnl"] = df["pos"].shift(1).fillna(0.0) * df["ret"] - turnover * (cost_bps / 10000.0)
    df["pnl"] = df["pnl"].clip(-0.95, 0.95)
    df["equity"] = (1.0 + df["pnl"]).cumprod()

    hit = hit_ratio(df["pnl"], df["ret"])
    sh = sharpe_ratio(df["pnl"])
    mdd = safe_max_drawdown(df["equity"])

    out_dir = RUNS / sym
    safe_mkdir(out_dir)
    df.to_csv(out_dir / "oos.csv", index=False)
    (out_dir / "summary_plus.json").write_text(
        json.dumps(
            {
                "asset": sym,
                "hit": hit,
                "sharpe": sh,
                "maxDD": mdd,
            },
            indent=2,
        )
    )
    print(f"✅ {sym}: oos.csv written ({len(df)} rows)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_min", type=int, default=504, help="Minimum training bars before OOS starts.")
    ap.add_argument("--step", type=int, default=21, help="Walk-forward step size (bars).")
    ap.add_argument("--cost_bps", type=float, default=1.0, help="Turnover cost in bps.")
    ap.add_argument("--min_rows", type=int, default=600, help="Minimum series length to evaluate.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not DATA.exists():
        raise SystemExit("data/ folder not found.")

    count = 0
    for p in sorted(DATA.glob("*.csv")):
        sym = p.stem.replace("_prices", "")
        try:
            close = load_close(p).astype(float).ffill().dropna()
            if len(close) < args.min_rows:
                print(f"skip {sym}: not enough history ({len(close)} rows)")
                continue

            pos, ret = build_oos_positions(
                close=close,
                train_min=max(120, int(args.train_min)),
                step=max(5, int(args.step)),
                cost_bps=float(args.cost_bps),
            )
            eval_and_write(sym=sym, close=close, pos=pos, ret=ret, cost_bps=float(args.cost_bps))
            count += 1
        except Exception as exc:
            print(f"skip {sym}: error {exc}")

    print(f"\n✅ Done. Wrote oos.csv for {count} assets.")


if __name__ == "__main__":
    main()
