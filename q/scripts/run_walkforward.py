#!/usr/bin/env python3
import argparse, os, json
import pandas as pd
from pathlib import Path
from qengine.data import load_csv
from qengine.signals import dna_signal, trend_signal, momentum_signal
from qengine.bandit import ExpWeightsBandit
from qengine.bandit_v2 import ThompsonBandit
from qengine.risk import apply_risk
from qengine.crisis import crisis_anchor_from_vix, crisis_anchor_from_internal
from qengine.walkforward import walkforward
from qengine.explain import explain_trades
from qengine.anomaly import anomaly_triage, apply_quarantine


def _fit_bandit(signals: dict, returns: pd.Series, eta: float):
    bandit_type = str(os.getenv("Q_BANDIT_TYPE", "thompson")).strip().lower()
    if bandit_type == "thompson":
        prior_file = str(os.getenv("Q_BANDIT_PRIOR_FILE", "")).strip() or None
        return ThompsonBandit(
            n_arms=len(signals),
            decay=float(os.getenv("Q_THOMPSON_DECAY", "0.995")),
            magnitude_scaling=str(os.getenv("Q_THOMPSON_MAGNITUDE_SCALING", "1")).strip().lower()
            not in {"0", "false", "off", "no"},
            prior_file=prior_file,
        ).fit(signals, returns)
    return ExpWeightsBandit(eta=eta).fit(signals, returns)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder with CSVs")
    ap.add_argument("--asset", required=True, help="Filename of CSV, e.g., SPY.csv")
    ap.add_argument("--vix", default="", help="Optional VIX CSV for crisis anchor")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--target_vol", type=float, default=0.20)
    ap.add_argument("--cap", type=float, default=0.25)
    ap.add_argument("--dd_th", type=float, default=-0.10)
    ap.add_argument("--max_flips_60", type=int, default=6)
    ap.add_argument("--eta", type=float, default=0.4, help="Bandit learning rate")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    df = load_csv(os.path.join(args.data, args.asset))
    feats = pd.DataFrame(index=df.index)
    feats["Close"] = df["Close"]

    # base signals
    dna_sig, drift, vel = dna_signal(df["Close"])
    feats["dna_sig"]   = dna_sig
    feats["dna_drift"] = drift
    feats["dna_vel"]   = vel
    feats["trend_sig"] = trend_signal(df["Close"])
    feats["mom_sig"]   = momentum_signal(df["Close"])

    # crisis anchors (use VIX if available; otherwise internal)
    if args.vix and os.path.exists(os.path.join(args.data, args.vix)):
        vix = load_csv(os.path.join(args.data, args.vix)).reindex(feats.index).ffill()
        feats["vix"] = vix["Close"]
        crisis_mult_full = crisis_anchor_from_vix(vix["Close"])
    else:
        crisis_mult_full = crisis_anchor_from_internal(feats["dna_drift"])

    # anomaly triage for quarantine
    quarantine_full = anomaly_triage(feats["Close"], feats["dna_drift"], z_th=4.0)

    def make_position(train_df, test_df):
        # fit bandit on train
        signals_train = {
            "dna":   train_df["dna_sig"],
            "trend": train_df["trend_sig"],
            "mom":   train_df["mom_sig"],
        }
        returns_train = train_df["Close"].pct_change().fillna(0.0)
        bandit = _fit_bandit(signals_train, returns_train, eta=args.eta)
        W = bandit.get_weights()

        # frozen weights on test (no leakage)
        signals_test = {
            "dna":   test_df["dna_sig"],
            "trend": test_df["trend_sig"],
            "mom":   test_df["mom_sig"],
        }
        ens = None
        for k, s in signals_test.items():
            w = W.get(k, 0.0)
            ens = s*w if ens is None else ens + s*w
        ens = ens.apply(lambda x: 1.0 if x>0 else (-1.0 if x<0 else 0.0))

        # crisis multiplier
        cm = crisis_mult_full.reindex(test_df.index).fillna(1.0)
        ens = ens * cm

        # risk management
        pos, strat_r, eq = apply_risk(ens, test_df["Close"],
                                      target_vol=args.target_vol,
                                      cap=args.cap,
                                      dd_th=args.dd_th,
                                      max_flips_60=args.max_flips_60)

        # quarantine: skip flagged days
        quarantine = quarantine_full.reindex(test_df.index).fillna(False)
        pos = apply_quarantine(pos, quarantine, scale=0.0)
        return pos

    # walk-forward
    full_pos, metrics = walkforward(feats, make_position)
    metrics.to_csv(outdir/"metrics_per_fold.csv", index=False)

    # explanations + equity
    test_df = feats.loc[full_pos.index]
    cards = explain_trades(test_df, full_pos, notes="bandit(dna,trend,mom); crisis; risk; quarantine")
    cards.to_csv(outdir/"explain_cards.csv")

    r = (df["Close"].loc[full_pos.index]).pct_change().fillna(0.0)
    strat = full_pos.shift(1).fillna(0.0) * r
    eq = (1.0 + strat).cumprod()
    eq.to_csv(outdir/"equity_curve.csv", header=["equity"])

    # summary
    hit = ((full_pos.shift(1).fillna(0.0).apply(lambda x: 0 if x==0 else (1 if x>0 else -1))) ==
           (r.apply(lambda x: 0 if x==0 else (1 if x>0 else -1)))).mean()
    import numpy as np
    sharpe = (strat.mean()*252) / (strat.std(ddof=1)*(252**0.5)) if strat.std(ddof=1)>0 else 0.0
    peak = eq.cummax(); mdd = (eq/peak - 1.0).min()
    summary = {"asset": args.asset, "hit_rate": float(hit), "sharpe": float(sharpe), "max_dd": float(mdd)}
    with open(outdir/"summary.json","w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
