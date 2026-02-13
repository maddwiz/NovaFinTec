# tools/walk_forward_batch.py — run OOS walk-forward for all assets, build a table
import json, math
from pathlib import Path
import pandas as pd
from qmods.io import load_close
from qmods.meta_council import momentum_signal, meanrev_signal, carry_signal

DATA = Path("data")
RUNS = Path("runs_plus")
WF   = RUNS / "walk_forward"
WF.mkdir(parents=True, exist_ok=True)

def safe_equity(close, pos, cost_bps):
    import numpy as np
    r = np.diff(pd.Series(close).apply(lambda x: math.log(max(x, 1e-12))).values)
    ret = pd.Series([0.0] + list(pd.Series(r).clip(-0.20, 0.20)), index=close.index)
    pos = pd.Series(pos, index=close.index).fillna(0.0)
    pos_lag = pos.shift(1).fillna(0.0)
    turnover = pos.diff().abs().fillna(0.0)
    cost = cost_bps / 10000.0
    strat = pos_lag * ret - turnover * cost
    eq = (1 + strat).cumprod()
    hit = float((strat.shift(1).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) ==
                 strat.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))).mean())
    sd = float(strat.std(ddof=1))
    sh = float((strat.mean() / sd) * (252 ** 0.5)) if sd > 1e-12 else 0.0
    mdd = float((eq/eq.cummax() - 1).min())
    return hit, sh, mdd

def wf_one(close: pd.Series, cost_bps=1.0, train_min=504, step=21):
    import numpy as np
    mom = momentum_signal(close)
    mr  = meanrev_signal(close)
    car = carry_signal(close)
    grid = [
        (0.9,0.1,0.0),(0.8,0.2,0.0),(0.7,0.2,0.1),
        (0.6,0.3,0.1),(0.5,0.4,0.1),(0.4,0.4,0.2),
        (0.3,0.5,0.2),(0.2,0.6,0.2),(0.1,0.7,0.2),(0.0,0.8,0.2)
    ]
    pos_all = pd.Series(0.0, index=close.index)
    for t in range(train_min, len(close), step):
        best = None; best_sh = -9e9
        for wm, wr, wc in grid:
            meta = wm*mom[:t] + wr*mr[:t] + wc*car[:t]
            pos  = pd.Series(np.tanh(meta), index=close.index[:t])
            hit, sh, mdd = safe_equity(close.iloc[:t], pos, cost_bps)
            if sh > best_sh:
                best_sh = sh; best = (wm, wr, wc)
        wm, wr, wc = best
        meta_f = wm*mom[:t+step] + wr*mr[:t+step] + wc*car[:t+step]
        pos_all.iloc[:t+step] = pd.Series(np.tanh(meta_f), index=close.index[:t+step])
    return safe_equity(close, pos_all, cost_bps)

def main():
    rows = []
    for csv in sorted(DATA.glob("*.csv")):
        if csv.stem.lower() == "news":  # skip
            continue
        try:
            close = load_close(csv)
            if close is None or len(close) < 600:
                continue
            hit, sh, mdd = wf_one(close)
            (WF / f"{csv.stem}.json").write_text(json.dumps({"asset": csv.stem, "hit": hit, "sharpe": sh, "max_dd": mdd}, indent=2))
            rows.append({"asset": csv.stem, "hit": hit, "sharpe": sh, "max_dd": mdd})
            print(f"{csv.stem:10s}  hit={hit:.3f}  sharpe={sh:.3f}  maxDD={mdd:.3f}")
        except Exception as e:
            print("skip", csv.stem, "->", e)
            continue
    df = pd.DataFrame(rows).sort_values(["sharpe","hit"], ascending=False)
    df.to_json(RUNS/"walk_forward_table.json", orient="records", indent=2)
    df.to_csv(RUNS/"walk_forward_table.csv", index=False)
    print("Wrote:", RUNS/"walk_forward_table.json")
    print("Wrote:", RUNS/"walk_forward_table.csv")

if __name__ == "__main__":
    main()
