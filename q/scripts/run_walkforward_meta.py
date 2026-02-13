import argparse, pathlib, json
import pandas as pd
import numpy as np

def walkforward(data):
    returns = data["Close"].pct_change().dropna()
    hits = (np.sign(returns) == np.sign(returns.shift(-1))).sum()
    total = returns.shape[0]
    hit_rate = hits / total if total else 0.0
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() else 0.0
    max_dd = (data["Close"].cummax() - data["Close"]).max() / data["Close"].cummax().max()
    return {"hit_rate": float(hit_rate), "sharpe": float(sharpe), "max_dd": float(max_dd)}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--asset", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    path = pathlib.Path(args.data) / args.asset
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    metrics = walkforward(df)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result = {"asset": args.asset, **metrics}

    # Save to file
    with (out / "summary.json").open("w") as f:
        json.dump(result, f)

    # Print to screen
    print(json.dumps(result))
