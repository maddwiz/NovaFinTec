import pathlib, pandas as pd
from qmods.overlay import save_overlay_gif

DATA = pathlib.Path("data")
OUT  = pathlib.Path("runs_plus")
# choose the pairs you want (you can add more later)
PAIRS = [("IWM","RSP"), ("LQD_TR","HYG_TR")]

for a, b in PAIRS:
    df_a = pd.read_csv(DATA/f"{a}.csv", parse_dates=["Date"], index_col="Date")
    df_b = pd.read_csv(DATA/f"{b}.csv", parse_dates=["Date"], index_col="Date")
    close_a = df_a["Close"].astype(float).to_numpy()
    close_b = df_b["Close"].astype(float).to_numpy()
    save_overlay_gif(close_a, close_b, OUT/f"{a}__{b}", name="overlay", frames=80, step=5, fps=12)
    print(f"Wrote overlay for {a}__{b} → {OUT}/{a}__{b}/overlay.gif")
