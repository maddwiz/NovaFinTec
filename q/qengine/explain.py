
import pandas as pd, numpy as np

def explain_trades(test_df: pd.DataFrame, pos: pd.Series, notes: str=""):
    df = pd.DataFrame(index=test_df.index)
    df["pos"] = pos.reindex(df.index).fillna(0.0)
    if "dna_drift" in test_df.columns:
        df["dna_drift"] = test_df["dna_drift"]
    if "dna_vel" in test_df.columns:
        df["dna_vel"] = test_df["dna_vel"]
    if "vix" in test_df.columns:
        df["vix"] = test_df["vix"]
    df["notes"] = notes
    return df
