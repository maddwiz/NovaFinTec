import numpy as np, pandas as pd, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_signals_png(date_index, close, meta, drift, bpm, out_path: pathlib.Path):
    plt.figure(figsize=(10,6))
    ax1 = plt.gca()
    ax1.plot(date_index, close/np.nanmax(close), label="Close (norm)")
    ax1.plot(date_index, (meta-np.nanmin(meta))/(np.nanmax(meta)-np.nanmin(meta)+1e-9), label="Meta (norm)", alpha=0.7)
    ax1.set_ylabel("normed price/meta")
    ax2 = ax1.twinx()
    ax2.plot(date_index, drift, label="DNA drift", alpha=0.5)
    ax2.plot(date_index, bpm/200.0, label="Heartbeat/200", alpha=0.5)
    ax2.set_ylabel("drift / bpm scaled")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines+lines2, labels+labels2, loc="upper left")
    plt.title("Signals overview")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
