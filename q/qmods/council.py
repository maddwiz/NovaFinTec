from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd


@dataclass
class CouncilVote:
    name: str
    score: pd.Series  # one score per symbol in [-1, 1]


class CouncilMember:
    name: str = "BaseMember"

    def signal_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def vote(self, ctx: dict) -> CouncilVote:
        sig = self.signal_matrix(ctx["prices"])
        if sig.empty:
            return CouncilVote(self.name, pd.Series(dtype=float))
        return CouncilVote(self.name, sig.iloc[-1].fillna(0.0).clip(-1.0, 1.0))


class MomentumRep(CouncilMember):
    name = "MomentumRep"

    def __init__(self, lookback=63):
        self.lb = int(lookback)

    def signal_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        s = prices.pct_change(self.lb)
        return (s / 0.15).clip(-1.0, 1.0)


class MeanRevRep(CouncilMember):
    name = "MeanRevRep"

    def __init__(self, lookback=5):
        self.lb = int(lookback)

    def signal_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        s = -prices.pct_change(self.lb)
        return (s / 0.05).clip(-1.0, 1.0)


class CarryRep(CouncilMember):
    name = "CarryRep"

    def signal_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        ma_fast = prices.rolling(21, min_periods=10).mean()
        ma_slow = prices.rolling(126, min_periods=30).mean()
        gap = (ma_fast - ma_slow) / (ma_slow.abs() + 1e-12)
        return (gap / 0.08).clip(-1.0, 1.0)


class VolBreakoutRep(CouncilMember):
    name = "VolBreakoutRep"

    def signal_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        r1 = prices.pct_change()
        vol = r1.rolling(63, min_periods=20).std(ddof=1)
        z = r1 / (vol + 1e-12)
        return z.clip(-2.5, 2.5) / 2.5


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) < 8:
        return np.nan
    return float(x[m].rank(pct=True).corr(y[m].rank(pct=True)))


def _member_quality(sig: pd.DataFrame, fwd: pd.DataFrame, lookback: int = 252) -> float:
    return _member_stats(sig, fwd, lookback=lookback)["quality"]


def _member_stats(sig: pd.DataFrame, fwd: pd.DataFrame, lookback: int = 252) -> dict[str, float]:
    if sig.empty or fwd.empty:
        return {"quality": 0.10, "polarity": 0.0, "mean_ic": 0.0, "samples": 0.0}
    idx = sig.index.intersection(fwd.index)
    if len(idx) == 0:
        return {"quality": 0.10, "polarity": 0.0, "mean_ic": 0.0, "samples": 0.0}
    idx = idx[-max(40, int(lookback)) :]
    ics = []
    for t in idx:
        ic = _rank_corr(sig.loc[t], fwd.loc[t])
        if np.isfinite(ic):
            ics.append(ic)
    if not ics:
        return {"quality": 0.10, "polarity": 0.0, "mean_ic": 0.0, "samples": 0.0}
    mean_ic = float(np.mean(ics))
    abs_ic = abs(mean_ic)
    # Reliability uses absolute IC so anti-predictive members are still useful once flipped.
    quality = float(np.clip(0.10 + 0.90 * np.tanh(3.0 * abs_ic), 0.10, 1.00))

    # Polarity maps IC to [-1, 1], with a neutral deadband near zero to reduce noise.
    polarity = float(np.clip(mean_ic / 0.03, -1.0, 1.0))
    if abs(polarity) < 0.15:
        polarity = 0.0

    return {
        "quality": quality,
        "polarity": polarity,
        "mean_ic": mean_ic,
        "samples": float(len(ics)),
    }


def aggregate(votes: list[CouncilVote], qualities: dict[str, float] | None = None) -> tuple[pd.Series, dict]:
    if not votes:
        return pd.Series(dtype=float), {"confidence": 0.0, "disagreement": 0.0}

    df = pd.concat([v.score for v in votes], axis=1)
    df.columns = [v.name for v in votes]
    df = df.fillna(0.0)

    if qualities is None:
        qualities = {}
    w = np.array([float(qualities.get(name, 1.0)) for name in df.columns], dtype=float)
    w = np.clip(w, 0.05, None)
    w = w / (w.sum() + 1e-12)

    raw = (df.values * w.reshape(1, -1)).sum(axis=1)
    raw = pd.Series(raw, index=df.index).clip(-1.0, 1.0)

    disagreement = float(df.std(axis=1).mean()) if df.shape[1] > 1 else 0.0
    confidence = float(np.clip(1.0 - disagreement, 0.35, 1.0))
    raw = raw * confidence

    denom = raw.abs().sum()
    final_w = raw / denom if denom > 0 else raw * 0.0
    meta = {"confidence": confidence, "disagreement": disagreement}
    return final_w, meta


def run_council(prices: pd.DataFrame, out_json="runs_plus/council.json"):
    prices = prices.sort_index().ffill().dropna(how="all")
    fwd = prices.pct_change().shift(-1).replace([np.inf, -np.inf], np.nan)

    members = [MomentumRep(), MeanRevRep(), CarryRep(), VolBreakoutRep()]
    ctx = {"prices": prices}
    votes = []
    mats = {}
    qualities = {}
    polarities = {}
    member_ic = {}
    member_samples = {}

    for m in members:
        mat = m.signal_matrix(prices).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        mats[m.name] = mat
        stats = _member_stats(mat, fwd, lookback=252)
        qualities[m.name] = float(stats["quality"])
        polarities[m.name] = float(stats["polarity"])
        member_ic[m.name] = float(stats["mean_ic"])
        member_samples[m.name] = float(stats["samples"])
        votes.append(CouncilVote(m.name, mat.iloc[-1].fillna(0.0) * float(stats["polarity"])))

    final_w, meta = aggregate(votes, qualities=qualities)
    out = {
        "members": {v.name: v.score.to_dict() for v in votes},
        "member_quality": {k: float(v) for k, v in qualities.items()},
        "member_polarity": {k: float(v) for k, v in polarities.items()},
        "member_mean_ic": {k: float(v) for k, v in member_ic.items()},
        "member_ic_samples": {k: float(v) for k, v in member_samples.items()},
        "ensemble_confidence": float(meta["confidence"]),
        "ensemble_disagreement": float(meta["disagreement"]),
        "final_weights": final_w.to_dict(),
    }
    Path(out_json).write_text(json.dumps(out, indent=2))
    return out
