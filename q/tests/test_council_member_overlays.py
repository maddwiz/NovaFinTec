import numpy as np
import pandas as pd

import tools.make_council_votes as mcv
import tools.run_credit_council_member as rccm
import tools.run_microstructure_council_member as rmcm
from qengine.bandit import ExpWeightsBandit


def test_credit_member_broadcast_mapping():
    sig = np.array([0.5, -0.5], dtype=float)
    names = ["SPY", "HYG", "TLT", "XLF"]

    mat, gains = rccm.build_credit_council_member(
        sig,
        names,
        direct_symbols={"SPY"},
        inverse_symbols={"HYG"},
        attenuated_symbols={"TLT"},
        attenuation=0.30,
        default_gain=1.0,
    )

    assert mat.shape == (2, 4)
    assert np.allclose(gains, np.array([1.0, -1.0, 0.30, 1.0]))
    assert np.allclose(mat[0], np.array([0.5, -0.5, 0.15, 0.5]))
    assert np.allclose(mat[1], np.array([-0.5, 0.5, -0.15, -0.5]))


def test_microstructure_member_broadcast_mapping():
    sig = np.array([0.8, -0.2], dtype=float)
    names = ["QQQ", "IEF", "GLD"]

    mat, gains = rmcm.build_microstructure_council_member(
        sig,
        names,
        direct_symbols={"QQQ"},
        inverse_symbols={"IEF"},
        attenuated_symbols={"GLD"},
        attenuation=0.25,
        default_gain=1.0,
    )

    assert mat.shape == (2, 3)
    assert np.allclose(gains, np.array([1.0, -1.0, 0.25]))
    assert np.allclose(mat[0], np.array([0.8, -0.8, 0.2]))
    assert np.allclose(mat[1], np.array([-0.2, 0.2, -0.05]))


def test_credit_member_main_writes_zero_matrix_when_signal_missing(tmp_path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)

    np.savetxt(runs / "asset_returns.csv", np.zeros((6, 3), dtype=float), delimiter=",")
    (runs / "asset_names.csv").write_text("asset\nSPY\nHYG\nTLT\n", encoding="utf-8")

    monkeypatch.setattr(rccm, "ROOT", root)
    monkeypatch.setattr(rccm, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = rccm.main()
    assert rc == 0

    out = np.loadtxt(runs / "council_credit_leadlag.csv", delimiter=",")
    assert out.shape == (6, 3)
    assert np.allclose(out, np.zeros((6, 3)))


def test_make_council_votes_augments_optional_members(tmp_path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)

    base = np.arange(12, dtype=float).reshape(6, 2)
    cred = np.full((5, 3), 2.0, dtype=float)
    micro = np.full((7, 1), -1.0, dtype=float)

    np.savetxt(runs / "council_credit_leadlag.csv", cred, delimiter=",")
    np.savetxt(runs / "council_microstructure.csv", micro, delimiter=",")

    monkeypatch.setattr(mcv, "ROOT", root)
    monkeypatch.setattr(mcv, "RUNS", runs)

    aug, info = mcv._augment_with_optional_members(base)

    assert aug.shape == (5, 6)
    assert np.allclose(aug[:, :2], base[-5:])
    assert int(info["optional_member_count"]) == 2
    files = {row["file"] for row in info["optional_members"]}
    assert files == {"council_credit_leadlag.csv", "council_microstructure.csv"}


def test_bandit_handles_additional_members_without_oos_degradation():
    rng = np.random.default_rng(123)
    t = 240
    split = 160

    x = np.linspace(0.0, 20.0, t)
    y = np.linspace(0.0, 18.0, t)

    s0 = np.sign(np.sin(x) + 0.10 * rng.normal(size=t))
    s1 = np.sign(rng.normal(size=t))
    s2 = np.sign(np.cos(y) + 0.10 * rng.normal(size=t))
    s0[s0 == 0] = 1.0
    s1[s1 == 0] = 1.0
    s2[s2 == 0] = 1.0

    ret = np.zeros(t, dtype=float)
    for i in range(1, t):
        # Base alpha mostly from s0; extra member s2 adds small orthogonal signal.
        ret[i] = 0.006 * s0[i - 1] + 0.005 * s2[i - 1] + 0.002 * rng.normal()

    def _fit_weights(sig_mat: np.ndarray) -> np.ndarray:
        idx = pd.RangeIndex(split)
        signals = {f"s{i}": pd.Series(sig_mat[:split, i], index=idx) for i in range(sig_mat.shape[1])}
        returns = pd.Series(ret[:split], index=idx)
        w = ExpWeightsBandit(eta=0.6).fit(signals, returns).get_weights()
        vec = np.asarray([float(w.get(f"s{i}", 0.0)) for i in range(sig_mat.shape[1])], dtype=float)
        return vec

    def _oos_hit_rate(sig_mat: np.ndarray, w: np.ndarray) -> float:
        preds = []
        reals = []
        for i in range(split, t):
            p = float(np.sign(np.dot(w, np.sign(sig_mat[i - 1]))))
            r = float(np.sign(ret[i]))
            if r == 0.0:
                continue
            preds.append(p)
            reals.append(r)
        if not preds:
            return 0.5
        return float(np.mean(np.asarray(preds) == np.asarray(reals)))

    base = np.column_stack([s0, s1]).astype(float)
    aug = np.column_stack([s0, s1, s2]).astype(float)

    w_base = _fit_weights(base)
    w_aug = _fit_weights(aug)
    hr_base = _oos_hit_rate(base, w_base)
    hr_aug = _oos_hit_rate(aug, w_aug)

    assert np.isfinite(w_base).all()
    assert np.isfinite(w_aug).all()
    assert np.isclose(float(w_base.sum()), 1.0, atol=1e-9)
    assert np.isclose(float(w_aug.sum()), 1.0, atol=1e-9)
    assert hr_aug >= hr_base - 1e-9
