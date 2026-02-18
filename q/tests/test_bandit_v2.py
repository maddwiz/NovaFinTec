import numpy as np
import pandas as pd

from qengine.bandit_v2 import ThompsonBandit


def test_consistently_correct_arm_accumulates_alpha():
    b = ThompsonBandit(n_arms=2, decay=1.0, magnitude_scaling=False)
    for _ in range(30):
        b.update(0, reward=1.0, magnitude=1.0)
        b.update(1, reward=0.0, magnitude=1.0)
    assert float(b.alpha[0]) > float(b.beta_[0])
    assert float(b.beta_[1]) > float(b.alpha[1])


def test_decay_fades_old_information():
    b = ThompsonBandit(n_arms=1, decay=0.95, magnitude_scaling=False)
    b.update(0, reward=1.0, magnitude=1.0)
    a0 = float(b.alpha[0])
    for _ in range(15):
        b.update(0, reward=0.0, magnitude=1.0)
    assert float(b.alpha[0]) < a0 + 5.0


def test_magnitude_scaling_gives_larger_credit():
    b1 = ThompsonBandit(n_arms=1, decay=1.0, magnitude_scaling=True)
    b2 = ThompsonBandit(n_arms=1, decay=1.0, magnitude_scaling=True)
    b1.update(0, reward=1.0, magnitude=0.005)
    b2.update(0, reward=1.0, magnitude=0.05)
    assert float(b2.alpha[0]) > float(b1.alpha[0])


def test_fit_get_weights_compatibility():
    idx = pd.RangeIndex(120)
    sig = {
        "a": pd.Series(np.sign(np.sin(np.arange(120) / 8.0)), index=idx),
        "b": pd.Series(np.sign(np.cos(np.arange(120) / 6.0)), index=idx),
    }
    ret = pd.Series(np.sign(np.sin((np.arange(120) + 1) / 8.0)) * 0.01, index=idx)
    b = ThompsonBandit(n_arms=2).fit(sig, ret)
    w = b.get_weights()
    assert set(w.keys()) == {"a", "b"}
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_confidence_intervals_and_effective_sample_size_finite():
    b = ThompsonBandit(n_arms=3)
    for _ in range(10):
        b.update_all(np.array([1.0, 0.0, 1.0]), np.array([0.02, 0.01, 0.03]))
    ci = b.confidence_intervals()
    ess = b.effective_sample_size()
    assert ci.shape == (3,)
    assert ess.shape == (3,)
    assert np.isfinite(ci).all()
    assert np.isfinite(ess).all()
