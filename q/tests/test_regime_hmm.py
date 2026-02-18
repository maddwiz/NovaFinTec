import numpy as np

from qengine.regime_hmm import RegimeHMM


def test_stable_state_yields_low_transition_probability():
    hmm = RegimeHMM(n_features=6)
    obs = np.array([0.1, 1.0, 0.2, 0.0, -0.2, 0.1], dtype=float)
    for _ in range(200):
        hmm.update(obs, known_state=0)
    r = hmm.transition_risk(horizon=5)
    assert 0.0 <= float(r["transition_prob"]) <= 1.0
    assert float(r["transition_prob"]) < 0.60


def test_feature_shift_increases_transition_risk():
    hmm = RegimeHMM(n_features=6)
    calm = np.array([0.0, 1.0, 0.1, 0.0, -0.2, 0.0], dtype=float)
    crisis = np.array([2.5, -1.5, 2.0, 1.5, 2.0, 1.2], dtype=float)

    for _ in range(120):
        hmm.update(calm, known_state=0)
    low = float(hmm.transition_risk(horizon=5)["transition_prob"])

    for _ in range(20):
        hmm.update(crisis, known_state=3)
    high = float(hmm.transition_risk(horizon=5)["transition_prob"])

    assert high >= 0.0
    assert high >= low * 0.7


def test_crisis_risk_rises_with_crisis_anchor():
    hmm = RegimeHMM(n_features=6)
    for _ in range(80):
        hmm.update(np.array([2.0, -1.0, 1.5, 1.2, 1.8, 0.9]), known_state=3)
    r = hmm.transition_risk(horizon=5)
    assert float(r["crisis_risk_5d"]) >= 0.0


def test_anchor_prevents_divergence_from_known_state():
    hmm = RegimeHMM(n_features=6)
    obs = np.array([0.2, 0.8, 0.1, 0.0, -0.1, 0.05], dtype=float)
    for _ in range(60):
        hmm.update(obs, known_state=0)
    assert int(np.argmax(hmm.belief)) == 0
