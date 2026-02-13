import numpy as np

from qmods.synapses_small import SynapseEnsemble


def test_synapse_ensemble_predict_shapes_and_bounds():
    rng = np.random.default_rng(7)
    X = rng.normal(0.0, 1.0, size=(220, 5))
    y = 0.25 * X[:, 0] - 0.15 * X[:, 1] + 0.1 * np.tanh(X[:, 2]) + rng.normal(0.0, 0.1, size=220)

    m = SynapseEnsemble(n_models=3, hidden=8, epochs=120, patience=20, sample_frac=0.8, seed=11)
    m.fit(X, y)
    p = m.predict(X)
    c = m.predict_confidence(X)

    assert p.shape == (220,)
    assert c.shape == (220,)
    assert np.isfinite(p).all()
    assert np.isfinite(c).all()
    assert np.min(c) >= 0.0
    assert np.max(c) <= 1.0
    assert m.pred_std_ is not None
    assert np.min(m.pred_std_) >= 0.0
