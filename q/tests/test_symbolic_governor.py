import numpy as np

from qmods.symbolic_governor import build_symbolic_governor


def test_symbolic_governor_bounds_and_shape():
    T = 300
    t = np.linspace(0.0, 8.0, T)
    s = 0.4 * np.sin(t)
    a = np.clip(0.3 + 0.2 * np.cos(t), 0.0, 1.0)
    c = np.clip(0.6 + 0.2 * np.sin(0.5 * t), 0.0, 1.0)
    n = np.clip(2.0 + 1.5 * np.sin(0.8 * t), 0.0, None)
    st, g, info = build_symbolic_governor(s, a, c, n, lo=0.72, hi=1.12, smooth=0.88)

    assert st.shape == (T,)
    assert g.shape == (T,)
    assert np.isfinite(st).all() and np.isfinite(g).all()
    assert float(np.min(st)) >= 0.0 - 1e-9
    assert float(np.max(st)) <= 1.0 + 1e-9
    assert float(np.min(g)) >= 0.72 - 1e-9
    assert float(np.max(g)) <= 1.12 + 1e-9
    assert info["status"] == "ok"


def test_symbolic_governor_penalizes_negative_affective_state():
    T = 260
    calm_sig = np.full(T, 0.4, dtype=float)
    risk_sig = np.full(T, -0.6, dtype=float)
    calm_aff = np.full(T, 0.1, dtype=float)
    risk_aff = np.full(T, 0.8, dtype=float)
    conf = np.full(T, 0.8, dtype=float)
    ev = np.full(T, 2.0, dtype=float)

    s1, g1, _ = build_symbolic_governor(calm_sig, calm_aff, conf, ev)
    s2, g2, _ = build_symbolic_governor(risk_sig, risk_aff, conf, ev)
    assert float(np.mean(s2)) > float(np.mean(s1))
    assert float(np.mean(g2)) < float(np.mean(g1))
