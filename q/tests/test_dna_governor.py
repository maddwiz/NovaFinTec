import numpy as np

from qmods.dna_governor import build_dna_stress_governor


def test_dna_stress_governor_bounds():
    T = 320
    t = np.linspace(0.0, 10.0, T)
    drift = 0.2 + 0.1 * np.sin(t)
    vel = np.gradient(drift)
    z = (drift - np.mean(drift)) / (np.std(drift) + 1e-9)
    st = np.where(z > 0.8, 1.0, np.where(z < -0.8, -1.0, 0.0))

    stress, gov, info = build_dna_stress_governor(
        drift=drift,
        velocity=vel,
        drift_z=z,
        regime_state=st,
        lo=0.72,
        hi=1.12,
        smooth=0.88,
    )
    assert stress.shape == (T,)
    assert gov.shape == (T,)
    assert np.isfinite(stress).all() and np.isfinite(gov).all()
    assert float(np.min(stress)) >= 0.0 - 1e-9
    assert float(np.max(stress)) <= 1.0 + 1e-9
    assert float(np.min(gov)) >= 0.72 - 1e-9
    assert float(np.max(gov)) <= 1.12 + 1e-9
    assert info["status"] == "ok"


def test_dna_stress_governor_penalizes_higher_drift():
    T = 280
    low_drift = np.full(T, 0.05, dtype=float)
    hi_drift = np.full(T, 0.40, dtype=float)
    z_low = np.full(T, -0.5, dtype=float)
    z_hi = np.full(T, 1.2, dtype=float)
    st_low = np.full(T, -1.0, dtype=float)
    st_hi = np.full(T, 1.0, dtype=float)

    s1, g1, _ = build_dna_stress_governor(
        drift=low_drift,
        velocity=np.zeros(T),
        drift_z=z_low,
        regime_state=st_low,
    )
    s2, g2, _ = build_dna_stress_governor(
        drift=hi_drift,
        velocity=np.zeros(T),
        drift_z=z_hi,
        regime_state=st_hi,
    )
    assert float(np.mean(s2)) > float(np.mean(s1))
    assert float(np.mean(g2)) < float(np.mean(g1))


def test_dna_stress_governor_penalizes_acceleration_and_transitions():
    T = 300
    t = np.linspace(0.0, 12.0, T)
    drift = 0.15 + 0.05 * np.sin(t)
    vel = np.gradient(drift)
    z = (drift - np.mean(drift)) / (np.std(drift) + 1e-9)

    calm_state = np.zeros(T, dtype=float)
    flip_state = np.where(np.sin(2.6 * t) > 0, 1.0, -1.0)
    low_acc = np.zeros(T, dtype=float)
    hi_acc = np.clip(np.gradient(vel) * 8.0, 0.0, None)

    s1, g1, _ = build_dna_stress_governor(
        drift=drift,
        velocity=vel,
        acceleration=low_acc,
        drift_z=z,
        regime_state=calm_state,
    )
    s2, g2, _ = build_dna_stress_governor(
        drift=drift,
        velocity=vel,
        acceleration=hi_acc,
        drift_z=z,
        regime_state=flip_state,
    )

    assert float(np.mean(s2)) > float(np.mean(s1))
    assert float(np.mean(g2)) < float(np.mean(g1))
