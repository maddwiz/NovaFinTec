from qmods.novaspine_hive import build_hive_query, hive_boost, hive_resonance


def test_hive_resonance_bounds_and_order():
    r0 = hive_resonance(0, 4, [])
    r1 = hive_resonance(2, 4, [0.2, 0.3])
    r2 = hive_resonance(4, 4, [0.8, 0.9, 1.0])
    assert 0.0 <= r0 <= 1.0
    assert 0.0 <= r1 <= 1.0
    assert 0.0 <= r2 <= 1.0
    assert r2 > r1 > r0


def test_hive_boost_behavior():
    b_disabled = hive_boost(1.0, status_ok=False)
    b_low = hive_boost(0.1, status_ok=True)
    b_high = hive_boost(0.9, status_ok=True)
    assert abs(b_disabled - 1.0) < 1e-9
    assert 0.85 <= b_low <= 1.10
    assert 0.85 <= b_high <= 1.10
    assert b_high > b_low


def test_build_hive_query_contains_inputs():
    q = build_hive_query("fx", 1.23, 0.62, -0.19)
    assert "FX" in q
    assert "1.230" in q
    assert "0.620" in q
    assert "-0.190" in q
