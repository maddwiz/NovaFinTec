from qmods.hive_transparency import build_hive_snapshot


def test_build_hive_snapshot_merges_and_orders_by_weight():
    out = build_hive_snapshot(
        hive_names=["eq", "fx"],
        metrics_by_hive={
            "EQ": {"sharpe_oos": 1.2, "hit_rate": 0.56, "max_dd": -0.12},
            "FX": {"sharpe_oos": 0.8, "hit_rate": 0.53, "max_dd": -0.09},
        },
        latest_weights={"FX": 0.2, "EQ": 0.6, "RATES": 0.1},
        feedback_by_hive={"EQ": {"boost": 1.05, "resonance": 0.8, "status": "ok"}},
    )

    rows = out["rows"]
    assert [r["hive"] for r in rows][:2] == ["EQ", "FX"]
    assert rows[0]["novaspine_boost"] == 1.05
    assert rows[1]["novaspine_boost"] == 1.0
    assert any(r["hive"] == "RATES" for r in rows)

    summary = out["summary"]
    assert summary["hive_count"] == 3
    assert summary["top_hive"] == "EQ"
    assert summary["weight_l1"] >= 0.9
