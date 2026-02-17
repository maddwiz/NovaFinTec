import json
from pathlib import Path

import tools.run_runtime_combo_search as rcs


def test_runtime_combo_search_selects_best_valid(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rcs, "ROOT", tmp_path)
    monkeypatch.setattr(rcs, "RUNS", runs)

    monkeypatch.setenv("Q_RUNTIME_SEARCH_FLOORS", "0.18,0.22")
    monkeypatch.setenv("Q_RUNTIME_SEARCH_FLAGS", "a,b")

    monkeypatch.setattr(rcs, "_base_runtime_env", lambda: {})

    def _fake_eval(env):
        floor = float(env.get("Q_RUNTIME_TOTAL_FLOOR", "0.18"))
        disabled = {x for x in env.get("Q_DISABLE_GOVERNORS", "").split(",") if x}
        # Best valid case: floor 0.18 with only b disabled.
        sharpe = 1.0
        if floor == 0.18 and disabled == {"b"}:
            sharpe = 1.6
        elif floor == 0.22 and disabled == {"a"}:
            sharpe = 1.4
        return {
            "robust_sharpe": sharpe,
            "robust_hit_rate": 0.50,
            "robust_max_drawdown": -0.04,
            "promotion_ok": True,
            "cost_stress_ok": True,
            "health_ok": True,
            "health_alerts_hard": 0,
            "rc": [{"step": "x", "code": 0}],
        }

    monkeypatch.setattr(rcs, "_eval_combo", _fake_eval)
    rc = rcs.main()
    assert rc == 0

    sel = json.loads((runs / "runtime_profile_selected.json").read_text(encoding="utf-8"))
    assert abs(float(sel["runtime_total_floor"]) - 0.18) < 1e-9
    assert sel["disable_governors"] == ["b"]
