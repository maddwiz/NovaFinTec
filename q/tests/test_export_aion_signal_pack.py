from pathlib import Path

import tools.export_aion_signal_pack as ex


def _write_series(path: Path, vals):
    path.write_text("\n".join(str(v) for v in vals), encoding="utf-8")


def test_runtime_context_uses_governor_components(tmp_path: Path):
    _write_series(tmp_path / "global_governor.csv", [0.9, 0.8])
    _write_series(tmp_path / "quality_governor.csv", [1.0, 0.95])
    _write_series(tmp_path / "quality_runtime_modifier.csv", [1.0, 0.9])
    _write_series(tmp_path / "meta_mix_reliability_governor.csv", [0.98, 1.02])
    _write_series(tmp_path / "dna_stress_governor.csv", [0.95, 0.90])
    _write_series(tmp_path / "reflex_health_governor.csv", [0.96, 1.00])
    _write_series(tmp_path / "symbolic_governor.csv", [0.94, 0.98])
    _write_series(tmp_path / "novaspine_context_boost.csv", [1.0, 1.05])

    ctx = ex._runtime_context(tmp_path)
    assert 0.50 <= ctx["runtime_multiplier"] <= 1.10
    assert ctx["active_component_count"] >= 5
    assert ctx["components"]["global_governor"]["found"] is True
    assert ctx["components"]["meta_mix_reliability_governor"]["found"] is True
    assert ctx["components"]["dna_stress_governor"]["found"] is True
    assert ctx["components"]["reflex_health_governor"]["found"] is True
    assert ctx["components"]["symbolic_governor"]["found"] is True


def test_runtime_context_defaults_to_neutral_when_missing(tmp_path: Path):
    ctx = ex._runtime_context(tmp_path)
    assert ctx["runtime_multiplier"] == 1.0
    assert ctx["active_component_count"] == 0
    assert ctx["regime"] == "risk_on"
