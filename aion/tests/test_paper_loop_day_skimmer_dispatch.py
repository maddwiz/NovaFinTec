from aion.exec import paper_loop
from aion.exec import skimmer_loop


def test_main_dispatches_to_day_skimmer(monkeypatch):
    called = {"n": 0}

    def _fake_run():
        called["n"] += 1
        return 17

    monkeypatch.setattr(paper_loop.cfg, "TRADING_MODE", "day_skimmer", raising=False)
    monkeypatch.setattr(skimmer_loop, "run_day_skimmer_loop", _fake_run)

    rc = paper_loop.main()
    assert rc == 17
    assert called["n"] == 1
