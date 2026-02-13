from qmods.symbolic import _score_text


def test_symbolic_score_positive_vs_negative():
    sent_pos, aff_pos, conf_pos, _, _ = _score_text("Very strong bullish breakout and profit expansion!")
    sent_neg, aff_neg, conf_neg, _, _ = _score_text("Bearish warning with losses and failure.")

    assert sent_pos > sent_neg
    assert conf_pos > 0.0
    assert conf_neg > 0.0
    assert aff_pos >= 0.0
    assert aff_neg >= 0.0


def test_symbolic_negation_and_fear_behavior():
    sent_a, aff_a, conf_a, _, aux_a = _score_text("bullish breakout")
    sent_b, aff_b, conf_b, _, aux_b = _score_text("not bullish breakout")
    sent_c, aff_c, conf_c, _, aux_c = _score_text("panic crisis crash!!!")

    assert sent_b < sent_a
    assert aff_c > aff_a
    assert conf_c >= conf_a
    assert aux_b["neg"] >= 0.0
    assert aux_c["fear"] > 0.0
    assert 0.0 <= conf_b <= 1.0
    assert 0.0 <= conf_c <= 1.0
