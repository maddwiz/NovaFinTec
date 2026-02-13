import numpy as np

from qmods.dream_coherence import build_dream_coherence_governor


def test_dream_coherence_bounds_and_shape():
    T = 320
    t = np.linspace(0.0, 12.0, T)
    ret = 0.004 * np.sin(t) + 0.0015 * np.cos(1.3 * t)
    sig = {
        "reflex_latent": 0.9 * np.sin(t - 0.2),
        "symbolic_latent": 0.8 * np.sin(t - 0.15),
        "meta_mix": 0.7 * np.sin(t - 0.1),
    }
    g, info = build_dream_coherence_governor(sig, ret)
    assert g.shape == (T,)
    assert np.isfinite(g).all()
    assert float(np.min(g)) >= 0.70 - 1e-9
    assert float(np.max(g)) <= 1.15 + 1e-9
    assert info["status"] == "ok"
    assert len(info["signals"]) == 3


def test_dream_coherence_rewards_consistency():
    T = 360
    t = np.linspace(0.0, 15.0, T)
    rng = np.random.default_rng(7)
    ret = 0.0035 * np.sin(t) + 0.001 * np.cos(1.7 * t) + 0.0005 * rng.standard_normal(T)

    consistent = {
        "reflex_latent": np.sin(t - 0.10),
        "symbolic_latent": 0.95 * np.sin(t - 0.12),
        "meta_mix": 0.85 * np.sin(t - 0.08),
        "synapses_pred": 0.70 * np.sin(t - 0.05),
    }
    incoherent = {
        "reflex_latent": rng.standard_normal(T),
        "symbolic_latent": rng.standard_normal(T),
        "meta_mix": rng.standard_normal(T),
        "synapses_pred": rng.standard_normal(T),
    }

    g1, i1 = build_dream_coherence_governor(consistent, ret)
    g2, i2 = build_dream_coherence_governor(incoherent, ret)

    assert float(np.mean(g1)) > float(np.mean(g2))
    assert float(i1["mean_coherence"]) > float(i2["mean_coherence"])


def test_dream_coherence_causal_delay_alignment_detects_lead():
    T = 420
    t = np.linspace(0.0, 20.0, T)
    rng = np.random.default_rng(11)

    # Synthetic return process with smooth structure.
    ret = 0.0032 * np.sin(t) + 0.0010 * np.cos(1.1 * t) + 0.0004 * rng.standard_normal(T)
    base = np.sin(t - 0.15)

    # This stream leads the target process; causal delay should be selected (>0).
    lead_stream = np.roll(base, -2)
    lead_stream[-2:] = lead_stream[-3]

    sig = {
        "lead_stream": lead_stream,
        "support_stream": 0.7 * base + 0.2 * rng.standard_normal(T),
    }

    g0, i0 = build_dream_coherence_governor(sig, ret, max_causal_delay=0)
    g1, i1 = build_dream_coherence_governor(sig, ret, max_causal_delay=3)

    assert g0.shape == g1.shape
    d = i1.get("per_signal_causal_delay", {}).get("lead_stream", 0)
    assert int(d) >= 1
    assert float(i1["mean_coherence"]) >= float(i0["mean_coherence"]) - 1e-6


def test_dream_coherence_shock_penalty_reduces_high_vol_exposure():
    T = 420
    t = np.linspace(0.0, 18.0, T)
    rng = np.random.default_rng(23)

    base = np.sin(t - 0.12)
    ret = 0.0022 * base + 0.0006 * rng.standard_normal(T)
    ret[T // 2 :] = 0.0068 * base[T // 2 :] + 0.0030 * rng.standard_normal(T // 2)

    sig = {
        "reflex_latent": 0.95 * base,
        "symbolic_latent": 0.85 * base,
        "meta_mix": 0.80 * base,
    }
    g, info = build_dream_coherence_governor(sig, ret)
    lo = float(np.mean(g[: T // 2]))
    hi = float(np.mean(g[T // 2 :]))

    assert info["mean_shock_penalty"] <= 1.0
    assert hi < lo
