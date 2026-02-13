# qmods/dreams.py — v0.5 moving dreams (feature-aware RGB dreams)
import numpy as np
import imageio.v2 as imageio

def _norm01(x):
    x = np.asarray(x, dtype=float)
    # replace non-finite with nan
    x[~np.isfinite(x)] = np.nan
    # robust min/max with nan support
    lo = np.nanmin(x) if np.any(np.isfinite(x)) else 0.0
    hi = np.nanmax(x) if np.any(np.isfinite(x)) else 1.0
    span = hi - lo if (hi - lo) > 1e-9 else 1.0
    y = (x - lo) / span
    y = np.where(np.isfinite(y), y, 0.0)
    return y

def _tile_strip(x, H, W):
    """Make an HxW image by tiling the last W values of x across rows."""
    if x.size == 0:
        return np.zeros((H, W), dtype=float)
    xw = x[-W:] if x.size >= W else np.pad(x, (W - x.size, 0), mode='edge')
    return np.tile(xw, (H, 1))

def _dream_features(series):
    s = np.asarray(series, dtype=float)
    if s.size == 0:
        return np.zeros(1), np.zeros(1), np.zeros(1)
    r = np.r_[0.0, np.diff(np.log(np.maximum(s, 1e-12)))]
    mom = np.convolve(r, np.ones(8) / 8.0, mode="same")
    vol = np.sqrt(np.convolve((r - r.mean()) ** 2, np.ones(16) / 16.0, mode="same"))
    return _norm01(s), _norm01(mom), _norm01(vol)

def save_dream_gif(series, out_path, frames=120, step=3, fps=14):
    lvl, mom, vol = _dream_features(series)
    if lvl.size == 0:
        return
    # use sizes divisible by 16 to avoid ffmpeg warnings
    H, W = 160, 160
    imgs = []
    r_layer = np.zeros((H, W), dtype=float)
    g_layer = np.zeros((H, W), dtype=float)
    b_layer = np.zeros((H, W), dtype=float)
    # precompute a gentle swirl
    rr = np.sin(np.linspace(0, 2*np.pi, H))[:, None]
    cc = np.cos(np.linspace(0, 2*np.pi, W))[None, :]
    swirl = rr * cc
    for i in range(frames):
        j = min(lvl.size, 1 + i*step)
        sl = _tile_strip(lvl[:j], H, W)
        sm = _tile_strip(mom[:j], H, W)
        sv = _tile_strip(vol[:j], H, W)
        # smooth persistence layers
        r_layer = 0.90*r_layer + 0.10*sl
        g_layer = 0.92*g_layer + 0.08*np.clip(sm + 0.5, 0.0, 1.0)
        b_layer = 0.93*b_layer + 0.07*sv
        r = np.clip(0.8*r_layer + 0.2*swirl, 0.0, 1.0)
        g = np.clip(0.85*g_layer + 0.15*(1.0 - swirl), 0.0, 1.0)
        b = np.clip(0.9*b_layer + 0.1*np.abs(swirl), 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
        imgs.append((255*rgb).astype(np.uint8))
    imageio.mimsave(out_path, imgs, duration=1.0/max(fps,1))

def save_dream_mp4(series, out_path, frames=180, step=2, fps=18):
    try:
        import imageio_ffmpeg  # noqa: F401
    except Exception:
        return  # skip MP4 if backend not present
    lvl, mom, vol = _dream_features(series)
    if lvl.size == 0:
        return
    # sizes divisible by 16
    H, W = 192, 192
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=6)
    r_layer = np.zeros((H, W), dtype=float)
    g_layer = np.zeros((H, W), dtype=float)
    b_layer = np.zeros((H, W), dtype=float)
    for i in range(frames):
        j = min(lvl.size, 1 + i*step)
        sl = _tile_strip(lvl[:j], H, W)
        sm = _tile_strip(mom[:j], H, W)
        sv = _tile_strip(vol[:j], H, W)
        r_layer = 0.90*r_layer + 0.10*sl
        g_layer = 0.92*g_layer + 0.08*np.clip(sm + 0.5, 0.0, 1.0)
        b_layer = 0.93*b_layer + 0.07*sv
        pulse = np.sin(2*np.pi*(i/30.0))*0.15
        r = np.clip(r_layer*(0.85 + pulse), 0.0, 1.0)
        g = np.clip(g_layer*(0.90 - pulse), 0.0, 1.0)
        b = np.clip(b_layer*(0.80 + 0.5*pulse), 0.0, 1.0)
        frame = (255*np.stack([r, g, b], axis=-1)).astype(np.uint8)
        writer.append_data(frame)
    writer.close()
