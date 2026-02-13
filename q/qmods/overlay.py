import numpy as np, pathlib, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def _latent(x: np.ndarray, w: int = 64) -> np.ndarray:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size < w + 2:
        x = np.pad(x, (0, max(0, w + 2 - x.size)), mode='edge')
    r = np.diff(np.log(np.maximum(x, 1e-12)))
    r = (r - r.mean()) / (r.std() + 1e-8)
    spec = np.fft.rfft(r)
    mag = np.abs(spec)
    h = 64; k = min(len(mag), 64)
    tex = np.zeros((h, 64))
    base = (mag[:k] / (mag[:k].max() + 1e-8))
    for i in range(h):
        tex[i, :k] = np.roll(base, i // 2)
    return tex

def _fig_to_frame(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)

def save_overlay_gif(close_a: np.ndarray, close_b: np.ndarray, out_dir: pathlib.Path,
                     name: str, frames: int = 60, step: int = 5, fps: int = 12):
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = []
    T = max(10, int(frames))
    for t in range(T):
        ta = close_a[max(0, len(close_a) - (t+1)*step - 128):len(close_a) - t*step]
        tb = close_b[max(0, len(close_b) - (t+1)*step - 128):len(close_b) - t*step]
        tex_a = _latent(ta); tex_b = _latent(tb)
        mix = 0.5*tex_a + 0.5*tex_b
        fig = plt.figure(figsize=(2.4,2.4))
        plt.axis('off'); plt.imshow(mix, aspect='auto')
        imgs.append(_fig_to_frame(fig))
    gif_path = out_dir / f"{name}.gif"
    try:
        mp4_path = out_dir / f"{name}.mp4"
        imageio.mimsave(gif_path, imgs, fps=fps)
        imageio.mimsave(mp4_path, imgs, fps=fps)
    except Exception:
        imageio.mimsave(gif_path, imgs, fps=fps)
