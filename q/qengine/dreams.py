
from PIL import Image, ImageDraw
import numpy as np, math

def dream_from_dna(drift_series, width=640, height=320, seed=0):
    rng = np.random.RandomState(seed)
    data = (np.nan_to_num(drift_series).astype(float))
    if len(data)==0:
        data = np.zeros(256)
    vel = np.r_[0.0, np.diff(data)]
    energy = np.abs(vel)
    d_norm = (data - data.min()) / (data.max() - data.min() + 1e-9)
    v_norm = (vel - vel.min()) / (vel.max() - vel.min() + 1e-9)
    e_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-9)
    img = Image.new("RGB", (width, height), (0,0,0))
    draw = ImageDraw.Draw(img)
    w = width / max(1, len(d_norm))
    for i, v in enumerate(d_norm):
        h = int(v * height)
        x0 = int(i*w); x1 = int((i+1)*w)
        # R=drift level, G=drift velocity, B=energy/churn
        color = (
            int(255 * v),
            int(255 * v_norm[i]),
            int(255 * e_norm[i]),
        )
        draw.rectangle([x0, height-h, x1, height], fill=color)

    # add a deterministic dreamline (phase-warped sine) to encode cyclical state
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    pts = []
    for i in range(len(d_norm)):
        x = int(i * w)
        y = int((0.15 + 0.70 * (1.0 - d_norm[i])) * height + 12.0 * math.sin(0.12 * i + phase))
        pts.append((x, max(0, min(height - 1, y))))
    if len(pts) > 2:
        draw.line(pts, fill=(255, 255, 255), width=2)
    return img
