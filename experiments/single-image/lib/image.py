from __future__ import annotations
import numpy as np

def synth_blobs(h: int, w: int, n: int = 2, seed: int = 0) -> np.ndarray:
    """
    Simple blob generator: sum of n Gaussian blobs, normalized to [0,1].
    Returns (h,w) float32.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]

    img = np.zeros((h, w), dtype=np.float32)
    for _ in range(n):
        cy = rng.uniform(0, h - 1)
        cx = rng.uniform(0, w - 1)
        sy = rng.uniform(h * 0.10, h * 0.25)
        sx = rng.uniform(w * 0.10, w * 0.25)

        blob = np.exp(-(((yy - cy) ** 2) / (2 * sy**2) + ((xx - cx) ** 2) / (2 * sx**2)))
        img += blob.astype(np.float32)

    # Normalize
    img -= img.min()
    if img.max() > 1e-8:
        img /= img.max()
    return img

def synth_bars(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.float32)
    if rng.random() < 0.5:
        # vertical bar
        x0 = rng.integers(0, w)
        width = max(1, w // 8)
        img[:, max(0, x0 - width): min(w, x0 + width)] = 1.0
    else:
        # horizontal bar
        y0 = rng.integers(0, h)
        width = max(1, h // 8)
        img[max(0, y0 - width): min(h, y0 + width), :] = 1.0
    return img

def synth_checker(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = int(rng.integers(2, 5))
    yy, xx = np.mgrid[0:h, 0:w]
    img = ((yy // k + xx // k) % 2).astype(np.float32)
    return img
