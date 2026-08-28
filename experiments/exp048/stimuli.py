"""Deterministic stimulus construction, with no execution or storage."""

import numpy as np
import torch

from .recipe import DT, N_CLASSES, N_IN


def encode_stream(
    digit_pixels: np.ndarray,
    tau_ms: float,
    input_rate_hz: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Concatenate Poisson-encoded digits into one (T_stream, 1, N_IN)
    spike tensor. Each digit is encoded for tau_ms at the given
    Poisson rate; rate scaling for τ-compensation is handled by the
    caller.
    """
    tau_steps = int(round(tau_ms / DT))
    p_step = input_rate_hz * DT / 1000.0
    n_digits = digit_pixels.shape[0]
    streams: list[torch.Tensor] = []
    for d in range(n_digits):
        pixels = torch.from_numpy(digit_pixels[d : d + 1]).clamp(0, 1)
        # Same scheme as encode_images_poisson: per-step Bernoulli at p_step
        # weighted by pixel intensity in [0,1].
        rand = torch.rand(tau_steps, 1, N_IN, generator=generator)
        spikes = (rand < (p_step * pixels.unsqueeze(0))).float()
        streams.append(spikes)
    return torch.cat(streams, dim=0)


def encode_varying_stream(
    digit_pixels: np.ndarray,
    segments: list[tuple[float, float]],
    generator: torch.Generator,
) -> torch.Tensor:
    """Concatenate digits with per-segment (τ_ms, rate_hz)."""
    streams: list[torch.Tensor] = []
    for d, (tau_ms, rate_hz) in enumerate(segments):
        tau_steps = int(round(tau_ms / DT))
        p_step = rate_hz * DT / 1000.0
        pixels = torch.from_numpy(digit_pixels[d : d + 1]).clamp(0, 1)
        rand = torch.rand(tau_steps, 1, N_IN, generator=generator)
        spikes = (rand < (p_step * pixels.unsqueeze(0))).float()
        streams.append(spikes)
    return torch.cat(streams, dim=0)


def pick_diverse_digits(X_te, y_te, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Pick n digits of n different classes, deterministic by seed."""
    rng = np.random.default_rng(seed)
    classes = list(range(N_CLASSES))
    rng.shuffle(classes)
    classes = classes[:n]
    pixels: list[np.ndarray] = []
    labels: list[int] = []
    for c in classes:
        idx = np.where(y_te == c)[0]
        if idx.size == 0:
            continue
        i = int(rng.choice(idx))
        pixels.append(X_te[i])
        labels.append(c)
    return np.stack(pixels, axis=0), np.array(labels, dtype=np.int64)
