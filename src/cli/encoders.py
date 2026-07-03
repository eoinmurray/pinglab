"""Encoders that turn images into spike trains.

Everything here is pure-data — no model state, no CLI plumbing. Lifted out
of cli.py.
"""

from __future__ import annotations

import models as M
import torch

EVAL_SEED = 20260415


def encode_images_poisson(images, T_steps, dt, max_rate_hz, generator=None):
    """Encode (B, N_in) pixel intensities as Poisson spike trains.

    Returns (T_steps, B, N_in) float spikes. Single canonical encoder used by
    train, infer, and image paths so identical pixels with the same dt
    and max_rate produce the same spike train regardless of mode.
    """
    pixels = images.clamp(0, 1)
    p = max_rate_hz * dt / 1000.0
    B, n_in = pixels.shape
    if generator is not None:
        # Generator dictates device (usually CPU); generate there then move.
        rand = torch.rand(
            T_steps, B, n_in, device=generator.device, generator=generator
        ).to(pixels.device)
    else:
        rand = torch.rand(T_steps, B, n_in, device=pixels.device)
    return (rand < pixels.unsqueeze(0) * p).float()


def encode_batch(X_b, dt, generator=None):
    """Encode a pre-moved pixel batch as spikes using the canonical scheme.

    Shared by train, infer, and calibration loops so the three paths can't
    drift. Routes 3-d already-spiked tensors through a transpose passthrough
    and everything else through vanilla Poisson rate coding. Output is always
    returned on X_b.device. Pass `generator` (typically a CPU torch.Generator
    with a fixed seed) for deterministic eval — same weights + same split +
    same generator seed → identical spike trains → identical accuracy.
    """
    if X_b.ndim == 3:
        # (B, T, N_in) pre-spiked → (T, B, N_in); ignore dt/generator.
        return X_b.permute(1, 0, 2).contiguous()
    return encode_images_poisson(X_b, M.T_steps, dt, M.max_rate_hz, generator=generator)
