"""Encoders that turn images into spike trains, plus the count-preserving
transports (upsample / downsample) used by nb013's eval-dt sweep.

Everything here is pure-data — no model state, no CLI plumbing. Lifted out
of cli.py.
"""

from __future__ import annotations

import models as M
import torch

EVAL_SEED = 20260415

FROZEN_MODES = ("upsample", "downsample", "resample")


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


def encode_smnist(images, dt, max_rate_hz, t_ms_per_row=10.0, generator=None):
    """Encode MNIST images as sequential row-by-row Poisson spikes.

    Each row (28 pixels) is presented for t_ms_per_row ms.
    Returns (T_steps, B, 28) float32 tensor of 0/1 spikes.

    images: (B, 784) pixel intensities in [0, 1]
    """
    B = images.shape[0]
    n_rows = 28
    n_cols = 28
    steps_per_row = int(t_ms_per_row / dt)
    T_steps = n_rows * steps_per_row
    device = images.device

    pixels = images.reshape(B, n_rows, n_cols)  # (B, 28, 28)
    p_spike = max_rate_hz * dt / 1000.0

    # Fully vectorized: for each row, broadcast probabilities across its timesteps.
    # Build (n_rows, steps_per_row, B, n_cols) probability tensor.
    # pixels[:, row, :] is held constant for steps_per_row steps.
    probs = pixels.permute(1, 0, 2).unsqueeze(1)  # (n_rows, 1, B, n_cols)
    probs = probs.expand(n_rows, steps_per_row, B, n_cols).contiguous()
    probs = probs.reshape(T_steps, B, n_cols) * p_spike
    if generator is not None:
        rand = torch.rand(
            T_steps, B, n_cols, device=generator.device, generator=generator
        ).to(device)
    else:
        rand = torch.rand(T_steps, B, n_cols, device=device)
    return (rand < probs).float()


def encode_batch(X_b, dt, use_smnist, generator=None):
    """Encode a pre-moved pixel batch as spikes using the canonical scheme.

    Shared by train, infer, and calibration loops so the three paths can't
    drift. Routes smnist through the row-by-row sequential encoder, 3-d
    already-spiked tensors (e.g. SHD) through a transpose passthrough, and
    everything else through vanilla Poisson rate coding. Output is always
    returned on X_b.device. Pass `generator` (typically a CPU torch.Generator
    with a fixed seed) for deterministic eval — same weights + same split +
    same generator seed → identical spike trains → identical accuracy.
    """
    if X_b.ndim == 3:
        # (B, T, N_in) pre-spiked → (T, B, N_in); ignore dt/generator.
        return X_b.permute(1, 0, 2).contiguous()
    if use_smnist:
        return encode_smnist(X_b, dt, M.max_rate_hz, generator=generator).to(X_b.device)
    return encode_images_poisson(X_b, M.T_steps, dt, M.max_rate_hz, generator=generator)


def downsample_spikes_count(spikes_ref, dt_ref, dt_target):
    """Sum-pool fine→coarse: integer count of spikes per coarse step
    (Parthasarathy et al. §2.3 integer-bin downsample). Per-ms rate is
    preserved exactly; total spike count is preserved up to leftover steps
    trimmed at the end when T_ref is not divisible by k. Output is not
    binary."""
    if dt_target < dt_ref - 1e-9:
        raise ValueError(
            f"downsample requires dt_target >= dt_ref; "
            f"got dt_target={dt_target}, dt_ref={dt_ref}"
        )
    k = round(dt_target / dt_ref)
    if k == 1:
        return spikes_ref
    T_target = spikes_ref.shape[0] // k
    trimmed = spikes_ref[: T_target * k]
    blocks = trimmed.reshape(T_target, k, *spikes_ref.shape[1:])
    return blocks.sum(dim=1)


def upsample_spikes_zeropad(spikes_ref, dt_ref, dt_target):
    """Zero-pad coarse→fine: expand T axis by k = dt_ref / dt_target and
    place each reference spike at the *first* fine sub-step of its block
    (Parthasarathy et al. §2.1 / Fig 1B). Total spike count is preserved
    exactly; per-ms rate is preserved; per-step rate drops by 1/k."""
    if dt_target > dt_ref + 1e-9:
        raise ValueError(
            f"upsample requires dt_target <= dt_ref; "
            f"got dt_target={dt_target}, dt_ref={dt_ref}"
        )
    k = round(dt_ref / dt_target)
    if k == 1:
        return spikes_ref
    T_ref = spikes_ref.shape[0]
    T_target = T_ref * k
    out = torch.zeros(
        (T_target, *spikes_ref.shape[1:]),
        dtype=spikes_ref.dtype,
        device=spikes_ref.device,
    )
    out[::k] = spikes_ref
    return out


def transport_spikes_bin(spikes_ref, dt_ref, dt_target):
    """Paper count-preserving transport: identity at dt_target == dt_ref,
    zero-pad when dt_target < dt_ref (§2.1 Fig 1B), sum-pool when
    dt_target > dt_ref (§2.3). Requires an integer ratio in either
    direction."""
    if abs(dt_target - dt_ref) < 1e-9:
        return spikes_ref
    if dt_target < dt_ref:
        ratio = dt_ref / dt_target
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(
                f"zero-pad requires integer dt_ref/dt_target; "
                f"got {dt_ref}/{dt_target}={ratio:.4f}"
            )
        return upsample_spikes_zeropad(spikes_ref, dt_ref, dt_target)
    ratio = dt_target / dt_ref
    if abs(ratio - round(ratio)) > 1e-6:
        raise ValueError(
            f"downsample requires integer dt_target/dt_ref; "
            f"got {dt_target}/{dt_ref}={ratio:.4f}"
        )
    return downsample_spikes_count(spikes_ref, dt_ref, dt_target)


class FrozenEncoder:
    """Deterministic encoder that controls how input spike patterns are
    transported across dt values during a sweep.

    dt_ref is the *training* dt — the anchor the paper describes when a
    network trained at dt_ref is evaluated at a sweep of other dts
    (Parthasarathy, Burghi & O'Leary 2024, Fig 1B / §2.1, §2.3).

    Modes (one transport per paper-named case):
      * upsample    — generate Poisson at dt_ref, then zero-pad to finer
                      eval-dt (§2.1 / Fig 1B). Each reference spike lands
                      at the first fine sub-step of its block. Requires
                      eval-dt <= dt_ref. Binary.
      * downsample  — generate Poisson at dt_ref, then sum-pool to coarser
                      eval-dt (§2.3). Requires eval-dt >= dt_ref. Output
                      is non-binary (integer counts per coarse bin).
      * resample    — draw a fresh Poisson stream at the *target* dt with
                      the same per-ms rate (§2.1 alternative). Binary;
                      sampling noise re-introduced. Works in both
                      directions.

    At eval-dt == dt_ref all three modes produce the same Poisson stream
    (identity transport). Call reset() before each new sweep dt so batch
    indices line up across iterations.
    """

    def __init__(self, dt_ref, t_ms, base_seed=42, mode="upsample"):
        if mode not in FROZEN_MODES:
            raise ValueError(
                f"unknown frozen-encoder mode {mode!r}; expected one of {FROZEN_MODES}"
            )
        self.dt_ref = dt_ref
        self.t_ms = t_ms
        self.base_seed = base_seed
        self.mode = mode
        self.batch_idx = 0

    def reset(self):
        self.batch_idx = 0

    def __call__(self, X_b, dt, use_smnist, generator=None):
        # generator arg ignored — FrozenEncoder seeds per-batch deterministically
        if X_b.ndim == 3:
            # Already-spiked inputs (e.g. SHD) — passthrough; dt-transport on
            # event-based data would need a rebinning path we haven't built yet.
            return X_b.permute(1, 0, 2).contiguous()
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.batch_idx)
        self.batch_idx += 1

        if self.mode == "resample":
            if use_smnist:
                return encode_smnist(X_b, dt, M.max_rate_hz, generator=g)
            T_target = int(self.t_ms / dt)
            return encode_images_poisson(X_b, T_target, dt, M.max_rate_hz, generator=g)

        if use_smnist:
            spk_ref = encode_smnist(X_b, self.dt_ref, M.max_rate_hz, generator=g)
        else:
            T_ref = int(self.t_ms / self.dt_ref)
            spk_ref = encode_images_poisson(
                X_b, T_ref, self.dt_ref, M.max_rate_hz, generator=g
            )

        if abs(dt - self.dt_ref) < 1e-9:
            return spk_ref
        if self.mode == "upsample":
            return upsample_spikes_zeropad(spk_ref, self.dt_ref, dt)
        if self.mode == "downsample":
            return downsample_spikes_count(spk_ref, self.dt_ref, dt)
        raise AssertionError(f"unhandled frozen-encoder mode {self.mode!r}")
