"""Retained spike-time transforms, including rounding, boundary clamping and collisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .recipe import F_GAMMA_REFERENCE_HZ

if TYPE_CHECKING:
    import torch


def _build_override(
    s_i_base: "torch.Tensor",
    condition: str,
    generator,
    dt_ms: float = 0.1,
) -> "torch.Tensor":
    """Construct the I-spike override tensor for one batch.

    s_i_base: (T, B, N_I) baseline recorded I-spikes.
    Returns (T, B, N_I) override tensor preserving per-(trial, cell)
    spike counts in expectation.

    Conditions:
      - phase_shuffled_i: permute time axis per trial (all I cells share permutation)
      - poisson_matched_i: per-(trial, cell) Bernoulli at matched mean rate
      - jitter_sigma_{X}: cycle-coherent Gaussian jitter with σ = X ms.
        Uses F_GAMMA_REFERENCE_HZ as the cycle period.
      - cell_jitter_sigma_{X}: per-spike Gaussian jitter with σ = X ms
        (destroys within-burst synchrony; preserves burst placement on average).
    """
    import torch

    if s_i_base.ndim == 2:  # (T, N_I) when batch size is 1
        s_i_base = s_i_base.unsqueeze(1)
    T, B, N_I = s_i_base.shape
    if condition == "phase_shuffled_i":
        out = torch.empty_like(s_i_base)
        for b in range(B):
            perm = torch.randperm(T, generator=generator)
            out[:, b, :] = s_i_base[perm, b, :]
    elif condition == "poisson_matched_i":
        counts = s_i_base.sum(dim=0)
        p = (counts / float(T)).clamp(0.0, 1.0).unsqueeze(0).expand(T, B, N_I)
        out = (torch.rand(T, B, N_I, generator=generator) < p).to(s_i_base.dtype)
    elif condition.startswith("jitter_sigma_"):
        sigma_ms = float(condition.split("_")[-1])
        out = _jitter_i_stream(s_i_base, sigma_ms, dt_ms, generator)
    elif condition.startswith("cell_jitter_sigma_"):
        sigma_ms = float(condition.split("_")[-1])
        out = _cell_jitter_i_stream(s_i_base, sigma_ms, dt_ms, generator)
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return out


def _jitter_i_stream(
    s_i_base: "torch.Tensor",
    sigma_ms: float,
    dt_ms: float,
    generator,
) -> "torch.Tensor":
    """Cycle-coherent jitter on the I-spike stream.

    Bins time into blocks of one gamma cycle (1 / F_GAMMA_REFERENCE_HZ
    ≈ 28 ms at the trained operating point), draws one Gaussian offset
    Δ ~ 𝒩(0, σ²) per
    (trial, cycle), and shifts every I-spike in that block by Δ.
    Within-burst cross-cell synchrony is preserved exactly; what's
    perturbed is the *placement* of each burst relative to where the
    baseline cycle put it.

    The diagnostic prediction: rate release should be small when
    σ ≪ 1/f_γ (bursts barely move from their phase-locked slots) and
    large when σ ≳ 1/f_γ (bursts can land anywhere within the cycle,
    losing phase relation to E).

    σ in milliseconds; the conversion to timesteps uses dt_ms.
    """
    import torch

    T, B, N_I = s_i_base.shape
    if sigma_ms <= 0.0:
        return s_i_base.clone()

    cycle_period_ms = 1000.0 / F_GAMMA_REFERENCE_HZ
    cycle_period_steps = max(1, int(round(cycle_period_ms / dt_ms)))
    n_cycles = (T + cycle_period_steps - 1) // cycle_period_steps
    sigma_steps = sigma_ms / dt_ms

    # Per-(trial, cycle) Gaussian offset, in timestep units, rounded.
    offsets = torch.randn(B, n_cycles, generator=generator) * sigma_steps
    offsets_int = offsets.round().long()

    spike_positions = s_i_base.nonzero(as_tuple=False)  # (n_spikes, 3): (t, b, n)
    if spike_positions.numel() == 0:
        return s_i_base.clone()
    t_orig = spike_positions[:, 0]
    b_idx = spike_positions[:, 1]
    n_idx = spike_positions[:, 2]
    cycle_idx = (t_orig // cycle_period_steps).clamp(0, n_cycles - 1)
    # Look up the per-(b, cycle) offset for each spike, add, clamp.
    jitter = offsets_int[b_idx, cycle_idx]
    new_t = (t_orig + jitter).clamp(0, T - 1)
    out = torch.zeros_like(s_i_base)
    out.index_put_(
        (new_t, b_idx, n_idx),
        torch.ones(spike_positions.shape[0], dtype=s_i_base.dtype),
        accumulate=False,
    )
    return out


def _cell_jitter_i_stream(
    s_i_base: "torch.Tensor",
    sigma_ms: float,
    dt_ms: float,
    generator,
) -> "torch.Tensor":
    """Per-spike (per-I-cell) Gaussian jitter on the I-spike stream.

    Each spike gets its own independent Gaussian offset Δ ~ 𝒩(0, σ²).
    Within-burst cross-cell synchrony is destroyed — different I-cells
    that fired at the same timestep in baseline land at different times
    in the override. Burst placement is preserved on average (each
    spike's offset has zero mean), but the burst itself smears across
    a window of width ≈ σ.

    Complements `_jitter_i_stream` (cycle-coherent): the cycle-coherent
    sweep tests whether the *placement* of each burst relative to the
    gamma cycle matters; per-cell jitter tests whether the *sharpness*
    of each burst matters.

    Spike times are clamped to the valid range. Collisions at the same
    cell and timestep merge, so realised rates must still be measured.
    """
    import torch

    T, B, N_I = s_i_base.shape
    if sigma_ms <= 0.0:
        return s_i_base.clone()

    sigma_steps = sigma_ms / dt_ms
    spike_positions = s_i_base.nonzero(as_tuple=False)  # (n_spikes, 3): (t, b, n)
    if spike_positions.numel() == 0:
        return s_i_base.clone()
    t_orig = spike_positions[:, 0]
    b_idx = spike_positions[:, 1]
    n_idx = spike_positions[:, 2]
    # Independent Gaussian offset per spike, rounded to timestep grid.
    n_spikes = spike_positions.shape[0]
    offsets = (torch.randn(n_spikes, generator=generator) * sigma_steps).round().long()
    new_t = (t_orig + offsets).clamp(0, T - 1)
    out = torch.zeros_like(s_i_base)
    out.index_put_(
        (new_t, b_idx, n_idx),
        torch.ones(n_spikes, dtype=s_i_base.dtype),
        accumulate=False,
    )
    return out
