"""Count-preserving spike-time transforms for the two retained jitter arms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .recipe import (
    F_GAMMA_REFERENCE_HZ,
    JITTER_BOUNDARY_POLICY,
    JITTER_COLLISION_POLICY,
)

if TYPE_CHECKING:
    import torch


def _build_override(
    s_i_base: "torch.Tensor",
    condition: str,
    generator,
    dt_ms: float = 0.1,
    *,
    return_diagnostics: bool = False,
) -> "torch.Tensor | tuple[torch.Tensor, dict]":
    """Construct the I-spike override tensor for one batch.

    s_i_base: (T, B, N_I) baseline recorded I-spikes.
    Returns an override tensor with shape (T, B, N_I).

    Conditions:
      - jitter_sigma_{X}: fixed-window group jitter with σ = X ms.
        Uses F_GAMMA_REFERENCE_HZ to define the window duration.
      - cell_jitter_sigma_{X}: per-spike Gaussian jitter with σ = X ms
        (events receive independent offsets).
    """
    if s_i_base.ndim == 2:  # (T, N_I) when batch size is 1
        s_i_base = s_i_base.unsqueeze(1)
    if condition.startswith("jitter_sigma_"):
        sigma_ms = float(condition.split("_")[-1])
        out = _jitter_i_stream(s_i_base, sigma_ms, dt_ms, generator)
    elif condition.startswith("cell_jitter_sigma_"):
        sigma_ms = float(condition.split("_")[-1])
        out = _cell_jitter_i_stream(s_i_base, sigma_ms, dt_ms, generator)
    else:
        raise ValueError(f"unknown condition {condition!r}")
    if return_diagnostics:
        return out
    return out[0]


def _resolve_collisions(candidate_t, b_idx, n_idx, T: int, N_I: int):
    """Move duplicate destinations to nearest free circular timesteps.

    Candidate order is stable. Later events at an occupied destination try
    +1, -1, +2, -2, ... around their original candidate. Because every input
    cell stream is binary, it contains at most T events and a complete unique
    assignment always exists.
    """
    import torch

    current = candidate_t.clone()
    original = candidate_t.clone()
    attempts = torch.zeros_like(candidate_t)
    moved = torch.zeros(candidate_t.shape, dtype=torch.bool, device=candidate_t.device)
    while current.numel():
        keys = ((b_idx * N_I + n_idx) * T) + current
        order = torch.argsort(keys, stable=True)
        sorted_keys = keys[order]
        duplicate_sorted = sorted_keys[1:] == sorted_keys[:-1]
        if not bool(duplicate_sorted.any()):
            break
        duplicate = torch.zeros_like(moved)
        duplicate[order[1:][duplicate_sorted]] = True
        attempts[duplicate] += 1
        if int(attempts.max()) > 2 * T:
            raise RuntimeError("could not resolve inhibitory spike collisions")
        attempt = attempts[duplicate]
        distance = (attempt + 1) // 2
        displacement = torch.where(attempt % 2 == 1, distance, -distance)
        current[duplicate] = (original[duplicate] + displacement).remainder(T)
        moved |= duplicate
    max_steps = int(((attempts + 1) // 2).max()) if attempts.numel() else 0
    return current, moved, max_steps


def _finish_override(s_i_base, spike_positions, candidate_t, wrapped):
    import torch

    T, _B, N_I = s_i_base.shape
    b_idx = spike_positions[:, 1]
    n_idx = spike_positions[:, 2]
    new_t, moved, max_steps = _resolve_collisions(candidate_t, b_idx, n_idx, T, N_I)
    out = torch.zeros_like(s_i_base)
    out.index_put_(
        (new_t, b_idx, n_idx),
        torch.ones(
            spike_positions.shape[0],
            dtype=s_i_base.dtype,
            device=s_i_base.device,
        ),
        accumulate=False,
    )
    if not torch.equal(out.sum(dim=0), s_i_base.sum(dim=0)):
        raise RuntimeError("jitter changed a trial/cell inhibitory spike count")
    diagnostics = {
        "boundary_policy": JITTER_BOUNDARY_POLICY,
        "collision_policy": JITTER_COLLISION_POLICY,
        "input_spikes": int(spike_positions.shape[0]),
        "output_spikes": int(out.count_nonzero()),
        "boundary_wrapped_spikes": int(wrapped.count_nonzero()),
        "collision_resolved_spikes": int(moved.count_nonzero()),
        "max_collision_resolution_steps": max_steps,
        "per_trial_cell_count_invariant": True,
    }
    return out, diagnostics


def _unchanged(s_i_base):
    spikes = int(s_i_base.count_nonzero())
    return s_i_base.clone(), {
        "boundary_policy": JITTER_BOUNDARY_POLICY,
        "collision_policy": JITTER_COLLISION_POLICY,
        "input_spikes": spikes,
        "output_spikes": spikes,
        "boundary_wrapped_spikes": 0,
        "collision_resolved_spikes": 0,
        "max_collision_resolution_steps": 0,
        "per_trial_cell_count_invariant": True,
    }


def _jitter_i_stream(
    s_i_base: "torch.Tensor",
    sigma_ms: float,
    dt_ms: float,
    generator,
) -> tuple["torch.Tensor", dict]:
    """Fixed-window group jitter on the I-spike stream.

    Bins the original timeline into fixed windows of duration
    1 / F_GAMMA_REFERENCE_HZ, draws one Gaussian offset Δ ~ 𝒩(0, σ²)
    per (trial, window), and shifts every inhibitory event originating in
    that window by Δ. The windows are a fixed clock, not detected cycles.

    σ is in milliseconds; the conversion to timesteps uses dt_ms. Proposed
    times wrap around the presentation. Same-cell/time collisions move to the
    nearest free circular timestep, preserving every trial/cell spike count.
    """
    import torch

    T, B, _N_I = s_i_base.shape
    if sigma_ms <= 0.0:
        return _unchanged(s_i_base)

    cycle_period_ms = 1000.0 / F_GAMMA_REFERENCE_HZ
    cycle_period_steps = max(1, int(round(cycle_period_ms / dt_ms)))
    n_cycles = (T + cycle_period_steps - 1) // cycle_period_steps
    sigma_steps = sigma_ms / dt_ms

    # Per-(trial, fixed window) Gaussian offset, in timestep units, rounded.
    offsets = torch.randn(B, n_cycles, generator=generator) * sigma_steps
    offsets_int = offsets.round().long()

    spike_positions = s_i_base.nonzero(as_tuple=False)  # (n_spikes, 3): (t, b, n)
    if spike_positions.numel() == 0:
        return _unchanged(s_i_base)
    t_orig = spike_positions[:, 0]
    b_idx = spike_positions[:, 1]
    cycle_idx = (t_orig // cycle_period_steps).clamp(0, n_cycles - 1)
    # Look up the per-(b, window) offset for each spike, then wrap.
    jitter = offsets_int[b_idx, cycle_idx]
    proposed = t_orig + jitter
    return _finish_override(
        s_i_base,
        spike_positions,
        proposed.remainder(T),
        (proposed < 0) | (proposed >= T),
    )


def _cell_jitter_i_stream(
    s_i_base: "torch.Tensor",
    sigma_ms: float,
    dt_ms: float,
    generator,
) -> tuple["torch.Tensor", dict]:
    """Independent-spike Gaussian jitter on the I-spike stream.

    Each spike gets its own independent Gaussian offset Δ ~ 𝒩(0, σ²).
    Events that shared a baseline timestep can land at different replay
    times because each event receives its own zero-mean offset.

    Proposed times wrap around the presentation. Same-cell/time collisions
    move to the nearest free circular timestep, preserving every trial/cell
    spike count.
    """
    import torch

    T, _B, _N_I = s_i_base.shape
    if sigma_ms <= 0.0:
        return _unchanged(s_i_base)

    sigma_steps = sigma_ms / dt_ms
    spike_positions = s_i_base.nonzero(as_tuple=False)  # (n_spikes, 3): (t, b, n)
    if spike_positions.numel() == 0:
        return _unchanged(s_i_base)
    t_orig = spike_positions[:, 0]
    # Independent Gaussian offset per spike, rounded to timestep grid.
    n_spikes = spike_positions.shape[0]
    offsets = (torch.randn(n_spikes, generator=generator) * sigma_steps).round().long()
    proposed = t_orig + offsets
    return _finish_override(
        s_i_base,
        spike_positions,
        proposed.remainder(T),
        (proposed < 0) | (proposed >= T),
    )
