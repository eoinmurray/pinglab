"""Differentiable PING health loss for regularizing spiking networks.

Computes a spectral comb loss from spike tensors using only differentiable
PyTorch ops (FFT, softmax, element-wise arithmetic).
"""
from __future__ import annotations

import torch


def ping_health_loss(
    spikes: torch.Tensor,
    dt_ms: float,
    bin_ms: float = 5.0,
    f_target: float = 30.0,
    f_search_lo: float = 5.0,
    f_search_hi: float = 80.0,
    sigma_hz: float = 1.75,
    n_harmonics: int = 5,
    alpha: float = 1.0,
    beta: float = 0.01,
    gamma: float = 0.1,
) -> torch.Tensor:
    """Differentiable PING health loss.

    Args:
        spikes: Binary spike tensor ``[B, T, N]``.
        dt_ms: Simulation timestep in milliseconds.
        bin_ms: Width of population-rate bins in ms.
        f_target: Target gamma frequency in Hz.
        f_search_lo: Lower bound of frequency search band (Hz).
        f_search_hi: Upper bound of frequency search band (Hz).
        sigma_hz: Width of each Gaussian comb tooth (Hz).
        n_harmonics: Number of harmonic teeth in the comb mask.
        alpha: Weight for spectral concentration term.
        beta: Weight for frequency deviation term.
        gamma: Weight for noise-to-signal ratio term.

    Returns:
        Scalar loss tensor with gradient.
    """
    B, T, N = spikes.shape
    bin_size = max(1, int(bin_ms / dt_ms))
    n_bins = T // bin_size

    # 1. Bin spikes into population rate: average over batch for stability
    # Truncate T to fit evenly into bins
    truncated = spikes[:, :n_bins * bin_size, :]  # [B, n_bins*bin_size, N]
    reshaped = truncated.reshape(B, n_bins, bin_size, N)  # [B, n_bins, bin_size, N]
    # Sum over bin_size and neurons, average over batch → [n_bins]
    rate = reshaped.sum(dim=(0, 2, 3)) / B  # [n_bins]

    # 2. Mean-subtract and compute PSD via FFT
    rate = (rate - rate.mean()).contiguous()
    spectrum = torch.fft.rfft(rate, n=rate.shape[0])
    psd = (spectrum.real ** 2 + spectrum.imag ** 2) / n_bins  # [n_freq]

    # Frequency axis
    dt_s = bin_ms / 1000.0
    freqs = torch.fft.rfftfreq(n_bins, d=dt_s).to(psd.device)  # [n_freq]

    # 3. Find f0 via soft-argmax over search band
    band_mask = (freqs >= f_search_lo) & (freqs <= f_search_hi)
    band_indices = torch.where(band_mask)[0]

    if len(band_indices) == 0:
        # No valid frequencies — return zero loss
        return torch.tensor(0.0, device=spikes.device, requires_grad=True)

    psd_band = psd[band_indices]
    freqs_band = freqs[band_indices]

    # Temperature-scaled softmax for soft argmax
    temperature = 0.1
    weights = torch.softmax(psd_band / (psd_band.max().detach() * temperature + 1e-8), dim=0)
    f0 = (weights * freqs_band).sum()  # differentiable f0 estimate

    # 4. Build Gaussian comb mask from f0 and harmonics
    comb = torch.zeros_like(freqs)
    for k in range(1, n_harmonics + 1):
        comb = comb + torch.exp(-((freqs - k * f0) ** 2) / (2 * sigma_hz ** 2))
    comb = comb.clamp(0, 1)

    # 5. Signal and noise power
    p_signal = (comb * psd).sum()
    p_noise = ((1 - comb) * psd).sum()
    p_total = psd.sum()

    # 6. Composite loss
    eps = 1e-8
    l_concentration = -torch.log(p_signal / (p_total + eps) + eps)
    l_frequency = (f0 - f_target) ** 2
    l_noise = p_noise / (p_signal + eps)

    loss = alpha * l_concentration + beta * l_frequency + gamma * l_noise
    return loss
