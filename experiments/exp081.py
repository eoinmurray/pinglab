"""EXP081: standalone analytical and empirical filtered-response theory.

The experiment independently specifies its encoder, synapse, membrane, and
finite-window feature. It generates new single-pixel simulations, derives the
stationary linear filter, and compares analytical moments with those simulations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import theme  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import (
    finalize_prepared_run,  # noqa: E402
    preserve_active_view,  # noqa: E402
)
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp081"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SMOKE = os.environ.get("PINGLAB_SMOKE") == "1"

PRESENTATION_MS = 200.0
DT_MS = 0.1
N_TIMESTEPS = int(round(PRESENTATION_MS / DT_MS))
PROBES_US = (0.6, 1.2, 2.4)
INPUT_RATES_HZ = np.linspace(0.0, 25.0, 101)
MOMENT_DRAWS = 32 if SMOKE else 512
DISTRIBUTION_RATES_HZ = (0.25, 3.0, 25.0)
FREQUENCY_RESPONSE_RATES_HZ = (0.25, 3.0, 25.0)
NOMINAL_PROBE_US = 1.2
FREQUENCY_PLOT_BOUNDS_HZ = (0.1, 200.0)
DISTRIBUTION_DRAWS = 128 if SMOKE else 4096
SEED = 81
FREQUENCY_BOUNDS_HZ = (1e-4, 1e6)
FREQUENCY_GRID_POINTS = 1025 if SMOKE else 16385
COARSE_GRID_POINTS = 513 if SMOKE else 8193

PARAMETERS = {
    "C_m_nF": 1.0,
    "g_L_uS": 0.05,
    "E_L_mV": -65.0,
    "E_e_mV": 0.0,
    "tau_ampa_ms": 2.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: int) -> int:
    digest = hashlib.sha256(
        ":".join(str(part) for part in (SEED, *parts)).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def torch_device() -> Any:
    import torch

    requested = os.environ.get("EXP081_DEVICE", "auto")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def simulate_features(
    input_rates_hz: np.ndarray,
    probes_uS: np.ndarray,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Generate fresh finite-window features for aligned drive/probe conditions."""
    import torch

    rates, probes = np.broadcast_arrays(
        np.asarray(input_rates_hz, dtype=np.float32),
        np.asarray(probes_uS, dtype=np.float32),
    )
    device = torch_device()
    probability = torch.as_tensor(rates.reshape(-1, 1) * DT_MS / 1000.0, device=device)
    probe = torch.as_tensor(probes.reshape(-1, 1), device=device)
    shape = (rates.size, draws)
    conductance = torch.zeros(shape, device=device)
    voltage = torch.full(shape, PARAMETERS["E_L_mV"], device=device)
    feature_sum = torch.zeros(shape, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    decay = math.exp(-DT_MS / PARAMETERS["tau_ampa_ms"])
    for _ in range(N_TIMESTEPS):
        events = torch.rand(shape, device=device, generator=generator) < probability
        conductance = conductance * decay + probe * events
        total_g = PARAMETERS["g_L_uS"] + conductance
        equilibrium = (
            PARAMETERS["g_L_uS"] * PARAMETERS["E_L_mV"]
            + conductance * PARAMETERS["E_e_mV"]
        ) / total_g
        voltage = equilibrium + (voltage - equilibrium) * torch.exp(
            -DT_MS * total_g / PARAMETERS["C_m_nF"]
        )
        feature_sum += voltage - PARAMETERS["E_L_mV"]
    output = (feature_sum / N_TIMESTEPS).cpu().numpy()
    return output.reshape(*rates.shape, draws)


def linear_operating_point(
    input_rate_hz: np.ndarray | float,
    probe_uS: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    rate = np.asarray(input_rate_hz, dtype=np.float64)
    probe = np.asarray(probe_uS, dtype=np.float64)
    mean_g = rate * probe * PARAMETERS["tau_ampa_ms"] / 1000.0
    mean_v = (
        PARAMETERS["g_L_uS"] * PARAMETERS["E_L_mV"] + mean_g * PARAMETERS["E_e_mV"]
    ) / (PARAMETERS["g_L_uS"] + mean_g)
    return mean_g, mean_v


def synapse_membrane_transfer(
    frequency_hz: np.ndarray,
    input_rate_hz: np.ndarray | float,
    probe_uS: np.ndarray | float,
) -> np.ndarray:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    omega = 2.0 * np.pi * frequency / 1000.0
    mean_g, mean_v = linear_operating_point(input_rate_hz, probe_uS)
    synapse = np.asarray(probe_uS) / (1j * omega + 1.0 / PARAMETERS["tau_ampa_ms"])
    membrane = (PARAMETERS["E_e_mV"] - mean_v) / (
        1j * omega * PARAMETERS["C_m_nF"] + PARAMETERS["g_L_uS"] + mean_g
    )
    return synapse * membrane


def complete_transfer(
    frequency_hz: np.ndarray,
    input_rate_hz: np.ndarray | float,
    probe_uS: np.ndarray | float,
) -> np.ndarray:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    omega = 2.0 * np.pi * frequency / 1000.0
    argument = omega * PRESENTATION_MS / 2.0
    averaging = np.exp(-1j * argument) * np.sinc(argument / np.pi)
    return averaging * synapse_membrane_transfer(frequency_hz, input_rate_hz, probe_uS)


def predicted_variance(
    input_rates_hz: np.ndarray,
    probe_uS: np.ndarray,
    *,
    grid_points: int = FREQUENCY_GRID_POINTS,
) -> np.ndarray:
    rates = np.asarray(input_rates_hz, dtype=np.float64).reshape(-1)
    probe = np.asarray(probe_uS, dtype=np.float64).reshape(-1)
    if rates.shape != probe.shape:
        raise ValueError("input_rates_hz and probe_uS must have matching shapes")
    result = np.zeros_like(rates)
    frequencies = np.geomspace(*FREQUENCY_BOUNDS_HZ, grid_points)
    for start in range(0, rates.size, 128):
        indices = np.arange(start, min(start + 128, rates.size))
        positive = indices[rates[indices] > 0]
        if positive.size == 0:
            continue
        transfer = complete_transfer(
            frequencies[None, :], rates[positive, None], probe[positive, None]
        )
        integrand = np.abs(transfer) ** 2 * rates[positive, None] / 1000.0
        integral = np.trapezoid(integrand, frequencies, axis=1)
        dc = (
            np.abs(
                complete_transfer(
                    np.asarray([[0.0]]),
                    rates[positive, None],
                    probe[positive, None],
                )[:, 0]
            )
            ** 2
        )
        low_tail = dc * rates[positive] / 1000.0 * FREQUENCY_BOUNDS_HZ[0]
        result[positive] = 2.0 / 1000.0 * (integral + low_tail)
    return result


def plot_moments(
    empirical_mean: np.ndarray,
    empirical_sd: np.ndarray,
) -> None:
    theme.apply()
    colors = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25), constrained_layout=True)
    for index, (probe, color) in enumerate(zip(PROBES_US, colors, strict=True)):
        axes[0].plot(
            INPUT_RATES_HZ,
            empirical_mean[index],
            color=color,
            label=f"{probe:g} μS",
        )
        axes[1].plot(INPUT_RATES_HZ, empirical_sd[index], color=color)
    axes[0].set(title="A  Mean feature", ylabel="Mean feature z (mV)")
    axes[1].set(title="B  Feature SD", ylabel="Feature SD (mV)")
    for axis in axes:
        axis.set(xlabel="Input rate (Hz)", xlim=(0, 25))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.14)
    axes[0].legend(frameon=False)
    fig.savefig(FIGURES / "empirical_moments.svg", metadata={"Date": None})
    plt.close(fig)


def plot_distributions(samples: np.ndarray) -> None:
    theme.apply()
    fig, axes = plt.subplots(
        1, 3, figsize=(6.5, 2.75), constrained_layout=True, sharex=True, sharey=True
    )
    upper = float(np.ceil(np.max(samples) / 5.0) * 5.0)
    bins = np.linspace(0.0, upper, 61)
    for axis, rate, values in zip(axes, DISTRIBUTION_RATES_HZ, samples, strict=True):
        axis.hist(
            values,
            bins=bins,
            weights=np.full(values.shape, 1.0 / values.size),
            color=theme.INK_BLACK,
            alpha=0.72,
        )
        axis.set_title(f"{rate:g} Hz")
        axis.set_xlabel("Feature z (mV)")
        axis.set_yscale("log")
        axis.set_ylim(0.5 / values.size, 1.0)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Probability per bin (log scale)")
    fig.savefig(FIGURES / "response_distributions.svg", metadata={"Date": None})
    plt.close(fig)


def plot_frequency_response() -> None:
    theme.apply()
    frequency = np.geomspace(*FREQUENCY_PLOT_BOUNDS_HZ, 1400)
    rates = FREQUENCY_RESPONSE_RATES_HZ
    colors = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    reference = abs(
        synapse_membrane_transfer(np.asarray([0.0]), rates[0], NOMINAL_PROBE_US)[0]
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(6.5, 3.25), constrained_layout=True, sharey=True
    )
    for rate, color in zip(rates, colors, strict=True):
        unaveraged = np.abs(
            synapse_membrane_transfer(frequency, rate, NOMINAL_PROBE_US)
        )
        averaged = np.abs(complete_transfer(frequency, rate, NOMINAL_PROBE_US))
        axes[0].semilogx(
            frequency,
            20 * np.log10(np.maximum(unaveraged / reference, 1e-8)),
            color=color,
            label=f"{rate:g} Hz drive",
        )
        axes[1].semilogx(
            frequency,
            20 * np.log10(np.maximum(averaged / reference, 1e-8)),
            color=color,
        )
    axes[0].set(
        title="A  Synapse + membrane",
        ylabel="Magnitude relative to low-drive DC (dB)",
    )
    axes[1].set(title="B  After 200 ms averaging")
    for axis in axes:
        axis.set(xlabel="Frequency (Hz)", ylim=(-90, 4))
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(FIGURES / "frequency_response.svg", metadata={"Date": None})
    plt.close(fig)


def plot_comparison(
    empirical_mean: np.ndarray,
    empirical_sd: np.ndarray,
    analytical_mean: np.ndarray,
    analytical_sd: np.ndarray,
) -> None:
    theme.apply()
    colors = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25), constrained_layout=True)
    for index, (probe, color) in enumerate(zip(PROBES_US, colors, strict=True)):
        axes[0].plot(
            INPUT_RATES_HZ,
            analytical_mean[index],
            color=color,
            label=f"{probe:g} μS",
        )
        axes[0].scatter(
            INPUT_RATES_HZ,
            empirical_mean[index],
            s=7,
            color=color,
            alpha=0.28,
            edgecolors="none",
        )
        axes[1].plot(INPUT_RATES_HZ, analytical_sd[index], color=color)
        axes[1].scatter(
            INPUT_RATES_HZ,
            empirical_sd[index],
            s=7,
            color=color,
            alpha=0.28,
            edgecolors="none",
        )
    axes[0].set(title="A  Mean feature", ylabel="Mean feature z (mV)")
    axes[1].set(title="B  Feature SD", ylabel="Feature SD (mV)")
    for axis in axes:
        axis.set(xlabel="Input rate (Hz)", xlim=(0, 25))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.14)
    axes[0].legend(frameon=False)
    fig.savefig(FIGURES / "analytical_empirical.svg", metadata={"Date": None})
    plt.close(fig)


def summarize(predicted: np.ndarray, empirical: np.ndarray) -> dict[str, float]:
    valid = (predicted > 0) | (empirical > 0)
    positive = valid & (empirical > 0)
    return {
        "pearson_r": float(np.corrcoef(predicted[valid], empirical[valid])[0, 1]),
        "mean_absolute_error_mV": float(
            np.mean(np.abs(predicted[valid] - empirical[valid]))
        ),
        "median_predicted_empirical_ratio": float(
            np.median(predicted[positive] / empirical[positive])
        ),
    }


@preserve_active_view(SLUG)
def main() -> None:
    started = time.perf_counter()
    FIGURES.mkdir(parents=True, exist_ok=True)
    rates_hz, probes = np.meshgrid(INPUT_RATES_HZ, np.asarray(PROBES_US))
    features = simulate_features(rates_hz, probes, MOMENT_DRAWS, stable_seed(1))
    empirical_mean = features.mean(axis=-1)
    empirical_sd = features.std(axis=-1, ddof=1)
    distribution_samples = simulate_features(
        np.asarray(DISTRIBUTION_RATES_HZ),
        np.full(len(DISTRIBUTION_RATES_HZ), NOMINAL_PROBE_US),
        DISTRIBUTION_DRAWS,
        stable_seed(2),
    )
    _, stationary_voltage = linear_operating_point(rates_hz, probes)
    analytical_mean = stationary_voltage - PARAMETERS["E_L_mV"]
    analytical_variance = predicted_variance(
        rates_hz.reshape(-1), probes.reshape(-1)
    ).reshape(rates_hz.shape)
    analytical_sd = np.sqrt(analytical_variance)
    coarse = predicted_variance(
        rates_hz.reshape(-1), probes.reshape(-1), grid_points=COARSE_GRID_POINTS
    ).reshape(rates_hz.shape)
    relative = np.divide(
        np.abs(analytical_variance - coarse),
        analytical_variance,
        out=np.zeros_like(analytical_variance),
        where=analytical_variance > 0,
    )
    np.savez_compressed(
        FIGURES / "moments.npz",
        input_rates_hz=rates_hz,
        probes_uS=probes,
        empirical_mean_mV=empirical_mean,
        empirical_sd_mV=empirical_sd,
        analytical_mean_mV=analytical_mean,
        analytical_sd_mV=analytical_sd,
    )
    np.savez_compressed(
        FIGURES / "distribution_samples.npz",
        input_rates_hz=np.asarray(DISTRIBUTION_RATES_HZ),
        samples_mV=distribution_samples,
    )
    plot_moments(empirical_mean, empirical_sd)
    plot_distributions(distribution_samples)
    plot_frequency_response()
    plot_comparison(empirical_mean, empirical_sd, analytical_mean, analytical_sd)
    numbers = {
        "status": "complete",
        "purpose": "derive and test the stationary linear-filter description of the finite-window pixel feature",
        "parameters": {
            "presentation_ms": PRESENTATION_MS,
            "dt_ms": DT_MS,
            "probes_uS": list(PROBES_US),
            "pixel_intensity": 1.0,
            "input_rate_grid_hz": INPUT_RATES_HZ.tolist(),
            "distribution_rates_hz": list(DISTRIBUTION_RATES_HZ),
            "frequency_response_rates_hz": list(FREQUENCY_RESPONSE_RATES_HZ),
            "frequency_plot_bounds_hz": list(FREQUENCY_PLOT_BOUNDS_HZ),
            "nominal_probe_uS": NOMINAL_PROBE_US,
            "moment_draws": MOMENT_DRAWS,
            "distribution_draws": DISTRIBUTION_DRAWS,
        },
        "comparison": {
            "mean": summarize(analytical_mean, empirical_mean),
            "standard_deviation": summarize(analytical_sd, empirical_sd),
        },
        "quadrature": {
            "maximum_relative_refinement_change": float(relative.max()),
            "fine_grid_points": FREQUENCY_GRID_POINTS,
            "coarse_grid_points": COARSE_GRID_POINTS,
        },
        "runtime_s": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "device": str(torch_device()),
        },
    }
    run_id = next_run_id(SLUG)
    write_numbers(
        FIGURES,
        run_id=run_id,
        duration_s=numbers["runtime_s"],
        payload=numbers,
    )
    (FIGURES / "reproducer.json").write_text(
        json.dumps({"command": "uv run python experiments/exp081.py"}, indent=2) + "\n"
    )
    (FIGURES / "_run.txt").write_text(f"{int(run_id.lstrip('r'))}\n")
    print("exp081 complete", flush=True)
    finalize_prepared_run(SLUG, run_id)


if __name__ == "__main__":
    main()
