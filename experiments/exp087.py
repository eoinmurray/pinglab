"""EXP087: demonstrate stable pulse-packet propagation in a synfire chain."""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "snn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from execution import ExecutionSpec, simulate  # noqa: E402
from tools import snnlang as snn  # noqa: E402, TID251

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp087"
STATUS = "draft"

DT_MS = 0.1
LAYERS = 6
NEURONS_PER_LAYER = 100
PACKET_CHANNELS = 100
BACKGROUND_CHANNELS = 600
FEEDFORWARD_FAN_IN = 40
FEEDFORWARD_TOTAL_STRENGTH_US = 0.12
FEEDFORWARD_DELAY_MS = 1.0
BACKGROUND_FAN_IN = 100
BACKGROUND_EXCITATORY_STRENGTH_US = 0.24
BACKGROUND_INHIBITORY_STRENGTHS_US = (
    0.074,
    0.084078,
    0.087438,
    0.086094,
    0.084078,
    0.088781,
)
BACKGROUND_RATE_HZ = 100.0
TARGET_OUTPUT_RATE_HZ = 10.0
T_MS = 100.0
PACKET_CENTRE_MS = 70.0
NETWORK_SEED = 11
BACKGROUND_SEED = 7
BACKGROUND_SETTLED_START_MS = 30.0
RESPONSE_START_MS = 65.0
RESPONSE_END_MS = 95.0
VOLLEY_PEAK_WINDOW_MS = 1.0
VOLLEY_MEASUREMENT_WINDOW_MS = 3.0
VOLLEY_MIN_PEAK_SPIKES_BY_POOL = (12, 14, 17, 19, 24, 26)
STATE_ALPHAS = (20, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100)
STATE_SIGMAS_MS = (0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
REPRESENTATIVE_PACKETS = (
    ("weak", "Weak, diffuse", 20, 5.0),
    ("broad", "Broad, strong", 80, 3.0),
    ("oversized", "Narrow, oversized", 100, 0.2),
)
REFERENCE_PACKET = (80, 3.0)
METHOD_SVG_NAMES = (
    "packet_definition.svg",
    "packet_fates.svg",
    "synfire_state_space.svg",
    "raster_hero_storyboard.svg",
)

SCALE = {
    "status": STATUS,
    "simulation_run": True,
    "completed_methods": [1, 2, 3, 4],
    "dt_ms": DT_MS,
    "layers": LAYERS,
    "neurons_per_layer": NEURONS_PER_LAYER,
    "packet_channels": PACKET_CHANNELS,
    "background_channels": BACKGROUND_CHANNELS,
    "feedforward_fan_in": FEEDFORWARD_FAN_IN,
    "feedforward_total_strength_us": FEEDFORWARD_TOTAL_STRENGTH_US,
    "feedforward_delay_ms": FEEDFORWARD_DELAY_MS,
    "background_fan_in": BACKGROUND_FAN_IN,
    "background_excitatory_strength_us": BACKGROUND_EXCITATORY_STRENGTH_US,
    "background_inhibitory_strengths_us": BACKGROUND_INHIBITORY_STRENGTHS_US,
    "background_rate_hz": BACKGROUND_RATE_HZ,
    "target_output_rate_hz": TARGET_OUTPUT_RATE_HZ,
    "t_ms": T_MS,
    "packet_centre_ms": PACKET_CENTRE_MS,
    "network_seed": NETWORK_SEED,
    "background_seed": BACKGROUND_SEED,
}


@dataclass
class PacketRun:
    """Measured response to one pulse packet under one fixed network state."""

    packet_id: str
    label: str
    input_alpha: int
    requested_sigma_ms: float
    input_sigma_ms: float
    input_spikes: np.ndarray
    pool_spikes: list[np.ndarray]
    alphas: list[int]
    sigmas_ms: list[float | None]
    volley_windows: list[tuple[int, int] | None]

    @property
    def survives(self) -> bool:
        return self.alphas[-1] > 0


def exact_fan_in(total_strength: float, fan_in: int, source_size: int):
    """Create a fan-in-normalized sparse initializer."""
    return snn.LowerClampedNormal(
        total_strength,
        0.0,
        initial_zero_fraction=1.0 - fan_in / source_size,
        zeroing="exact_k",
    )


def author_network(
    *,
    feedforward_strength_us: float = FEEDFORWARD_TOTAL_STRENGTH_US,
    background_excitatory_strength_us: float
    | Sequence[float] = BACKGROUND_EXCITATORY_STRENGTH_US,
    background_inhibitory_strength_us: float
    | Sequence[float] = BACKGROUND_INHIBITORY_STRENGTHS_US,
) -> snn.Bundle:
    """Author six feedforward pools with explicit packet and background inputs."""
    net = snn.Network("diesmann_synfire_chain", dt=DT_MS * snn.ms)
    packet = net.input(
        "pulse_packet",
        shape=("time", "batch", PACKET_CHANNELS),
        signal_type="spikes",
        unit="spike",
    )
    background_excitation = net.input(
        "background_excitation",
        shape=("time", "batch", BACKGROUND_CHANNELS),
        signal_type="spikes",
        unit="spike",
    )
    background_inhibition = net.input(
        "background_inhibition",
        shape=("time", "batch", BACKGROUND_CHANNELS),
        signal_type="spikes",
        unit="spike",
    )

    pools = []
    with net.group("synfire_chain"):
        for layer in range(1, LAYERS + 1):
            excitatory_strength = (
                background_excitatory_strength_us
                if isinstance(background_excitatory_strength_us, (int, float))
                else background_excitatory_strength_us[layer - 1]
            )
            inhibitory_strength = (
                background_inhibitory_strength_us
                if isinstance(background_inhibitory_strength_us, (int, float))
                else background_inhibitory_strength_us[layer - 1]
            )
            pool = net.population(
                f"pool_{layer}",
                size=NEURONS_PER_LAYER,
                neuron=snn.COBA_LIF(
                    tau_mem=20 * snn.ms,
                    capacitance_nf=1.0,
                    leak_us=0.05,
                    resting_mv=-65.0,
                    threshold_mv=-50.0,
                    reset_mv=-65.0,
                    refractory_steps=round(2.0 / DT_MS),
                    voltage_grad_dampen=80.0,
                    initial_voltage_mv=-65.0,
                ),
            )
            pools.append(pool)
            net.connect(
                background_excitation,
                pool.excitatory,
                name=f"background_excitation_to_pool_{layer}",
                synapse=snn.AMPA(tau=2 * snn.ms),
                weight=exact_fan_in(
                    excitatory_strength,
                    BACKGROUND_FAN_IN,
                    BACKGROUND_CHANNELS,
                ),
                constraint=snn.NonNegative(),
            )
            net.connect(
                background_inhibition,
                pool.inhibitory,
                name=f"background_inhibition_to_pool_{layer}",
                synapse=snn.GABA(tau=9 * snn.ms),
                weight=exact_fan_in(
                    inhibitory_strength,
                    BACKGROUND_FAN_IN,
                    BACKGROUND_CHANNELS,
                ),
                constraint=snn.NonNegative(),
            )

        net.connect(
            packet,
            pools[0].excitatory,
            name="packet_to_pool_1",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=exact_fan_in(
                feedforward_strength_us,
                FEEDFORWARD_FAN_IN,
                PACKET_CHANNELS,
            ),
            constraint=snn.NonNegative(),
            delay=FEEDFORWARD_DELAY_MS * snn.ms,
        )
        for layer, (source, target) in enumerate(
            zip(pools[:-1], pools[1:], strict=True),
            start=1,
        ):
            net.connect(
                source.spikes,
                target.excitatory,
                name=f"pool_{layer}_to_pool_{layer + 1}",
                synapse=snn.AMPA(tau=2 * snn.ms),
                weight=exact_fan_in(
                    feedforward_strength_us,
                    FEEDFORWARD_FAN_IN,
                    NEURONS_PER_LAYER,
                ),
                constraint=snn.NonNegative(),
                connection="feedforward",
                delay=FEEDFORWARD_DELAY_MS * snn.ms,
            )

    net.expose(*(pool.spikes for pool in pools), name="pool_spikes")
    return snn.compile(net, target="tools/snn")


def make_background(
    rate_hz: float = BACKGROUND_RATE_HZ,
) -> dict[str, torch.Tensor]:
    """Generate fixed independent excitatory and inhibitory background spikes."""
    steps = round(T_MS / DT_MS)
    probability = rate_hz * DT_MS / 1_000.0
    shape = (steps, 1, BACKGROUND_CHANNELS)
    backgrounds = {}
    for name, seed in (
        ("background_excitation", BACKGROUND_SEED),
        ("background_inhibition", BACKGROUND_SEED + 1),
    ):
        rng = np.random.default_rng(seed)
        spikes = rng.random(shape, dtype=np.float32) < probability
        backgrounds[name] = torch.from_numpy(spikes.astype(np.float32))
    return backgrounds


def make_packet(alpha: int, sigma_ms: float) -> torch.Tensor:
    """Create one deterministic packet with alpha spikes and requested width."""
    if not 0 < alpha <= PACKET_CHANNELS:
        raise ValueError("packet alpha must lie between 1 and PACKET_CHANNELS")
    steps = round(T_MS / DT_MS)
    seed = 87_000 + 100 * alpha + round(10 * sigma_ms)
    rng = np.random.default_rng(seed)
    channels = rng.choice(PACKET_CHANNELS, size=alpha, replace=False)
    centre_step = round(PACKET_CENTRE_MS / DT_MS)
    times = np.rint(rng.normal(centre_step, sigma_ms / DT_MS, size=alpha)).astype(int)
    times = np.clip(times, 0, steps - 1)
    spikes = np.zeros((steps, 1, PACKET_CHANNELS), dtype=np.float32)
    spikes[times, 0, channels] = 1.0
    return torch.from_numpy(spikes)


def packet_width_ms(spikes: np.ndarray) -> float | None:
    """Return the standard deviation of pooled spike times."""
    times, _batch, _neurons = np.nonzero(spikes)
    if times.size < 2:
        return None
    return float(np.std(times.astype(float) * DT_MS))


def measure_volley(
    spikes: np.ndarray,
    *,
    search_start: int,
    search_end: int,
    min_peak_spikes: int,
) -> tuple[int, float | None, tuple[int, int] | None, int | None]:
    """Measure a synchronous volley above the calibrated background maximum."""
    start = round(RESPONSE_START_MS / DT_MS)
    end = round(RESPONSE_END_MS / DT_MS)
    search_start = max(start, search_start)
    search_end = min(end, search_end)
    peak_window = round(VOLLEY_PEAK_WINDOW_MS / DT_MS)
    counts = spikes[search_start:search_end].sum(axis=(1, 2)).astype(int)
    if counts.size < peak_window:
        return 0, None, None, None
    cumulative = np.pad(np.cumsum(counts), (1, 0))
    window_counts = cumulative[peak_window:] - cumulative[:-peak_window]
    best = int(np.argmax(window_counts))
    peak_alpha = int(window_counts[best])
    if peak_alpha < min_peak_spikes:
        return 0, None, None, None
    peak_step = search_start + best + peak_window // 2
    measurement_window = round(VOLLEY_MEASUREMENT_WINDOW_MS / DT_MS)
    window_start = max(start, peak_step - measurement_window // 2)
    window_end = min(end, window_start + measurement_window)
    selected = spikes[window_start:window_end]
    _steps, _batch, neurons = np.nonzero(selected)
    alpha = int(np.unique(neurons).size)
    return alpha, packet_width_ms(selected), (window_start, window_end), peak_step


def run_background_only(
    graph: dict,
    background: dict[str, torch.Tensor],
) -> dict[str, list[float] | float]:
    """Measure settled background rate and its strongest 1 ms coincidence."""
    steps = round(T_MS / DT_MS)
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={
                "pulse_packet": torch.zeros((steps, 1, PACKET_CHANNELS)),
                **background,
            },
            seed=NETWORK_SEED,
        )
    )
    settled_start = round(BACKGROUND_SETTLED_START_MS / DT_MS)
    peak_window = round(VOLLEY_PEAK_WINDOW_MS / DT_MS)
    settled_duration_s = (T_MS - BACKGROUND_SETTLED_START_MS) / 1_000.0
    counts = []
    rates_hz = []
    peak_counts = []
    for layer in range(LAYERS):
        spikes = result.recordings[f"pool_spikes_{layer}"].cpu().numpy()
        population_counts = spikes[settled_start:].sum(axis=(1, 2)).astype(int)
        count = int(population_counts.sum())
        cumulative = np.pad(np.cumsum(population_counts), (1, 0))
        window_counts = cumulative[peak_window:] - cumulative[:-peak_window]
        counts.append(count)
        rates_hz.append(count / (NEURONS_PER_LAYER * settled_duration_s))
        peak_counts.append(int(window_counts.max(initial=0)))
    return {
        "settled_spikes_by_pool": counts,
        "settled_rate_hz_by_pool": rates_hz,
        "mean_settled_rate_hz": float(np.mean(rates_hz)),
        "max_1ms_spikes_by_pool": peak_counts,
    }


def run_packet(
    graph: dict,
    *,
    packet_id: str,
    label: str,
    alpha: int,
    sigma_ms: float,
    background: dict[str, torch.Tensor],
) -> PacketRun:
    """Simulate and measure one packet trajectory through the chain."""
    input_alpha = alpha
    packet = make_packet(alpha, sigma_ms)
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={
                "pulse_packet": packet,
                **background,
            },
            seed=NETWORK_SEED,
        )
    )
    pool_spikes = []
    alphas = []
    sigmas_ms = []
    volley_windows = []
    extinct = False
    search_start = round((PACKET_CENTRE_MS - 5.0) / DT_MS)
    search_end = round((PACKET_CENTRE_MS + 10.0) / DT_MS)
    for layer in range(LAYERS):
        spikes = (
            result.recordings[f"pool_spikes_{layer}"].cpu().numpy().astype(np.uint8)
        )
        pool_spikes.append(spikes)
        if extinct:
            measured_alpha = 0
            sigma_ms_measured = None
            volley_window = None
            peak_step = None
        else:
            (
                measured_alpha,
                sigma_ms_measured,
                volley_window,
                peak_step,
            ) = measure_volley(
                spikes,
                search_start=search_start,
                search_end=search_end,
                min_peak_spikes=VOLLEY_MIN_PEAK_SPIKES_BY_POOL[layer],
            )
            extinct = measured_alpha == 0
            if peak_step is not None:
                search_start = peak_step + round(0.5 / DT_MS)
                search_end = peak_step + round(10.0 / DT_MS)
        alphas.append(measured_alpha)
        sigmas_ms.append(sigma_ms_measured)
        volley_windows.append(volley_window)
    return PacketRun(
        packet_id=packet_id,
        label=label,
        input_alpha=input_alpha,
        requested_sigma_ms=sigma_ms,
        input_sigma_ms=packet_width_ms(packet.numpy()) or 0.0,
        input_spikes=packet.numpy().astype(np.uint8),
        pool_spikes=pool_spikes,
        alphas=alphas,
        sigmas_ms=sigmas_ms,
        volley_windows=volley_windows,
    )


def run_operating_point_search() -> tuple[float, float, list[dict[str, object]]]:
    """Choose the weakest clean reference-packet propagation point."""
    rows = []
    for rate_hz in (BACKGROUND_RATE_HZ,):
        background = make_background(rate_hz)
        for strength_us in (0.10, 0.12, 0.14):
            graph = author_network(feedforward_strength_us=strength_us).graph
            run = run_packet(
                graph,
                packet_id="reference",
                label="Reference",
                alpha=REFERENCE_PACKET[0],
                sigma_ms=REFERENCE_PACKET[1],
                background=background,
            )
            background_metrics = run_background_only(graph, background)
            settled_rates = background_metrics["settled_rate_hz_by_pool"]
            peak_counts = background_metrics["max_1ms_spikes_by_pool"]
            clean = (
                all(8.0 <= value <= 12.0 for value in settled_rates)
                and all(
                    peak < threshold
                    for peak, threshold in zip(
                        peak_counts,
                        VOLLEY_MIN_PEAK_SPIKES_BY_POOL,
                        strict=True,
                    )
                )
                and all(value > 0 for value in run.alphas)
                and max(run.alphas) <= NEURONS_PER_LAYER
                and min(run.alphas[-2:]) >= 40
            )
            rows.append(
                {
                    "feedforward_strength_us": strength_us,
                    "background_rate_hz": rate_hz,
                    "alphas": run.alphas,
                    "sigmas_ms": run.sigmas_ms,
                    "background": background_metrics,
                    "accepted": clean,
                }
            )
    accepted = [row for row in rows if row["accepted"]]
    if not accepted:
        raise RuntimeError("operating-point search found no clean propagation")
    selected = min(
        accepted,
        key=lambda row: (
            row["feedforward_strength_us"],
            abs(row["background_rate_hz"] - BACKGROUND_RATE_HZ),
        ),
    )
    return (
        float(selected["feedforward_strength_us"]),
        float(selected["background_rate_hz"]),
        rows,
    )


def run_state_space(
    graph: dict,
    background: dict[str, torch.Tensor],
) -> list[PacketRun]:
    """Measure one layer trajectory at every requested packet state."""
    runs = []
    for alpha in STATE_ALPHAS:
        for sigma_ms in STATE_SIGMAS_MS:
            runs.append(
                run_packet(
                    graph,
                    packet_id=f"a{alpha}_s{sigma_ms:g}",
                    label=f"α={alpha}, σ={sigma_ms:g} ms",
                    alpha=alpha,
                    sigma_ms=sigma_ms,
                    background=background,
                )
            )
    return runs


def raster_points(run: PacketRun) -> tuple[np.ndarray, np.ndarray]:
    """Stack six pool rasters into one neuron-index axis."""
    times = []
    neurons = []
    for layer, spikes in enumerate(run.pool_spikes):
        step, _batch, neuron = np.nonzero(spikes)
        keep = (step * DT_MS >= RESPONSE_START_MS) & (step * DT_MS <= RESPONSE_END_MS)
        times.append(step[keep] * DT_MS)
        neurons.append(neuron[keep] + layer * NEURONS_PER_LAYER)
    return np.concatenate(times), np.concatenate(neurons)


def style_raster_axis(ax: plt.Axes) -> None:
    """Apply the shared six-band raster axes."""
    for boundary in range(1, LAYERS):
        ax.axhline(
            boundary * NEURONS_PER_LAYER - 0.5,
            color=theme.RULE,
            linewidth=0.8,
        )
    ax.set(
        xlim=(RESPONSE_START_MS, RESPONSE_END_MS),
        ylim=(LAYERS * NEURONS_PER_LAYER - 0.5, -0.5),
        yticks=[(layer + 0.5) * NEURONS_PER_LAYER for layer in range(LAYERS)],
        yticklabels=[f"P{layer}" for layer in range(1, LAYERS + 1)],
        xlabel="time (ms)",
        ylabel="pool",
    )


def plot_reference(run: PacketRun, out: Path) -> None:
    """Plot the selected reference packet through all six pools."""
    theme.apply()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.0, 7.2),
        gridspec_kw={"height_ratios": (2.8, 1.0, 1.0)},
    )
    time_ms, neurons = raster_points(run)
    axes[0].scatter(time_ms, neurons, marker="|", s=15, color=theme.INK_BLACK)
    style_raster_axis(axes[0])
    axes[0].set_title("One packet is regenerated through all six pools")
    pools = np.arange(0, LAYERS + 1)
    alphas = [run.input_alpha, *run.alphas]
    sigmas = [run.input_sigma_ms, *[value or 0.0 for value in run.sigmas_ms]]
    axes[1].plot(pools, alphas, "o-", color=theme.INK_BLACK)
    axes[1].set(ylabel="packet size α", xticks=pools, xticklabels=[])
    axes[2].plot(pools, sigmas, "o-", color=theme.DEEP_RED)
    axes[2].set(
        xlabel="input and pool number",
        ylabel="spread σ (ms)",
        xticks=pools,
        xticklabels=["input", *[f"P{i}" for i in range(1, LAYERS + 1)]],
    )
    for ax in axes[1:]:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_packet_fates(runs: list[PacketRun], out: Path) -> None:
    """Mirror the three-column Methods schematic with measured trajectories."""
    theme.apply()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.0, 6.8),
        gridspec_kw={"height_ratios": (2.5, 1.25)},
    )
    pools = np.arange(0, LAYERS + 1)
    for column, run in enumerate(runs):
        time_ms, neurons = raster_points(run)
        axes[0, column].scatter(
            time_ms,
            neurons,
            marker="|",
            s=12,
            color=theme.CYCLE[column],
        )
        style_raster_axis(axes[0, column])
        axes[0, column].set_title(run.label)
        if column:
            axes[0, column].set_ylabel("")
            axes[0, column].set_yticklabels([])
        alpha_axis = axes[1, column]
        sigma_axis = alpha_axis.twinx()
        alpha_axis.plot(
            pools,
            [run.input_alpha, *run.alphas],
            "o-",
            color=theme.INK_BLACK,
            label="α",
        )
        sigma_axis.plot(
            pools,
            [run.input_sigma_ms, *[value or 0.0 for value in run.sigmas_ms]],
            "o-",
            color=theme.DEEP_RED,
            label="σ",
        )
        alpha_axis.set(
            xlabel="input and pool",
            ylabel="α" if column == 0 else "",
            xticks=pools,
            xticklabels=["in", *[str(i) for i in range(1, LAYERS + 1)]],
            ylim=(-5, max(110, max([run.input_alpha, *run.alphas]) * 1.08)),
        )
        sigma_axis.set(ylabel="σ (ms)" if column == 2 else "")
        if column != 2:
            sigma_axis.set_yticklabels([])
        alpha_axis.spines["top"].set_visible(False)
        sigma_axis.spines["top"].set_visible(False)
    fig.suptitle("Measured packets either die or approach the stable packet", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def grid_edges(values: tuple[float, ...] | tuple[int, ...]) -> np.ndarray:
    """Return cell edges for an irregular one-dimensional grid."""
    centres = np.asarray(values, dtype=float)
    mids = (centres[:-1] + centres[1:]) / 2.0
    return np.concatenate(
        (
            [centres[0] - (mids[0] - centres[0])],
            mids,
            [centres[-1] + (centres[-1] - mids[-1])],
        )
    )


def plot_state_space(
    grid_runs: list[PacketRun],
    representative_runs: list[PacketRun],
    out: Path,
) -> None:
    """Show the measured extinction boundary and representative trajectories."""
    theme.apply()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4))
    fate = np.zeros((len(STATE_SIGMAS_MS), len(STATE_ALPHAS)))
    by_state = {(run.input_alpha, run.requested_sigma_ms): run for run in grid_runs}
    for row, sigma_ms in enumerate(STATE_SIGMAS_MS):
        for column, alpha in enumerate(STATE_ALPHAS):
            fate[row, column] = by_state[(alpha, sigma_ms)].survives
    axes[0].pcolormesh(
        grid_edges(STATE_ALPHAS),
        grid_edges(STATE_SIGMAS_MS),
        fate,
        cmap=ListedColormap(["#eceae4", "#d8edf4"]),
        edgecolors="white",
        linewidth=1.2,
        shading="flat",
    )
    axes[0].set(
        title="Initial packet fate",
        xlabel="input packet size α",
        ylabel="input spread σ (ms)",
        xticks=STATE_ALPHAS,
        yticks=STATE_SIGMAS_MS,
    )
    axes[0].text(28, 4.4, "extinction", color=theme.DIM)
    axes[0].text(72, 1.1, "propagation", color="#277da1")

    colors = (theme.DEEP_RED, theme.ELECTRIC_CYAN, theme.AMBER)
    for run, color in zip(representative_runs, colors, strict=True):
        x = [float(run.input_alpha)]
        y = [run.input_sigma_ms]
        for alpha, sigma_ms in zip(run.alphas, run.sigmas_ms, strict=True):
            if alpha <= 0 or sigma_ms is None:
                x.append(0.0)
                y.append(y[-1])
                break
            x.append(float(alpha))
            y.append(float(sigma_ms))
        axes[1].plot(x, y, "o-", color=color, label=run.label)
        for start_x, start_y, end_x, end_y in zip(
            x[:-1], y[:-1], x[1:], y[1:], strict=True
        ):
            axes[1].annotate(
                "",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.6},
            )
    axes[1].set(
        title="Layer-to-layer packet transformation",
        xlabel="packet size α",
        ylabel="spread σ (ms)",
        xlim=(-5, 108),
        ylim=(-0.1, 5.5),
    )
    axes[1].legend(frameon=False, loc="upper right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_raster_hero(runs: list[PacketRun], out: Path) -> None:
    """Animate the three measured rasters using Figure 7's top-row layout."""
    period_s = 8.0
    fps = 30
    window_ms = 30.0
    loop_start_ms = BACKGROUND_SETTLED_START_MS
    final_window_start_ms = T_MS - window_ms
    frames = round(period_s * fps)

    theme.apply()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.2), sharey=True)
    title = fig.suptitle("Measured packet trajectories rolling through time")
    for column, (ax, run) in enumerate(zip(axes, runs, strict=True)):
        event_times = []
        event_neurons = []
        for layer, spikes in enumerate(run.pool_spikes):
            steps, _batch, neurons = np.nonzero(spikes)
            keep = (steps * DT_MS >= loop_start_ms) & (steps * DT_MS < T_MS)
            event_times.extend(steps[keep] * DT_MS)
            event_neurons.extend(neurons[keep] + layer * NEURONS_PER_LAYER)
        for boundary in range(1, LAYERS):
            ax.axhline(
                boundary * NEURONS_PER_LAYER - 0.5,
                color=theme.RULE,
                linewidth=0.8,
            )
        ax.scatter(
            event_times,
            event_neurons,
            marker="|",
            s=12,
            linewidths=0.9,
            color=theme.CYCLE[column],
        )
        ax.set(
            title=run.label,
            xlim=(loop_start_ms, loop_start_ms + window_ms),
            ylim=(LAYERS * NEURONS_PER_LAYER - 0.5, -0.5),
            yticks=[
                (layer + 0.5) * NEURONS_PER_LAYER for layer in range(LAYERS)
            ],
            yticklabels=[f"P{layer}" for layer in range(1, LAYERS + 1)],
            xlabel="time (ms)",
        )
        if column == 0:
            ax.set_ylabel("pool")
        else:
            ax.tick_params(labelleft=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    def update(frame: int):
        fraction = frame / max(frames - 1, 1)
        window_start = loop_start_ms + fraction * (
            final_window_start_ms - loop_start_ms
        )
        window_end = window_start + window_ms
        for ax in axes:
            ax.set_xlim(window_start, window_end)
        title.set_text(
            "Measured packet trajectories rolling through time"
            f" · {window_start:04.1f}–{window_end:04.1f} ms"
        )
        return (*axes, title)

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1_000 / fps,
        blit=False,
    )
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=2_400,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    movie.save(out, writer=writer, dpi=120)
    plt.close(fig)


def run_record(run: PacketRun) -> dict[str, object]:
    """Convert one measured packet trajectory to JSON-safe values."""
    return {
        "id": run.packet_id,
        "label": run.label,
        "input": {
            "alpha": run.input_alpha,
            "requested_sigma_ms": run.requested_sigma_ms,
            "realised_sigma_ms": run.input_sigma_ms,
        },
        "layers": [
            {
                "pool": layer,
                "alpha": alpha,
                "sigma_ms": sigma_ms,
                "volley_window_ms": None
                if window is None
                else [window[0] * DT_MS, window[1] * DT_MS],
            }
            for layer, (alpha, sigma_ms, window) in enumerate(
                zip(
                    run.alphas,
                    run.sigmas_ms,
                    run.volley_windows,
                    strict=True,
                ),
                start=1,
            )
        ],
        "survives": run.survives,
    }


def main() -> None:
    meta = parse_meta(sys.argv)
    if meta.runpod:
        raise SystemExit("exp087 is a bounded local experiment")
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    current_artifacts = REPO / "artifacts" / "data" / SLUG
    with published_run(SLUG, run_id, scale=SCALE) as (_scratch, staging):
        for name in METHOD_SVG_NAMES:
            shutil.copy2(current_artifacts / name, staging / name)

        selected_strength, selected_rate, search_rows = run_operating_point_search()
        bundle = author_network(feedforward_strength_us=selected_strength)
        bundle.write(staging / "network.bundle", visualise=True)
        bundle.visualise(
            staging / "network.svg",
            view="circuit",
            expand_groups=("synfire_chain",),
        )
        background = make_background(selected_rate)
        background_metrics = run_background_only(bundle.graph, background)
        representative_runs = [
            run_packet(
                bundle.graph,
                packet_id=packet_id,
                label=label,
                alpha=alpha,
                sigma_ms=sigma_ms,
                background=background,
            )
            for packet_id, label, alpha, sigma_ms in REPRESENTATIVE_PACKETS
        ]
        reference = representative_runs[1]
        plot_reference(reference, staging / "reference_propagation.png")
        plot_packet_fates(representative_runs, staging / "packet_fates_measured.png")

        grid_runs = run_state_space(bundle.graph, background)
        plot_state_space(
            grid_runs,
            representative_runs,
            staging / "packet_state_space.png",
        )
        render_raster_hero(
            representative_runs,
            staging / "synfire_raster_hero.mp4",
        )

        trace_payload: dict[str, np.ndarray] = {}
        for run in representative_runs:
            trace_payload[f"{run.packet_id}_input"] = run.input_spikes
            for layer, spikes in enumerate(run.pool_spikes, start=1):
                trace_payload[f"{run.packet_id}_pool_{layer}"] = spikes
        np.savez_compressed(staging / "representative_spikes.npz", **trace_payload)

        record = {
            **SCALE,
            "question": "Can pulse packets converge to a stable size and width?",
            "operating_point": {
                "feedforward_strength_us": selected_strength,
                "background_rate_hz": selected_rate,
                "selection": "weakest tested strength carrying the reference packet cleanly through all six pools",
                "background": background_metrics,
                "search": search_rows,
            },
            "representative_packets": [run_record(run) for run in representative_runs],
            "state_space": [run_record(run) for run in grid_runs],
            "remaining_methods_unrun": [],
        }
        (staging / "protocol.json").write_text(json.dumps(record, indent=2) + "\n")
        write_numbers(
            staging,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload=record,
        )


if __name__ == "__main__":
    main()
