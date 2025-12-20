
from __future__ import annotations

import sys
from pathlib import Path
import shutil
import yaml
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

from lib.model import LocalConfig
from lib.projection import build_input_projection
from lib.packet import generate_packet_spike_times, stack_spike_times
from lib.phase import estimate_phase
from lib.input_drive import add_fiber_spikes_to_input

from pinglab import run_network, inputs
from pinglab.plots.styles import save_both, figsize
from pinglab.plots import save_raster
from pinglab.utils import slice_spikes

def main() -> None:
    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)
    config = LocalConfig.model_validate(data)

    projection = build_input_projection(
        N_E=config.base.N_E,
        num_fibers=config.projection.num_fibers,
        targets_per_fiber=config.projection.targets_per_fiber,
        seed=config.projection.seed,
    )

    print(
        f"Loaded config for {Path(__file__).parent.name} "
        f"with N_E={config.base.N_E}, N_I={config.base.N_I}"
    )
    print(
        "Projection:",
        f"shape={projection.shape}",
        f"first_fiber_targets={projection[0, :5].tolist()}",
    )

    times = np.linspace(
        config.packet.times.start,
        config.packet.times.stop,
        config.packet.times.num,
    )
    delays = np.linspace(
        config.packet.delays_ms.start,
        config.packet.delays_ms.stop,
        config.packet.delays_ms.num,
    )
    t0 = float(times[0] + delays[0])
    jitter = float(config.packet.jitter_ms[0])

    rng = np.random.default_rng(config.packet.seed)
    spikes_per_fiber = generate_packet_spike_times(
        num_fibers=config.projection.num_fibers,
        t0_ms=t0,
        width_ms=config.packet.width_ms,
        mean_spikes_per_fiber=config.packet.mean_spikes_per_fiber,
        jitter_ms=jitter,
        rng=rng,
    )
    spike_times, spike_fibers = stack_spike_times(spikes_per_fiber)
    print(
        "Packet:",
        f"t0_ms={t0:.2f}",
        f"jitter_ms={jitter:.2f}",
        f"total_spikes={spike_times.size}",
    )
    if spike_times.size > 0:
        order = np.argsort(spike_times)[:5]
        print(
            "Sample spikes:",
            f"times_ms={spike_times[order].tolist()}",
            f"fibers={spike_fibers[order].tolist()}",
        )

    t = np.arange(0.0, config.base.T, config.base.dt, dtype=np.float32)
    test_signal = np.sin(2.0 * np.pi * 40.0 * t / 1000.0)
    phase = estimate_phase(
        test_signal,
        dt_ms=config.base.dt,
        smoothing_ms=config.phase.smoothing_ms,
    )
    if phase.size > 0:
        print(
            "Phase test:",
            f"phase_sample={phase[:5].tolist()}",
        )

    num_steps = int(np.ceil(config.base.T / config.base.dt))
    baseline_input = inputs.tonic(
        N_E=config.base.N_E,
        N_I=config.base.N_I,
        I_E=config.default_inputs.I_E,
        I_I=config.default_inputs.I_I,
        noise_std=config.default_inputs.noise,
        num_steps=num_steps,
        seed=config.base.seed or 0,
    )
    baseline_result = run_network(config.base, external_input=baseline_input)
    print(f"Baseline run: spikes={len(baseline_result.spikes.times)}")

    if not baseline_result.instruments:
        raise RuntimeError("No instruments recorded; phase estimation needs g_i_mean_E.")

    def spike_rate_signal(
        *,
        spikes,
        is_E: bool,
        N: int,
        dt_ms: float,
        T_ms: float,
        bin_ms: float,
    ) -> np.ndarray:
        num_steps_local = int(np.ceil(T_ms / dt_ms))
        bin_steps = max(1, int(round(bin_ms / dt_ms)))
        counts = np.zeros(num_steps_local, dtype=np.float32)
        if is_E:
            mask = spikes.ids < config.base.N_E
        else:
            mask = spikes.ids >= config.base.N_E
        idx = (spikes.times[mask] / dt_ms).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < num_steps_local)]
        if idx.size:
            np.add.at(counts, idx, 1.0)
        if bin_steps > 1:
            kernel = np.ones(bin_steps, dtype=np.float32) / float(bin_steps)
            counts = np.convolve(counts, kernel, mode="same")
        rate_hz = counts / (N * (dt_ms / 1000.0))
        return rate_hz

    if config.phase.signal == "g_i_mean_E":
        phase_signal = np.asarray(
            baseline_result.instruments.g_i_mean_E, dtype=np.float32
        )
    elif config.phase.signal == "rate_E":
        phase_signal = spike_rate_signal(
            spikes=baseline_result.spikes,
            is_E=True,
            N=config.base.N_E,
            dt_ms=config.base.dt,
            T_ms=config.base.T,
            bin_ms=config.phase.rate_bin_ms,
        )
    elif config.phase.signal == "rate_I":
        phase_signal = spike_rate_signal(
            spikes=baseline_result.spikes,
            is_E=False,
            N=config.base.N_I,
            dt_ms=config.base.dt,
            T_ms=config.base.T,
            bin_ms=config.phase.rate_bin_ms,
        )
    else:
        raise ValueError(f"Unknown phase signal: {config.phase.signal}")

    phase_trace = estimate_phase(
        phase_signal,
        dt_ms=config.base.dt,
        smoothing_ms=config.phase.smoothing_ms,
    )
    print(
        "Phase trace:",
        f"signal={config.phase.signal}",
        f"phase_sample={phase_trace[:5].tolist()}",
    )

    def plot_phase_signal() -> None:
        t = np.arange(phase_signal.size) * config.base.dt
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(t, phase_signal, lw=0.6)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Phase signal")
        ax.set_title("Receiver LFP proxy (g_i_mean_E)")
        plt.tight_layout()

    def plot_phase_unwrapped() -> None:
        t = np.arange(phase_signal.size) * config.base.dt
        phase_unwrapped = np.unwrap(phase_trace)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(t, phase_unwrapped, lw=0.6)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Phase (rad, unwrapped)")
        ax.set_title("Receiver phase (unwrapped)")
        plt.tight_layout()

    save_both(data_path / "phase_signal.png", plot_phase_signal)
    save_both(data_path / "phase_unwrapped.png", plot_phase_unwrapped)

    target_mask = np.zeros(config.base.N_E, dtype=bool)
    for fiber_targets in projection:
        target_mask[fiber_targets] = True
    target_ids = np.where(target_mask)[0]

    def count_target_spikes_in_window(spikes, *, start_ms: float, stop_ms: float) -> int:
        t = spikes.times
        ids = spikes.ids
        mask = (t >= start_ms) & (t < stop_ms) & (ids < config.base.N_E)
        if not np.any(mask):
            return 0
        ids = ids[mask]
        return int(np.sum(target_mask[ids]))

    packet_times = np.linspace(
        config.packet.times.start,
        config.packet.times.stop,
        config.packet.times.num,
    )
    delays = np.linspace(
        config.packet.delays_ms.start,
        config.packet.delays_ms.stop,
        config.packet.delays_ms.num,
    )
    jitters = list(config.packet.jitter_ms)

    responses = np.zeros((len(jitters), len(delays)), dtype=np.float32)

    def run_condition(*, delay_ms: float, jitter_ms: float, seed: int):
        rng = np.random.default_rng(seed)
        spikes_lists: list[list[float]] = [
            [] for _ in range(config.projection.num_fibers)
        ]

        for t0 in packet_times:
            spikes_per_fiber = generate_packet_spike_times(
                num_fibers=config.projection.num_fibers,
                t0_ms=float(t0 + delay_ms),
                width_ms=config.packet.width_ms,
                mean_spikes_per_fiber=config.packet.mean_spikes_per_fiber,
                jitter_ms=float(jitter_ms),
                rng=rng,
            )
            for fid, arr in enumerate(spikes_per_fiber):
                if arr.size:
                    spikes_lists[fid].extend(arr.tolist())

        spikes_per_fiber = [
            np.array(times, dtype=np.float32) for times in spikes_lists
        ]

        baseline_input_trial = inputs.tonic(
            N_E=config.base.N_E,
            N_I=config.base.N_I,
            I_E=config.default_inputs.I_E,
            I_I=config.default_inputs.I_I,
            noise_std=config.default_inputs.noise,
            num_steps=num_steps,
            seed=seed,
        )
        pulse_input = add_fiber_spikes_to_input(
            baseline_input=baseline_input_trial,
            fiber_targets=projection,
            spikes_per_fiber=spikes_per_fiber,
            weight=config.projection.weight,
            dt=config.base.dt,
            pulse_width_ms=config.packet.width_ms,
        )
        result = run_network(config.base, external_input=pulse_input)
        return result, pulse_input

    for j, jitter in enumerate(jitters):
        for d, delay in enumerate(delays):
            trial_values = []
            for trial in range(config.packet.trials_per_condition):
                seed = int(config.packet.seed + trial)
                result, _ = run_condition(
                    delay_ms=float(delay),
                    jitter_ms=float(jitter),
                    seed=seed,
                )

                deltas = []
                for t0 in packet_times:
                    t_arrive = float(t0 + delay)
                    post = count_target_spikes_in_window(
                        result.spikes,
                        start_ms=t_arrive,
                        stop_ms=t_arrive + config.readout.window_ms,
                    )
                    baseline_post = count_target_spikes_in_window(
                        baseline_result.spikes,
                        start_ms=t_arrive,
                        stop_ms=t_arrive + config.readout.window_ms,
                    )
                    deltas.append(post - baseline_post)
                trial_values.append(float(np.mean(deltas)))

            responses[j, d] = float(np.mean(trial_values))
            print(
                "Sweep:",
                f"jitter={jitter:.2f}ms",
                f"delay={delay:.2f}ms",
                f"mean_evoked={responses[j, d]:.3f}",
            )

    def plot_response_curves() -> None:
        fig, ax = plt.subplots(figsize=figsize)
        for j, jitter in enumerate(jitters):
            ax.plot(delays, responses[j, :], marker="o", label=f"jitter={jitter:.1f} ms")
        ax.set_xlabel("Delay (ms)")
        ax.set_ylabel("Mean evoked response (Δ spikes)")
        ax.set_title("Phase-gated response vs delay")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()

    save_both(data_path / "response_vs_delay.png", plot_response_curves)

    best_idx = int(np.argmax(responses[0, :]))
    worst_idx = int(np.argmin(responses[0, :]))
    best_delay = float(delays[best_idx])
    worst_delay = float(delays[worst_idx])

    for label, delay in [("good", best_delay), ("bad", worst_delay)]:
        result, pulse_input = run_condition(
            delay_ms=delay,
            jitter_ms=float(jitters[0]),
            seed=int(config.packet.seed),
        )
        spikes_to_plot = result.spikes
        if config.plotting and config.plotting.raster:
            spikes_to_plot = slice_spikes(
                spikes_to_plot,
                start_time=config.plotting.raster.start_time,
                stop_time=config.plotting.raster.stop_time,
            )
        vertical_lines = [float(t0 + delay) for t0 in packet_times]
        save_raster(
            spikes_to_plot,
            path=data_path / f"raster_{label}.png",
            label=f"{label} phase (delay={delay:.2f} ms)",
            vertical_lines=vertical_lines,
            external_input=pulse_input,
            dt=config.base.dt,
        )


if __name__ == "__main__":
    main()
