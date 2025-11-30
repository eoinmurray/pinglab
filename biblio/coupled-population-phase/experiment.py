
from pathlib import Path
import shutil
import numpy as np

from pinglab.plots import save_raster
from pinglab.inputs import tonic
from pinglab.utils import load_config
from pinglab import run_network
from pinglab.multiprocessing import parallel

from local import (
    estimate_gamma_period,
    extract_spike_volley,
    measure_transfer_gain,
    add_pulse_to_input,
    add_volley_as_input,
    plot_baseline_traces,
    plot_coupled_raster,
    plot_phase_transfer_curve,
)


def inner(cfg: dict) -> tuple[float, float, object]:
    """Run single phase condition: B receives A's volley at shifted time."""
    config = cfg["config"]
    pop_B = cfg["pop_B"]
    baseline_input_B = cfg["baseline_input_B"]
    volley_times = cfg["volley_times"]
    phase_shift_ms = cfg["phase_shift_ms"]
    target_E_B = cfg["target_E_B"]
    volley_t = cfg["volley_t"]

    # Add volley as input to B at shifted phase
    input_B_with_volley = add_volley_as_input(
        baseline_input_B,
        target_E_B,
        volley_times,
        phase_shift_ms,
        config.coupling.g_AB,
        pop_B.base.tau_ampa,
        pop_B.base.dt,
    )

    # Run population B with volley input
    results_B = run_network(
        pop_B.base.model_copy(update={"external_input": input_B_with_volley})
    )

    # Measure transfer gain
    _, _, gain = measure_transfer_gain(
        results_B.spikes,
        target_E_B,
        volley_t + phase_shift_ms,
        config.pulse.pre_window_ms,
        config.pulse.post_window_ms,
    )

    return phase_shift_ms, gain, results_B.spikes


def main() -> None:
    print("=" * 80)
    print("COUPLED POPULATION PHASE EXPERIMENT")
    print("=" * 80)

    root = Path(__file__).parent
    data_path = root / "data"
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config = load_config(root / "config.yaml")

    if not config.populations or len(config.populations) != 2:
        raise ValueError("Experiment requires exactly two populations in config")

    # Unpack populations for convenience
    pop_A = config.populations[0]
    pop_B = config.populations[1]

    print("\n[1/5] Running Population A baseline...")

    num_steps_A = int(np.ceil(pop_A.base.T / pop_A.base.dt))
    baseline_input_A = tonic(
        N_E=pop_A.base.N_E,
        N_I=pop_A.base.N_I,
        I_E=pop_A.inputs.I_E,
        I_I=pop_A.inputs.I_I,
        noise_std=pop_A.inputs.noise,
        num_steps=num_steps_A,
        seed=pop_A.base.seed or 0,
    )

    baseline_A = run_network(
        pop_A.base.model_copy(update={"external_input": baseline_input_A})
    )

    if not baseline_A.instruments:
        raise RuntimeError("No instruments recorded for population A")

    # Save A baseline
    save_raster(
        baseline_A.spikes,
        path=data_path / "baseline_A_raster.png",
        label="Baseline A",
        dt=pop_A.base.dt,
        external_input=baseline_input_A,
    )

    print(f"   ✓ Population A: {len(baseline_A.spikes.times)} spikes")

    # ========================================================================
    # PHASE 2: Run Population B Baseline
    # ========================================================================
    print("\n[2/5] Running Population B baseline...")

    num_steps_B = int(np.ceil(pop_B.base.T / pop_B.base.dt))
    baseline_input_B = tonic(
        N_E=pop_B.base.N_E,
        N_I=pop_B.base.N_I,
        I_E=pop_B.inputs.I_E,
        I_I=pop_B.inputs.I_I,
        noise_std=pop_B.inputs.noise,
        num_steps=num_steps_B,
        seed=pop_B.base.seed or 1,
    )

    baseline_B = run_network(
        pop_B.base.model_copy(update={"external_input": baseline_input_B})
    )

    if not baseline_B.instruments:
        raise RuntimeError("No instruments recorded for population B")

    # Estimate gamma period for B
    lfp_B = np.array(baseline_B.instruments.g_i_mean_E)
    T_gamma_B_ms = estimate_gamma_period(lfp_B, dt=pop_B.base.dt)

    # Save B baseline
    save_raster(
        baseline_B.spikes,
        path=data_path / "baseline_B_raster.png",
        label="Baseline B",
        dt=pop_B.base.dt,
        external_input=baseline_input_B,
    )

    print(f"   ✓ Population B: {len(baseline_B.spikes.times)} spikes")
    print(f"   ✓ Gamma period (B): {T_gamma_B_ms:.1f} ms")

    # Save combined baseline traces (both populations share same dt and T)
    if baseline_A.instruments.V_mean_E is None:
        raise RuntimeError("No instruments recorded for population A")

    t = np.arange(len(baseline_A.instruments.V_mean_E)) * pop_A.base.dt

    plot_baseline_traces(
        t,
        np.array(baseline_A.instruments.V_mean_E),
        np.array(baseline_A.instruments.g_i_mean_E),
        np.array(baseline_B.instruments.V_mean_E),
        np.array(baseline_B.instruments.g_i_mean_E),
        data_path / "baseline_traces.png",
    )

    # ========================================================================
    # PHASE 3: Create Spike Volley in A
    # ========================================================================
    print("\n[3/5] Creating spike volley in Population A...")

    # Target 20% of E neurons in A for pulse
    rng = np.random.default_rng(pop_A.base.seed or 0)
    target_E_A = rng.choice(
        pop_A.base.N_E,
        size=int(0.2 * pop_A.base.N_E),
        replace=False
    )

    # Deliver pulse at t=600ms
    if not config.pulse:
        raise ValueError("Pulse configuration is missing in config file")

    pulse_t = 500.0
    pulse_input_A = add_pulse_to_input(
        baseline_input_A,
        target_E_A,
        pulse_t,
        config.pulse.width_ms,
        config.pulse.amp,
        pop_A.base.dt,
        num_steps_A,
    )

    # Run A with pulse
    results_A_pulse = run_network(
        pop_A.base.model_copy(update={"external_input": pulse_input_A})
    )

    # Extract volley
    volley_times, volley_count = extract_spike_volley(
        results_A_pulse.spikes,
        target_E_A,
        pulse_t,
        window_ms=50.0,
    )

    print(f"   ✓ Pulse delivered at t={pulse_t} ms")
    print(f"   ✓ Volley size: {volley_count} spikes")

    # ========================================================================
    # PHASE 4: Phase Sweep - Deliver Volley to B at Different Phase Lags
    # ========================================================================
    print("\n[4/5] Running phase sweep...")

    phase_lags_ms = np.linspace(0, 80, 40)

    # Choose 50% of E neurons in B as targets
    target_E_B = rng.choice(
        pop_B.base.N_E,
        size=int(0.5 * pop_B.base.N_E),
        replace=False
    )

    if not config.coupling:
        raise ValueError("Coupling configuration is missing in config file")

    # Volley arrives at same absolute time in each condition
    volley_t = pulse_t + config.coupling.delay_AB

    # Prepare configs for parallel execution
    cfgs = [{
        "config": config,
        "pop_B": pop_B,
        "baseline_input_B": baseline_input_B,
        "volley_times": volley_times,  # Relative to pulse
        "phase_shift_ms": phase_lag,
        "target_E_B": target_E_B,
        "volley_t": volley_t,
    } for phase_lag in phase_lags_ms]

    results = parallel(inner, cfgs)

    gains = []
    for i, (_, gain, spikes_B) in enumerate(results):
        gains.append(gain)
        phase_ms = phase_lags_ms[i]
        print(f"   ✓ Phase {phase_ms:3.0f} ms → Gain: {gain:+.1f} spikes")

        # Save coupled raster for this phase condition
        plot_coupled_raster(
            results_A_pulse.spikes,
            spikes_B,
            pop_A.base.N_E,
            pop_B.base.N_E,
            pulse_t,
            data_path / f"rasters_phase{int(phase_ms)}.png",
            title=f"Phase Lag = {phase_ms} ms",
        )

    gains = np.array(gains)

    # ========================================================================
    # PHASE 5: Plot Transfer Curve
    # ========================================================================
    print("\n[5/5] Plotting phase-transfer curve...")

    plot_phase_transfer_curve(
        phase_lags_ms,
        gains,
        data_path / "phase_transfer_gain.png",
    )

    # Summary statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Volley size (A): {volley_count} spikes")
    print(f"Gamma period (B): {T_gamma_B_ms:.1f} ms")
    print(f"Max transfer gain: {gains.max():+.1f} spikes")
    print(f"Min transfer gain: {gains.min():+.1f} spikes")
    print(f"Modulation range: {gains.max() - gains.min():.1f} spikes")
    print(f"Modulation factor: {gains.max() / (gains.min() + 1e-9):.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()
