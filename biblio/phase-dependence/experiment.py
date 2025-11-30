
from pathlib import Path
import shutil
import numpy as np


from pinglab.plots import save_raster, save_instrument_traces
from pinglab.inputs import tonic
from pinglab.utils import load_config
from pinglab.multiprocessing import parallel
from pinglab import run_network

from local import (
    estimate_trough_peak_and_period,
    add_pulse_to_input,
    compute_spike_delta,
    plot_phase_gain_curve,
)


def inner(cfg: dict) -> tuple[float, float]:
    config = cfg["config"]
    baseline_input = cfg["baseline_input"]
    peak_t_ms = cfg["peak_t_ms"]
    offset_ms = cfg["offset_ms"]
    target_E = cfg["target_E"]

    pulse_t = peak_t_ms + offset_ms
    pulse_input = add_pulse_to_input(  # copies
        baseline_input,
        target_E,
        pulse_t,
        config.pulse.width_ms,
        config.pulse.amp,
        config.base.dt,
        num_steps=int(np.ceil(config.base.T / config.base.dt)),
    )
    pulse_results = run_network(
        config.base.model_copy(update={"external_input": pulse_input})
    )
    delta = compute_spike_delta(
        pulse_results.spikes,
        target_E,
        pulse_t,
        config.pulse.pre_window_ms,
        config.pulse.post_window_ms,
    )
    return offset_ms, delta


def main() -> None:
    root = Path(__file__).parent
    data_path = root / "data"
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    config = load_config(root / "config.yaml")

    num_steps = int(np.ceil(config.base.T / config.base.dt))

    baseline_input = tonic(
        N_E=config.base.N_E,
        N_I=config.base.N_I,
        I_E=config.inputs.I_E,
        I_I=config.inputs.I_I,
        noise_std=config.inputs.noise,
        num_steps=num_steps,
        seed=config.base.seed or 0,
    )

    print("Run baseline")
    baseline_results = run_network(config.base.model_copy(update={"external_input": baseline_input}))

    if not baseline_results.instruments:
        raise RuntimeError("No baseline_results.instruments recorded in baseline")

    lfp_proxy = np.array(baseline_results.instruments.g_i_mean_E)
    peak_t_ms, T_gamma_ms = estimate_trough_peak_and_period(
        lfp_proxy, dt=config.base.dt
    )

    save_instrument_traces(baseline_results.instruments, data_path)
    save_raster(
        baseline_results.spikes,
        path=data_path / "raster_baseline.png",
        label="baseline",
        dt=config.base.dt,
        external_input=baseline_input,
    )
    
    rng = np.random.default_rng(config.base.seed or 0)
    target_E = rng.choice(config.base.N_E, size=max(1, int(0.1 * config.base.N_E)), replace=False)

    # Pulse sweep across gamma phases
    phase_offsets_ms = np.linspace(0, 180, 100)
    phase_times: list[float] = []
    deltas: list[float] = []

    cfgs = [{
        "config": config,
        "peak_t_ms": peak_t_ms,
        "offset_ms": offset_ms,
        "baseline_input": baseline_input,
        "target_E": target_E,
    } for idx, offset_ms in enumerate(phase_offsets_ms)]

    results = parallel(inner, cfgs)

    for offset_ms, delta in results:
        phase_times.append(offset_ms)
        deltas.append(delta)

    plot_phase_gain_curve(
        phase_times,
        deltas,
        T_gamma_ms,
        lfp_proxy,
        peak_t_ms,
        config.base.dt,
        data_path / "phase_gain_curve.png",
    )


if __name__ == "__main__":
    main()
