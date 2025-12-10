
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import shutil
import sys
from pinglab.inputs.tonic import tonic
from pinglab.run.run_network import run_network
from pinglab.types import NetworkResult
import yaml

from pinglab.analysis import population_isi_cv
from pinglab.utils import slice_spikes
from pinglab.plots.raster import save_raster
from pinglab.plots.styles import save_both, figsize

# Add the experiment directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from local.model import LocalConfig


def main() -> None:
    root = Path(__file__).parent
    data_path = root / "data"
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)
    config = LocalConfig.model_validate(data)

    # Sweep g_ei, I_E, and noise together to transition from AI to oscillating regime
    # At low g_ei/I_E + high noise: noise-driven asynchronous (CV ~0.55)
    # At high g_ei/I_E + low noise: PING oscillations (CV < 0.2)
    g_ei_values = np.linspace(0.5, 4.0, 10)
    I_E_values = np.linspace(0.4, 3.5, 10)  # Increase drive to activate PING
    noise_values = np.linspace(4.0, 0.5, 10)  # Reduce noise for regular oscillations
    cv_values = []

    for i, (g_ei, I_E, noise) in enumerate(zip(g_ei_values, I_E_values, noise_values)):
        print(f"Running g_ei={g_ei:.2f}, I_E={I_E:.2f}, noise={noise:.2f} ({i+1}/10)")

        # Update config with new g_ei
        run_cfg = config.base.model_copy(update={"g_ei": g_ei})

        external_input = tonic(
            N_E=int(config.base.N_E),
            N_I=int(config.base.N_I),
            I_E=I_E,
            I_I=config.default_inputs.I_I,
            noise_std=noise,
            num_steps=int(np.ceil(config.base.T / config.base.dt)),
            seed=config.base.seed if config.base.seed is not None else 0,
        )

        result: NetworkResult = run_network(run_cfg, external_input=external_input)

        sliced_spikes = slice_spikes(
            result.spikes,
            start_time=config.plotting.raster.start_time,
            stop_time=config.plotting.raster.stop_time,
        )

        # Save raster for this g_ei value
        save_raster(
            sliced_spikes,
            data_path / f"raster_gei_{i:02d}.png",
            # external_input=external_input,
            # dt=config.base.dt,
        )

        E_ids = np.arange(config.base.N_E)
        I_ids = np.arange(config.base.N_E, config.base.N_E + config.base.N_I)

        cv_pop_E, cv_E = population_isi_cv(result.spikes, neuron_ids=E_ids)
        cv_pop_I, cv_I = population_isi_cv(result.spikes, neuron_ids=I_ids)
        
        cv_values.append(cv_pop_E)
        print(f"  cv_E: {cv_pop_E:.3f}")

    # Plot CV vs g_ei curve
    def plot_cv_curve():
        plt.figure(figsize=figsize)
        plt.plot(g_ei_values, cv_values, "o-", linewidth=2, markersize=8)
        plt.xlabel("$g_{ei}$ (E→I synaptic weight)", fontsize=12)
        plt.ylabel("ISI CV", fontsize=12)
        plt.title("Transition from AI to Oscillating Regime", fontsize=14)
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)

    save_both(data_path / "cv_vs_gei.png", plot_cv_curve)

    print("\nSummary:")
    for g_ei, I_E, noise, cv in zip(g_ei_values, I_E_values, noise_values, cv_values):
        print(f"  g_ei={g_ei:.2f}, I_E={I_E:.2f}, noise={noise:.2f}: CV={cv:.3f}")


if __name__ == "__main__":
    main()
