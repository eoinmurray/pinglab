
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import sys

from pinglab.inputs.tonic import tonic
from pinglab.lib.weights_builder import build_adjacency_matrices
from pinglab.run import run_network, build_model_from_config
from pinglab.types import NetworkResult
from pinglab.analysis import population_isi_cv
from pinglab.utils import slice_spikes
from pinglab.plots.raster import save_raster
from pinglab.plots.styles import save_both, figsize

# Add the experiment directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from .model import LocalConfig


def experiment_6(config: LocalConfig, data_path: Path) -> None:
    # Sweep g_ei, I_E, and noise together to transition from AI to oscillating regime
    # At low g_ei/I_E + high noise: noise-driven asynchronous (CV ~0.55)
    # At high g_ei/I_E + low noise: PING oscillations (CV < 0.2)
    g_ei_values = np.linspace(0.5, 2.0, 10)
    I_E_values = np.linspace(0.4, 2.0, 10)  # Increase drive to activate PING
    noise_values = np.linspace(4.0, 0.5, 10)  # Reduce noise for regular oscillations
    cv_values = []

    for i, (g_ei, I_E, noise) in enumerate(zip(g_ei_values, I_E_values, noise_values)):
        print(f"Running g_ei={g_ei:.2f}, I_E={I_E:.2f}, noise={noise:.2f} ({i+1}/10)")

        run_cfg = config.base
        if config.weights is None:
            raise ValueError("weights must be provided for adjacency-only runs.")
        matrices = build_adjacency_matrices(
            N_E=run_cfg.N_E,
            N_I=run_cfg.N_I,
            mean_ee=config.weights.mean_ee,
            mean_ei=float(g_ei),
            mean_ie=config.weights.mean_ie,
            mean_ii=config.weights.mean_ii,
            std_ee=config.weights.std_ee,
            std_ei=config.weights.std_ei,
            std_ie=config.weights.std_ie,
            std_ii=config.weights.std_ii,
            p_ee=config.weights.p_ee,
            p_ei=config.weights.p_ei,
            p_ie=config.weights.p_ie,
            p_ii=config.weights.p_ii,
            clamp_min=config.weights.clamp_min,
            seed=run_cfg.seed,
        )

        external_input = tonic(
            N_E=int(config.base.N_E),
            N_I=int(config.base.N_I),
            I_E=I_E,
            I_I=config.default_inputs.I_I,
            noise_std=noise,
            num_steps=int(np.ceil(config.base.T / config.base.dt)),
            seed=config.base.seed if config.base.seed is not None else 0,
        )

        model = build_model_from_config(run_cfg)
        result: NetworkResult = run_network(
            run_cfg,
            external_input=external_input,
            model=model,
            weights=matrices.W,
        )

        sliced_spikes = slice_spikes(
            result.spikes,
            start_time=config.plotting.raster.start_time,
            stop_time=config.plotting.raster.stop_time,
        )

        # Save raster for this g_ei value
        save_raster(
            sliced_spikes,
            data_path / f"experiment_6_raster_gei_{i:02d}.png",
            # external_input=external_input,
            # dt=config.base.dt,
        )

        E_ids = np.arange(config.base.N_E)
        I_ids = np.arange(config.base.N_E, config.base.N_E + config.base.N_I)

        cv_pop_E, cv_E = population_isi_cv(result.spikes, neuron_ids=E_ids)
        cv_pop_I, cv_I = population_isi_cv(result.spikes, neuron_ids=I_ids)
        
        cv_values.append(cv_pop_E)
        print(f"  cv_E: {cv_pop_E:.3f}")

    # Plot CV vs g_ei
    def plot_cv_vs_gei():
        plt.figure(figsize=(figsize[0], figsize[0]))
        plt.plot(g_ei_values, cv_values, "o-", linewidth=2, markersize=8)
        plt.xlabel("$g_{ei}$", fontsize=12)
        plt.ylabel("ISI CV", fontsize=12)
        plt.title("CV vs $g_{ei}$", fontsize=14)
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)

    save_both(data_path / "experiment_6_cv_vs_gei.png", plot_cv_vs_gei)

    # Plot CV vs I_E
    def plot_cv_vs_ie():
        plt.figure(figsize=(figsize[0], figsize[0]))
        plt.plot(I_E_values, cv_values, "o-", linewidth=2, markersize=8)
        plt.xlabel("$I_E$", fontsize=12)
        plt.ylabel("ISI CV", fontsize=12)
        plt.title("CV vs $I_E$", fontsize=14)
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)

    save_both(data_path / "experiment_6_cv_vs_ie.png", plot_cv_vs_ie)

    # Plot CV vs noise
    def plot_cv_vs_noise():
        plt.figure(figsize=(figsize[0], figsize[0]))
        plt.plot(noise_values, cv_values, "o-", linewidth=2, markersize=8)
        plt.xlabel("noise", fontsize=12)
        plt.ylabel("ISI CV", fontsize=12)
        plt.title("CV vs noise", fontsize=14)
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)

    save_both(data_path / "experiment_6_cv_vs_noise.png", plot_cv_vs_noise)

    print("\nSummary:")
    for g_ei, I_E, noise, cv in zip(g_ei_values, I_E_values, noise_values, cv_values):
        print(f"  g_ei={g_ei:.2f}, I_E={I_E:.2f}, noise={noise:.2f}: CV={cv:.3f}")
