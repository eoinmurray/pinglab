from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pinglab.plots.styles import save_both, figsize
from pinglab.analysis import mean_firing_rates
from pinglab.types import NetworkResult, Spikes
from pinglab.multiprocessing import parallel
from local.inner import inner

def experiment_3(config, data_path: Path) -> None:
    cfgs_large = [
        {
            "config": config,
            "g_ei": 1.4,
            "I_E": value,
        }
        for i, value in enumerate(np.linspace(0.0, 100.0, 40))
    ]

    cfgs_small = [
        {
            "config": config,
            "g_ei": 1.4,
            "I_E": value,
        }
        for i, value in enumerate(np.linspace(0.0, 2.0, 40))
    ]

    results_large = parallel(inner, cfgs_large, label="Experiment 3 Large")
    results_small = parallel(inner, cfgs_small, label="Experiment 3 Small")

    def map_mean_firing_rates(result: NetworkResult) -> tuple[float, float]:
        spikes: Spikes = result.spikes

        E_firing_rate, I_firing_rate = mean_firing_rates(
            spikes,
            N_E=int(config.base.N_E),
            N_I=int(config.base.N_I),
        )

        return E_firing_rate, I_firing_rate

    rates_large = [map_mean_firing_rates(result) for result in results_large]
    rates_small = [map_mean_firing_rates(result) for result in results_small]

    def plot_fn_large():
        I_Es = [cfg["I_E"] for cfg in cfgs_large]
        E_rates = [rate[0] for rate in rates_large]
        I_rates = [rate[1] for rate in rates_large]

        _, ax_ras = plt.subplots(1, 1, figsize=figsize)
        ax_ras.plot(I_Es, E_rates, label="E population", marker="o")
        ax_ras.plot(I_Es, I_rates, label="I population", marker="o")
        ax_ras.set_xlabel("I_E")
        ax_ras.set_ylabel("Firing Rate (Hz)")
        ax_ras.set_title("Firing Rates vs I_E")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    def plot_fn_small():
        I_Es = [cfg["I_E"] for cfg in cfgs_small]
        E_rates = [rate[0] for rate in rates_small]
        I_rates = [rate[1] for rate in rates_small]

        _, ax_ras = plt.subplots(1, 1, figsize=figsize)
        ax_ras.plot(I_Es, E_rates, label="E population", marker="o")
        ax_ras.plot(I_Es, I_rates, label="I population", marker="o")
        ax_ras.set_xlabel("I_E")
        ax_ras.set_ylabel("Firing Rate (Hz)")
        ax_ras.set_title("Firing Rates vs I_E (Small Range)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    save_both(data_path / "firing_rates_vs_I_E_large", plot_fn_large)
    save_both(data_path / "firing_rates_vs_I_E_small", plot_fn_small)
