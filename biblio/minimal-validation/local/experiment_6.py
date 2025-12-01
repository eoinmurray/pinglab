
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from pinglab.analysis import population_isi_cv
from pinglab.plots.raster import save_raster
from pinglab.plots.styles import save_both, figsize
from pinglab.utils import slice_spikes
from pinglab.multiprocessing import parallel
from local.inner import inner
from local.model import LocalConfig


def experiment_6(config: LocalConfig, data_path: Path) -> None:
    cfgs = [
        {
            "config": config,
            "g_ei": config.experiment_6.g_ei,
            "I_E": value,
        }
        for i, value in enumerate(np.linspace(
            config.experiment_6.linspace.start,
            config.experiment_6.linspace.stop,
            config.experiment_6.linspace.num,
        ))
    ]

    results = parallel(inner, cfgs, label="Experiment 6")

    cv_E_list = []
    cv_I_list = []
    param_values = []

    for i, result in enumerate(results):
        cfg = cfgs[i]
        I_E = cfg["I_E"]

        sliced_spikes = slice_spikes(
            result.spikes,
            start_time=config.plotting.raster.start_time,
            stop_time=config.plotting.raster.stop_time,
        )

        cv_E, cv_I = population_isi_cv(sliced_spikes, N_E=config.base.N_E, N_I=config.base.N_I)
        param_values.append(I_E)
        cv_E_list.append(cv_E)
        cv_I_list.append(cv_I)

        save_raster(
            sliced_spikes,
            data_path / f"raster_population_isi_cv_I_E_{I_E:.2f}.png",
            label=f"I_E={cfgs[-1]['I_E']}",
        )

    def plot_fn():
        params = np.array(param_values)

        cvE = np.array([np.nan if v is None else v for v in cv_E_list])
        cvI = np.array([np.nan if v is None else v for v in cv_I_list])

        plt.figure(figsize=(figsize))
        plt.plot(params, cvE, "-o", label="E population", alpha=0.9)
        plt.plot(params, cvI, "-o", label="I population", alpha=0.9)

        plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)  # Poisson baseline

        plt.xlabel("I_E")
        plt.ylabel("ISI CV")
        plt.title("Irregularity → Synchrony transition via ISI CV")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

    save_both(data_path / "population_isi_cv", plot_fn)

    