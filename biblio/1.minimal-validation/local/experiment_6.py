
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from pinglab.analysis import population_isi_cv
from pinglab.plots.raster import save_raster
from pinglab.plots.styles import save_both, figsize
from pinglab.multiprocessing import parallel
from pinglab.utils import slice_spikes
from local.hotloop import hotloop
from local.model import LocalConfig


def experiment_6(config: LocalConfig, data_path: Path) -> None:
    if not config.experiment_6.linspace:
        raise RuntimeError("Experiment 6 is disabled in the configuration.")

    # config = config.model_copy(update={
    #     "default_inputs": config.default_inputs.model_copy(update={
    #         "noise": 1.5
    #     })
    # })

    cfgs = [
        {   
            "config": config,
            "I_E": config.experiment_6.I_E,
            "g_ei": value,
        }
        for value in np.linspace(
            config.experiment_6.linspace.start, 
            config.experiment_6.linspace.stop, 
            config.experiment_6.linspace.num
        )
    ]

    results = parallel(hotloop, cfgs, label="Experiment 6")

    cv_E_list = []
    cv_I_list = []
    param_values = []

    for i, result in enumerate(results):
        cfg = cfgs[i]

        if not config.plotting:
            raise RuntimeError("Plotting must be enabled for Experiment 6.")

        sliced_spikes = slice_spikes(
            result.spikes,
            start_time=config.plotting.raster.start_time,
            stop_time=config.plotting.raster.stop_time,
        )

        save_raster(
            sliced_spikes,
            data_path / f"experiment_6_raster_g_{cfg['g_ei']:.2f}.png",
            label=f"g_ei={cfg['g_ei']:.2f}",
        )

        cv_E, cv_I = population_isi_cv(sliced_spikes, N_E=config.base.N_E, N_I=config.base.N_I, min_spikes=2)
        param_values.append(cfg["g_ei"])
        cv_E_list.append(cv_E)
        cv_I_list.append(cv_I)

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

    save_both(data_path / "experiment_6_population_isi_cv", plot_fn)