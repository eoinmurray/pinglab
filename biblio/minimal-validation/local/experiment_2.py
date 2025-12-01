from pathlib import Path
import numpy as np

from pinglab.plots import save_raster
from pinglab.utils import slice_spikes
from pinglab.multiprocessing import parallel
from local.inner import inner
from local.model import LocalConfig

def experiment_2(config: LocalConfig, data_path: Path) -> None:
    cfgs = [
        {
            "config": config,
            "g_ei": config.experiment_2.g_ei,
            "I_E": value,
        }
        for value in np.linspace(
            config.experiment_2.linspace.start, 
            config.experiment_2.linspace.stop, 
            config.experiment_2.linspace.num
        )
    ]

    results = parallel(inner, cfgs, label="Experiment 2")

    for i, result in enumerate(results):
        sliced_spikes = slice_spikes(
            result.spikes,
            start_time=config.plotting.raster.start_time,
            stop_time=config.plotting.raster.stop_time,
        )

        save_raster(
            sliced_spikes,
            path=data_path / f"raster_I_E_{i + 1}.png",
            label=f"I_E={cfgs[i]['I_E']:.2f}",
        )
