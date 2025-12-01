from pathlib import Path
import numpy as np

from pinglab.plots import save_raster
from pinglab.utils import slice_spikes
from pinglab.multiprocessing import parallel
from local.inner import inner
from local.model import LocalConfig

def experiment_1(config: LocalConfig, data_path: Path) -> None:
    cfgs = [
        {   
            "config": config,
            "I_E": config.experiment_1.I_E,
            "g_ei": value,
        }
        for value in np.linspace(
            config.experiment_1.linspace.start, 
            config.experiment_1.linspace.stop, 
            config.experiment_1.linspace.num
        )
    ]

    results = parallel(inner, cfgs, label="Experiment 1")

    for i, result in enumerate(results):
        sliced_spikes = slice_spikes(
            result.spikes,
            start_time=config.plotting.raster.start_time,
            stop_time=config.plotting.raster.stop_time,
        )

        save_raster(
            sliced_spikes,
            path=data_path / f"raster_g_ei_{i + 1}.png",
            label=f"g_ei={cfgs[i]['g_ei']:.2f}",
        )
