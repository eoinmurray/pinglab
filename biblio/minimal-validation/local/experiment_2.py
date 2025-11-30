from pathlib import Path
import numpy as np

from pinglab.plots import save_raster
from pinglab.utils import slice_spikes
from pinglab.multiprocessing import parallel
from local.inner import inner


def experiment_2(config, data_path: Path) -> None:
    cfgs = [
        {
            "config": config,
            "g_ei": 1.4,
            "I_E": value,
        }
        for i, value in enumerate(np.linspace(1.15, 1.25, 10))
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
