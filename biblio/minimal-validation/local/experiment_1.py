from joblib import Parallel, delayed
from pathlib import Path
import numpy as np

from pinglab.plots import save_raster
from pinglab.utils import slice_spikes
from local.inner import inner

def experiment_1(config, data_path: Path) -> None:
    cfgs = [
        {
            "config": config,
            "I_E": 1.2,
            "g_ei": value,
        }
        for i, value in enumerate(np.linspace(1.2, 1.7, 10))
    ]

    results = Parallel(n_jobs=-1)(delayed(inner)(cfg) for cfg in cfgs)

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
