from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np

from pinglab.analysis import population_rate, rate_psd
from pinglab.plots.styles import save_both, figsize
from pinglab.utils import slice_spikes
from pinglab.multiprocessing import parallel
from local.hotloop import hotloop
from local.model import LocalConfig


def experiment_4(config: LocalConfig, data_path: Path) -> None:
    cfgs = [
        {
            "config": config,
            "g_ei": config.experiment_4.g_ei,
            "I_E": value,
        }
        for i, value in enumerate(np.linspace(
            config.experiment_4.linspace.start,
            config.experiment_4.linspace.stop,
            config.experiment_4.linspace.num
        ))
    ]

    results = parallel(hotloop, cfgs, label="Experiment 4")

    for i, result in enumerate(results):
        sliced_spikes = slice_spikes(
            result.spikes,
            start_time=config.plotting.raster.start_time,
            stop_time=config.plotting.raster.stop_time,
        )

        dt_ms = 1.0  # critical: choose dt_ms = 1.0 for PSD calculation
        _, rate_hz = population_rate(
            sliced_spikes,
            config.base.T,
            dt_ms,
            pop="E",
            N_E=int(config.base.N_E),
            N_I=int(config.base.N_I),
        )

        rate_smooth = gaussian_filter1d(rate_hz, sigma=2)  # 2 ms smoothing
        f, Pxx = rate_psd(rate_smooth, dt_ms)
        mask = (f >= 5) & (f <= 150)

        def plot_fn():
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

            ax1.plot(f[mask], Pxx[mask] / np.max(Pxx[mask]))
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel("Power Spectral Density")
            ax1.set_title(
                f"Power Spectral Density of Population Rate (E) for I_E = {cfgs[i]['I_E']:.2f}"
            )
            ax1.grid(True)

            times = sliced_spikes.times
            ids = sliced_spikes.ids
            types = getattr(sliced_spikes, "types", None)

            # 0 = E, 1 = I by convention
            mask_E = types == 0
            mask_I = types == 1
            ax2.scatter(times[mask_E], ids[mask_E], s=0.5, marker=".", label="E")
            ax2.scatter(
                times[mask_I],
                ids[mask_I],
                s=0.5,
                marker=".",
                alpha=0.7,
                label="I",
            )
            ax2.legend(loc="upper right", fontsize=8)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

        save_both(data_path / f"experiment_4_psd_I_E_{cfgs[i]['I_E']:.2f}", plot_fn)

