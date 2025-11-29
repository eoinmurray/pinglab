
from pathlib import Path
from matplotlib import pyplot as plt
from pinglab.analysis.ei_crosscorr import ei_crosscorr
from pinglab.plots.styles import save_both, figsize

from local.inner import inner


def experiment_5(config, data_path: Path) -> None:
    cfgs = {
            "config": config,
            "g_ei": 1.4,
            "I_E": 1.5,
        }

    result = inner(cfgs)

    centers, hist = ei_crosscorr(result.spikes, N_E=config.base.N_E)

    def plot_fn():
        plt.figure(figsize=figsize)
        plt.plot(centers, hist / hist.max())
        plt.title("E-I cross-correlogram")
        plt.xlabel("I lag relative to E (ms)")
        plt.ylabel("count")

    save_both(data_path / "ei_crosscorr", plot_fn)