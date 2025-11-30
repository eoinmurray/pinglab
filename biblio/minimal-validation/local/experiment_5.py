
from pathlib import Path
from matplotlib import pyplot as plt
from pinglab.analysis.crosscorr import crosscorr
from pinglab.plots.styles import save_both, figsize

from local.inner import inner


def experiment_5(config, data_path: Path) -> None:
    cfg = {
            "config": config,
            "g_ei": 1.4,
            "I_E": 1.5,
        }

    result = inner(cfg)

    centers, hist = crosscorr(result.spikes, N_E=config.base.N_E)

    def plot_fn():
        plt.figure(figsize=figsize)
        plt.plot(centers, hist / hist.max())
        plt.title("E-I cross-correlogram")
        plt.xlabel("I lag relative to E (ms)")
        plt.ylabel("count")

    save_both(data_path / "crosscorr", plot_fn)