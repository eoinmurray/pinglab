
from pathlib import Path
from matplotlib import pyplot as plt
from pinglab.analysis.crosscorr import crosscorr
from pinglab.plots.styles import save_both, figsize

from local.hotloop import hotloop
from local.model import LocalConfig


def experiment_5(config: LocalConfig, data_path: Path) -> None:
    cfg = {
            "config": config,
            "g_ei": config.experiment_5.g_ei,
            "I_E": config.experiment_5.I_E,
        }

    result = hotloop(cfg)

    centers, hist = crosscorr(result.spikes, N_E=config.base.N_E)

    def plot_fn():
        plt.figure(figsize=figsize)
        plt.plot(centers, hist / hist.max())
        plt.title("E-I cross-correlogram")
        plt.xlabel("I lag relative to E (ms)")
        plt.ylabel("count")

    save_both(data_path / "crosscorr", plot_fn)