
import sys
from pathlib import Path
import shutil
from pinglab.inputs import tonic
from pinglab.plots.raster import save_raster
from pinglab.plots.styles import save_both, figsize
from pinglab.run import run_network
from pinglab.types import InstrumentsConfig, NetworkResult
from pinglab.analysis import population_rate
from pinglab.utils import slice_spikes
import yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

from lib.model import LocalConfig


def autocorr_normalized(x: np.ndarray) -> np.ndarray | None:
    if x.size < 2:
        return None
    x_centered = x - np.mean(x)
    denom = np.dot(x_centered, x_centered)
    if denom == 0:
        return None
    ac = np.correlate(x_centered, x_centered, mode="full")
    ac = ac[ac.size // 2 :]
    return ac / denom


def rhythmicity_metrics(
    rate_hz: np.ndarray,
    bin_ms: float,
    min_lag_ms: float = 5.0,
    max_lag_ms: float = 200.0,
) -> dict[str, float | None]:
    ac = autocorr_normalized(rate_hz)
    if ac is None:
        return {"rhythmicity_index": None, "rhythmicity_lag_ms": None}

    lags_ms = np.arange(ac.size) * bin_ms
    mask = (lags_ms >= min_lag_ms) & (lags_ms <= max_lag_ms)
    if not np.any(mask):
        return {"rhythmicity_index": None, "rhythmicity_lag_ms": None}

    peak_idx = np.argmax(ac[mask])
    peak_lag = float(lags_ms[mask][peak_idx])
    peak_val = float(ac[mask][peak_idx])
    return {"rhythmicity_index": peak_val, "rhythmicity_lag_ms": peak_lag}


def save_autocorr_plot(
    ac: np.ndarray,
    bin_ms: float,
    output_path: Path,
) -> None:
    lags_ms = np.arange(ac.size) * bin_ms

    def plot_fn():
        _, ax = plt.subplots(figsize=figsize)
        ax.plot(lags_ms, ac, lw=1.0)
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Autocorrelation (normalized)")
        ax.set_title("Population rate autocorrelation")
        ax.grid(True)
        ax.set_xlim((0.0, lags_ms[-1]))
        plt.tight_layout()

    save_both(output_path, plot_fn)


def main() -> None:
    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)
    config = LocalConfig.model_validate(data)
    outputs = set(config.sweep.outputs)

    # Read sweep from config
    param = config.sweep.param
    values = np.linspace(
        config.sweep.linspace.start,
        config.sweep.linspace.stop,
        config.sweep.linspace.num,
    )

    rhythmicity_by_value: list[tuple[float, float | None]] = []

    for value in values:
        config_run = config.model_copy(deep=True)
        if hasattr(config_run.default_inputs, param):
            config_run.default_inputs = config_run.default_inputs.model_copy(
                update={param: float(value)}
            )
        else:
            config_run.base = config_run.base.model_copy(update={param: float(value)})

        T = float(config_run.base.T)
        dt = float(config_run.base.dt)
        N_E = int(config_run.base.N_E)
        N_I = int(config_run.base.N_I)

        external_input = tonic(
            N_E=N_E,
            N_I=N_I,
            I_E=config_run.default_inputs.I_E,
            I_I=config_run.default_inputs.I_I,
            noise_std=config_run.default_inputs.noise,
            num_steps=int(np.ceil(T / dt)),
            seed=config_run.base.seed if config_run.base.seed is not None else 0,
        )

        # Enable conductance recording for all neurons
        instruments_config = InstrumentsConfig(
            variables=["g_e", "g_i"],
            all_neurons=True,
        )
        run_cfg = config.base.model_copy(update={"instruments": instruments_config})

        result: NetworkResult = run_network(run_cfg, external_input=external_input)

        if "raster" in outputs:
            sliced_spikes = slice_spikes(
                result.spikes,
                start_time=config.plotting.raster.start_time,
                stop_time=config.plotting.raster.stop_time,
            )

            save_raster(sliced_spikes, data_path / f"raster_{value:.2f}.png")

        if "metrics" in outputs:
            bin_ms = 1.0
            t_ms, rate_hz = population_rate(
                result.spikes,
                T_ms=T,
                dt_ms=bin_ms,
                pop="E",
                N_E=N_E,
                N_I=N_I,
            )
            window_mask = (t_ms >= config.plotting.raster.start_time) & (
                t_ms < config.plotting.raster.stop_time
            )
            rate_window = rate_hz[window_mask]

            metrics = rhythmicity_metrics(rate_window, bin_ms=bin_ms)
            metrics_path = data_path / f"rhythmicity_{param}_{value:.2f}.yaml"
            with metrics_path.open("w") as f:
                yaml.safe_dump(metrics, f)

            rhythmicity_by_value.append((float(value), metrics["rhythmicity_index"]))

            ac = autocorr_normalized(rate_window)
            if ac is not None:
                save_autocorr_plot(
                    ac=ac,
                    bin_ms=bin_ms,
                    output_path=data_path / f"autocorr_{param}_{value:.2f}.png",
                )

    if "metrics" in outputs and rhythmicity_by_value:
        xs = np.array([v for v, _ in rhythmicity_by_value], dtype=float)
        ys = np.array(
            [np.nan if r is None else float(r) for _, r in rhythmicity_by_value],
            dtype=float,
        )

        def plot_fn():
            _, ax = plt.subplots(figsize=figsize)
            ax.plot(xs, ys, marker="o", lw=1.5)
            ax.set_xlabel(param)
            ax.set_ylabel("Rhythmicity index (autocorr peak)")
            ax.set_title("Rhythmicity vs input current")
            ax.grid(True)
            plt.tight_layout()

        save_both(data_path / f"rhythmicity_vs_{param}.png", plot_fn)

if __name__ == "__main__":
    main()
