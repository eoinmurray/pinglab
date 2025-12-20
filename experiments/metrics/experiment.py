
import sys
from pathlib import Path
import shutil
from pinglab.analysis import base_metrics
from pinglab.inputs import tonic
from pinglab.plots.raster import save_raster
from pinglab.run import run_network
from pinglab.types import InstrumentsConfig, NetworkResult
from pinglab.utils import slice_spikes
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

from lib.model import LocalConfig


def run_regime(
        config,
        data_path: Path,
        label: str = "regime",
    ):
        T = float(config.base.T)
        dt = float(config.base.dt)
        N_E = int(config.base.N_E)
        N_I = int(config.base.N_I)

        external_input = tonic(
            N_E=N_E,
            N_I=N_I,
            I_E=config.default_inputs.I_E,
            I_I=config.default_inputs.I_I,
            noise_std=config.default_inputs.noise,
            num_steps=int(np.ceil(T / dt)),
            seed=config.base.seed if config.base.seed is not None else 0,
        )

        # Enable conductance recording for all neurons
        instruments_config = InstrumentsConfig(
            variables=["g_e", "g_i"],
            all_neurons=True,
        )
        run_cfg = config.base.model_copy(update={"instruments": instruments_config})

        result: NetworkResult = run_network(run_cfg, external_input=external_input)

        sliced_spikes = slice_spikes(
            result.spikes,
            start_time=config.plotting.raster.start_time,
            stop_time=config.plotting.raster.stop_time,
        )

        save_raster(sliced_spikes, data_path / f"raster_{label}.png")

        base_metrics(
            config=config,
            run_result=result,
            data_path=data_path,
            label=label,
        )


def main() -> None:
    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)
    config = LocalConfig.model_validate(data)

    # Read sweep from config
    param = config.sweep.param
    values = np.linspace(
        config.sweep.linspace.start,
        config.sweep.linspace.stop,
        config.sweep.linspace.num,
    )

    for value in values:
        config_run = config.model_copy(deep=True)
        config_run.base = config_run.base.model_copy(update={param: float(value)})

        run_regime(
            config=config_run,
            data_path=data_path,
            label=f"{param}_{value:.2f}",
        )

if __name__ == "__main__":
    main()
