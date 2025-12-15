
import sys
from pathlib import Path
import shutil
import yaml
import numpy as np

from pinglab.inputs import tonic
from pinglab.plots.raster import save_raster
from pinglab.run.run_network import run_network
from pinglab.types import NetworkResult

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

from lib.model import LocalConfig


def main() -> None:
    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)
    config = LocalConfig.model_validate(data)

    external_input_A = tonic(
        N_E=int(config.base.N_E),
        N_I=int(config.base.N_I),
        I_E=config.default_inputs.I_E,
        I_I=config.default_inputs.I_I,
        noise_std=config.default_inputs.noise,
        num_steps=int(np.ceil(config.base.T / config.base.dt)),
        seed=config.base.seed if config.base.seed is not None else 0,
    )

    result_A: NetworkResult = run_network(config.base, external_input=external_input_A)

    save_raster(
        result_A.spikes,
        path=data_path / "raster_A.png",
        external_input=external_input_A,
        dt=config.base.dt,
    )


if __name__ == "__main__":
    main()
