
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import shutil
import sys
from pinglab.inputs.tonic import tonic
from pinglab.run.run_network import run_network
from pinglab.types import NetworkResult
from pinglab.types import NetworkResult
import yaml

from pinglab.analysis import population_isi_cv
from pinglab.plots.raster import save_raster

# Add the experiment directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from local.model import LocalConfig


def main() -> None:
    root = Path(__file__).parent
    data_path = root / "data"
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)
    config = LocalConfig.model_validate(data)

    external_input = tonic(
        N_E=int(config.base.N_E),
        N_I=int(config.base.N_I),
        I_E=config.default_inputs.I_E,
        I_I=config.default_inputs.I_I,
        noise_std=config.default_inputs.noise,
        num_steps=int(np.ceil(config.base.T / config.base.dt)),
        seed=config.base.seed if config.base.seed is not None else 0,
    )

    result: NetworkResult = run_network(config.base, external_input=external_input) 

    save_raster(
        result.spikes,
        data_path / f"raster.png",
        external_input=external_input,
        dt=config.base.dt,
    )

    cv_E, cv_I = population_isi_cv(result.spikes, N_E=config.base.N_E, N_I=config.base.N_I, min_spikes=2)

    print(f"cv_E: {cv_E:.3f}")

if __name__ == "__main__":
    main()
