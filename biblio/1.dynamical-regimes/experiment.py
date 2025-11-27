
from joblib import Parallel, delayed
from pathlib import Path
import numpy as np
import shutil

from pinglab import run_network
from pinglab.plots import save_raster
from pinglab.inputs import generate_tonic_noise_input
from pinglab.utils import load_config, slice_spikes
from pinglab.types import NetworkResult


def inner(cfg):
    config = cfg["config"]
    save_path = cfg["save_path"]

    g_ei = cfg["g_ei"]
    I_E = cfg["I_E"]
    N_E = cfg["N_E"]

    external_input = generate_tonic_noise_input(
        N_E=int(N_E),
        N_I=int(N_E * 0.25),
        I_E=I_E,
        I_I=config.inputs.I_I,
        noise_std=config.inputs.noise,
        num_steps=int(np.ceil(config.base.T / config.base.dt)),
        seed=config.base.seed if config.base.seed is not None else 0,
        )

    # Build update dict - when N_E changes, N_I must also change to maintain ratio
    update_dict = {
        "external_input": external_input,
        "N_E": int(N_E),
        "N_I": int(N_E * 0.25),  # Match the ratio used in external_input generation
        "g_ei": g_ei,
    }

    result: NetworkResult = run_network(config.base.model_copy(update=update_dict))

    spikes = result.spikes

    sliced_spikes = slice_spikes(
        spikes,
        start_time=config.plotting.raster.start_time,
        stop_time=config.plotting.raster.stop_time,
    )

    save_raster(
        sliced_spikes, 
        path=save_path, 
        label=f"g_ei={g_ei:.2f}, I_E={I_E:.2f}, N_E={N_E}",
    )
    return result


def main() -> None:
    root = Path(__file__).parent
    data_path = root / "data"
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    config = load_config(root / "config.yaml")

    cfgs = [
        {
            "config": config,
            "save_path": data_path / f"raster_g_ei_{i+1}.png",
            "N_E": config.base.N_E,
            "I_E": 1.2,
            "g_ei": value,
        }
        for i, value in enumerate(np.linspace(1.2, 1.7, 10))
    ]

    Parallel(n_jobs=-1)(delayed(inner)(cfg) for cfg in cfgs)

    cfgs = [
        {
            "config": config,
            "save_path": data_path / f"raster_I_E_{i + 1}.png",
            "N_E": 800,
            "g_ei": 1.4,
            "I_E": value,
        }
        for i, value in enumerate(np.linspace(1.15, 1.25, 10))
    ]

    Parallel(n_jobs=-1)(delayed(inner)(cfg) for cfg in cfgs)

if __name__ == "__main__":
    main()
