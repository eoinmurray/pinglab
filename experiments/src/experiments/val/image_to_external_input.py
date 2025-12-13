
from pathlib import Path
import sys
import numpy as np

# Add the experiment directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from .model import LocalConfig


def image_to_external_input(image: np.ndarray, config: LocalConfig) -> np.ndarray:
    flat = image.flatten() / np.max(image)

    I_min = 0.8
    I_max = 1.6

    I_E = I_min + flat * (I_max - I_min)

    num_steps = int(config.base.T / config.base.dt)
    N = config.base.N_E + config.base.N_I

    external_input = np.zeros((num_steps, N), dtype=np.float32)
    external_input[:, :config.base.N_E] = I_E[None, :]

    if config.default_inputs.noise > 0:
        external_input[:, :config.base.N_E] += np.random.normal(
            loc=0.0,
            scale=config.default_inputs.noise,
            size=(num_steps, config.base.N_E),
        ).astype(np.float32)

    return external_input