import numpy as np
from pinglab.run import run_network
from sklearn.linear_model import Ridge

from local.generate_image import generate_image
from local.build_feedforward_current import build_feedforward_current

def rates_from_spikes(spikes, N_E, T_window):
    times, ids = spikes.times, spikes.ids
    t0, t1 = T_window
    mask = (times >= t0) & (times < t1) & (ids < N_E)
    ids_E = ids[mask]
    counts = np.bincount(ids_E, minlength=N_E)
    dur = (t1 - t0) / 1000.0
    return counts / dur

def build_baseline_B(config, num_steps, N):
    ext = np.zeros((num_steps, N), dtype=np.float32)
    ext[:, :config.base.N_E] += config.default_inputs.I_E
    ext[:, config.base.N_E:] += config.default_inputs.I_I
    if config.default_inputs.noise > 0:
        ext += np.random.normal(
            0.0, config.default_inputs.noise, size=(num_steps, N)
        ).astype(np.float32)
    return ext

def collect_dataset(config, num_images: int, phase_ms: float = 0.0):
    X, Y = [], []
    num_steps = int(config.base.T / config.base.dt)
    N = config.base.N_E + config.base.N_I

    for _ in range(num_images):
        img = generate_image()                        # (side, side)
        flat = img.flatten().astype(np.float32)
        flat /= flat.max() + 1e-8                    # target pattern

        # A: image drive (your existing mapping)
        external_input_A = np.zeros((num_steps, N), dtype=np.float32)
        I_min, I_max = 0.8, 1.6
        I_E = I_min + flat * (I_max - I_min)
        external_input_A[:, :config.base.N_E] = I_E[None, :]

        result_A = run_network(config.base, external_input=external_input_A)

        # B: baseline + feedforward from A
        ff = build_feedforward_current(
            result_A.spikes,
            N_E=config.base.N_E,
            N_I=config.base.N_I,
            T=config.base.T,
            dt=config.base.dt,
            phase_ms=phase_ms,
        )
        ext_B = build_baseline_B(config, num_steps, N) + ff
        result_B = run_network(config.base, external_input=ext_B)

        rB = rates_from_spikes(result_B.spikes, config.base.N_E, (500, 1000))

        X.append(rB)
        Y.append(flat)

    return np.stack(X), np.stack(Y)
