import argparse
import time

import numpy as np

from pinglab.inputs import tonic
from pinglab.lib.weights_builder import build_adjacency_matrices
from pinglab.run import build_model_from_config, run_network
from pinglab.types import NetworkConfig


def make_config(N_E: int, N_I: int, T: float, dt: float, seed: int) -> NetworkConfig:
    return NetworkConfig(
        dt=dt,
        T=T,
        N_E=N_E,
        N_I=N_I,
        seed=seed,
        V_init=-65.0,
        E_L=-65.0,
        E_e=0.0,
        E_i=-80.0,
        C_m_E=1.0,
        g_L_E=0.1,
        C_m_I=1.0,
        g_L_I=0.1,
        V_th=-50.0,
        V_reset=-65.0,
        tau_ampa=5.0,
        tau_gaba=10.0,
        t_ref_E=2.0,
        t_ref_I=1.0,
        delay_ei=1.0,
        delay_ie=1.0,
        delay_ee=1.0,
        delay_ii=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark run_network backends.")
    parser.add_argument("--backend", choices=["dense", "event"], default="dense")
    parser.add_argument("--n-e", type=int, default=800)
    parser.add_argument("--n-i", type=int, default=200)
    parser.add_argument("--T", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = make_config(args.n_e, args.n_i, args.T, args.dt, args.seed)
    num_steps = int(np.ceil(config.T / config.dt))
    external_input = tonic(
        N_E=config.N_E,
        N_I=config.N_I,
        I_E=3.0,
        I_I=3.0,
        noise_std=0.5,
        num_steps=num_steps,
        seed=args.seed,
    )
    weights = build_adjacency_matrices(
        N_E=config.N_E,
        N_I=config.N_I,
        mean_ee=0.02,
        mean_ei=0.02,
        mean_ie=0.02,
        mean_ii=0.02,
        std_ee=0.0,
        std_ei=0.0,
        std_ie=0.0,
        std_ii=0.0,
        p_ee=0.02,
        p_ei=0.18,
        p_ie=0.04,
        p_ii=0.06,
        clamp_min=0.0,
        seed=args.seed,
    )

    model = build_model_from_config(config)

    start = time.perf_counter()
    result = run_network(
        config,
        external_input,
        model=model,
        weights=weights.W,
        connectivity_backend=args.backend,
    )
    elapsed = time.perf_counter() - start
    steps = int(np.ceil(config.T / config.dt))
    print(f"backend={args.backend} elapsed={elapsed:.3f}s steps={steps} step/s={steps/elapsed:.1f}")
    print(f"spikes={len(result.spikes.times)}")


if __name__ == "__main__":
    main()
