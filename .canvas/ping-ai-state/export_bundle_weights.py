"""Export exact initialized matrices from an authenticated SNNLang bundle."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "tools" / "snn"))

import infer  # noqa: E402
import models as M  # noqa: E402
from bundle import load_graph_bundle, translate_cobanet_v1  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--t-ms", type=float, default=1000.0)
    args = parser.parse_args()

    _, graph = load_graph_bundle(args.bundle)
    spec = translate_cobanet_v1(graph)
    infer._pin_run(spec.dt, args.t_ms, seed=args.seed)
    M.N_IN = spec.input_size
    M.N_INH = spec.hidden_size // 4
    M.EXACT_K_INITIALIZATION = spec.exact_k_initialization
    net = infer.build_net(
        "ping",
        w_in=spec.w_in,
        w_ee=spec.w_ee,
        w_ei=spec.w_ei,
        w_ie=spec.w_ie,
        w_ii=spec.w_ii,
        w_in_initial_zero_fraction=0.0,
        recurrent_initial_zero_fraction=spec.recurrent_initial_zero_fraction,
        device=infer._auto_device(),
        randomize_init=True,
        dales_law=True,
        hidden_sizes=[spec.hidden_size],
        readout_mode=spec.readout_mode,
        signed_readout=False,
        readout_bias=False,
        n_inh_per_layer=None,
        train_leak=False,
        adaptive_threshold=False,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.run_dir / "recurrent-weights.npz",
        w_ee=net.W_ee["1"].detach().cpu().numpy(),
        w_ei=net.W_ei["1"].detach().cpu().numpy(),
        w_ie=net.W_ie["1"].detach().cpu().numpy(),
        w_ii=net.W_ii["1"].detach().cpu().numpy(),
    )


if __name__ == "__main__":
    main()
