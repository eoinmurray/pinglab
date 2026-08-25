"""Rebuild and export the exact seed-7 matrices used by run-ping-v1."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "tools" / "snnsim"))

import infer  # noqa: E402
import models as M  # noqa: E402

infer._pin_run(0.25, 1000.0, seed=7)
M.N_IN = 400
M.N_INH = 100
M.EXACT_K_INITIALIZATION = True
net = infer.build_net(
    "ping",
    w_in=(0.01, 0.001),
    w_ee=(4.3, 1.29),
    w_ei=(0.6, 0.18),
    w_ie=(3.0, 0.9),
    w_ii=(0.4, 0.12),
    w_in_initial_zero_fraction=0.95,
    recurrent_initial_zero_fraction=0.975,
    device=infer._auto_device(),
    randomize_init=True,
    dales_law=True,
    hidden_sizes=[400],
    readout_mode="rate",
    signed_readout=False,
    readout_bias=False,
    n_inh_per_layer={1: 100},
    train_leak=False,
    adaptive_threshold=False,
)
np.savez_compressed(
    ROOT / "run-ping-v1" / "recurrent-weights.npz",
    w_ee=net.W_ee["1"].detach().cpu().numpy(),
    w_ei=net.W_ei["1"].detach().cpu().numpy(),
    w_ie=net.W_ie["1"].detach().cpu().numpy(),
    w_ii=net.W_ii["1"].detach().cpu().numpy(),
)
