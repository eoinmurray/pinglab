"""Reconstruct and save the exact seed-7 recurrent matrices used by the run."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "tools" / "snnsim"))

import infer  # noqa: E402
import models as M  # noqa: E402

infer._pin_run(0.25, 100.0, seed=7)
M.N_IN = 100
M.N_INH = 25
device = infer._auto_device()
net = infer.build_net(
    "ping",
    w_in=(0.3, 0.06),
    w_in_initial_zero_fraction=0.0,
    recurrent_initial_zero_fraction=0.0,
    device=device,
    randomize_init=True,
    dales_law=True,
    hidden_sizes=[100],
    readout_mode="rate",
    signed_readout=False,
    readout_bias=False,
    n_inh_per_layer={1: 25},
    train_leak=False,
    adaptive_threshold=False,
    ei_strength=0.5,
    ei_ratio=2.0,
)
np.savez_compressed(
    ROOT / "run" / "recurrent-weights.npz",
    w_ei=net.W_ei["1"].detach().cpu().numpy(),
    w_ie=net.W_ie["1"].detach().cpu().numpy(),
)
