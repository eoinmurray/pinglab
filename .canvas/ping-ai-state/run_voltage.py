"""Rerun the Canvas PING network and retain every membrane voltage."""

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "tools" / "snnsim"))

import infer  # noqa: E402
import models as M  # noqa: E402

DT_MS = 0.25
DURATION_MS = 100.0
SEED = 7
N_E = 100
N_I = 25
INPUT_RATE_HZ = 60.0
CONDITION = os.environ.get("PING_CONDITION", "baseline")
if CONDITION not in {"baseline", "w-ei-zero", "input-ramp"}:
    raise ValueError(f"unknown PING_CONDITION: {CONDITION}")
RUN_DIR = ROOT / "conditions" / CONDITION
RUN_DIR.mkdir(parents=True, exist_ok=True)

infer._pin_run(DT_MS, DURATION_MS, seed=SEED)
M.N_IN = N_E
M.N_INH = N_I
device = infer._auto_device()
net = infer.build_net(
    "ping",
    w_in=(0.3, 0.06),
    w_in_initial_zero_fraction=0.0,
    recurrent_initial_zero_fraction=0.0,
    device=device,
    randomize_init=True,
    dales_law=True,
    hidden_sizes=[N_E],
    readout_mode="rate",
    signed_readout=False,
    readout_bias=False,
    n_inh_per_layer={1: N_I},
    train_leak=False,
    adaptive_threshold=False,
    ei_strength=0.5,
    ei_ratio=2.0,
)
with torch.no_grad():
    net.W_ff[0].copy_(torch.eye(N_E, device=net.W_ff[0].device) * 0.3)
    if CONDITION == "w-ei-zero":
        net.W_ei["1"].zero_()

n_steps = int(DURATION_MS / DT_MS)
generator = torch.Generator().manual_seed(SEED + 1)
if CONDITION == "input-ramp":
    input_rate_hz = np.linspace(20.0, 160.0, n_steps, dtype=np.float32)
else:
    input_rate_hz = np.full(n_steps, INPUT_RATE_HZ, dtype=np.float32)
input_probability = torch.from_numpy(input_rate_hz * DT_MS / 1000.0).view(n_steps, 1, 1)
input_spikes = (
    torch.rand(n_steps, 1, N_E, generator=generator)
    < input_probability
).float().to(device)

net.eval()
net.recording = True
with torch.no_grad():
    net(input_spikes=input_spikes)

record = net.spike_record
v_e = record["v_e_1"].numpy()
v_i = record["v_i_1"].numpy()
g_e = record["ge_e_1"].numpy()
g_i = record["gi_e_1"].numpy()
e_spikes = record["hid"].numpy()
i_spikes = record["inh"].numpy()

e_t, e_cell = np.nonzero(e_spikes)
i_t, i_cell = np.nonzero(i_spikes)
if CONDITION == "baseline":
    # Keep the matched-pair baseline anchored to the previously verified run.
    raster = np.load(ROOT / "run" / "rasters.npz")
    assert np.array_equal(e_t.astype(np.int32), raster["e_t"])
    assert np.array_equal(e_cell.astype(np.int32), raster["e_cell"])
    assert np.array_equal(i_t.astype(np.int32), raster["i_t"])
    assert np.array_equal(i_cell.astype(np.int32), raster["i_cell"])
elif CONDITION == "w-ei-zero":
    assert torch.count_nonzero(net.W_ei["1"]).item() == 0

np.savez_compressed(
    RUN_DIR / "rasters.npz",
    e_t=e_t.astype(np.int32),
    e_cell=e_cell.astype(np.int32),
    i_t=i_t.astype(np.int32),
    i_cell=i_cell.astype(np.int32),
    dt=np.float32(DT_MS),
    T=np.int32(n_steps),
    n_e=np.int32(N_E),
    n_i=np.int32(N_I),
)

np.savez_compressed(
    RUN_DIR / "recurrent-weights.npz",
    w_ei=net.W_ei["1"].detach().cpu().numpy(),
    w_ie=net.W_ie["1"].detach().cpu().numpy(),
)

np.savez_compressed(
    RUN_DIR / "voltage-traces.npz",
    v_e=v_e,
    v_i=v_i,
    g_e=g_e,
    g_i=g_i,
    input_spikes=input_spikes.detach().cpu().numpy()[:, 0, :].astype(np.uint8),
    input_rate_hz=input_rate_hz,
    dt_ms=np.float32(DT_MS),
    resting_mv=np.float32(M.E_L),
    threshold_mv=np.float32(M.V_th),
)

print(f"{CONDITION}: {len(e_t)} E spikes, {len(i_t)} I spikes -> {RUN_DIR}")
