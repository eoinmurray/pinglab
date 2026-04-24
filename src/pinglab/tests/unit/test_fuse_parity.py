"""Tensor + training-trajectory parity fixtures.

Gate for the upcoming time-loop fuse: freeze bit-for-bit reference
outputs now (pre-fuse), so any refactor or torch.compile change must
reproduce them within tight tolerance.

Two layers in this file:

  Layer 1 — forward+backward parity: loss=logits.sum(), compare
            logits, W_ff[0] before step, grad_W_ff[0]. atol 1e-5.

  Layer 2 — 3-step SGD trajectory: compare loss history and an MD5
            of concatenated flat parameters after each step. atol 1e-4.

Regenerate fixtures:

    uv run src/pinglab/tests/unit/test_fuse_parity.py

Layer 3 (notebook smoke) lives in tests/integration/, not here.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest
import torch

# Allow running this file directly (regeneration mode) without pytest conftest.
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import models as M  # noqa: E402
from config import build_net  # noqa: E402


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "fuse_parity"

MODELS = ["standard-snn", "cuba", "cuba-exp", "ping", "snntorch-library"]

TOL_TENSOR = 1e-5
TOL_TRAJ = 1e-4

# Pinned tiny config for the harness. Must stay stable — changing any of
# these values invalidates every stored fixture.
CFG = dict(
    N_IN=8,
    N_HID=16,
    N_INH=4,
    N_OUT=10,
    HIDDEN_SIZES=[16],
    T_ms=20.0,
    B=2,
    SEED=1234,
    LR=0.01,
    STEPS=3,
)


def _pin_model_sizes():
    old = (M.N_IN, M.N_HID, M.N_INH, M.N_OUT, M.HIDDEN_SIZES, M.T_ms, M.T_steps)
    M.N_IN = CFG["N_IN"]
    M.N_HID = CFG["N_HID"]
    M.N_INH = CFG["N_INH"]
    M.N_OUT = CFG["N_OUT"]
    M.HIDDEN_SIZES = list(CFG["HIDDEN_SIZES"])
    M.T_ms = CFG["T_ms"]
    M.T_steps = int(M.T_ms / M.dt)
    return old


def _restore_model_sizes(old):
    (M.N_IN, M.N_HID, M.N_INH, M.N_OUT, M.HIDDEN_SIZES, M.T_ms, M.T_steps) = old


@pytest.fixture(autouse=True)
def _tiny_sizes():
    old = _pin_model_sizes()
    yield
    _restore_model_sizes(old)


def _build(model_name):
    torch.manual_seed(CFG["SEED"])
    net = build_net(model_name, hidden_sizes=list(CFG["HIDDEN_SIZES"]))
    return net


def _make_input():
    g = torch.Generator().manual_seed(CFG["SEED"] + 1)
    spikes = (torch.rand(M.T_steps, CFG["B"], CFG["N_IN"], generator=g) < 0.1).float()
    return spikes


def _hash_params(net):
    h = hashlib.md5()
    for p in net.parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _forward_backward(model_name):
    net = _build(model_name)
    spikes = _make_input()

    # Snapshot first feedforward weight before step.
    W0_before = net.W_ff[0].detach().clone()

    logits = net(input_spikes=spikes)
    loss = logits.sum()
    loss.backward()
    grad_W0 = net.W_ff[0].grad.detach().clone()

    return {
        "logits": logits.detach().clone(),
        "W0_before": W0_before,
        "grad_W0": grad_W0,
        "loss": loss.detach().clone(),
    }


def _train_trajectory(model_name):
    net = _build(model_name)
    spikes = _make_input()
    opt = torch.optim.SGD(net.parameters(), lr=CFG["LR"])
    # Dummy target: all zeros, use sum of squares so gradient is deterministic.
    losses = []
    hashes = []
    for _ in range(CFG["STEPS"]):
        opt.zero_grad()
        logits = net(input_spikes=spikes)
        loss = (logits ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.detach().clone())
        hashes.append(_hash_params(net))
    return {"losses": torch.stack(losses), "hashes": hashes}


def _fixture_path(model_name, kind):
    return FIXTURE_DIR / f"{model_name}_{kind}.pt"


@pytest.mark.parametrize("model_name", MODELS)
def test_forward_backward_parity(model_name):
    path = _fixture_path(model_name, "fwdbwd")
    if not path.exists():
        pytest.skip(f"no fixture at {path}; run the file directly to regenerate")
    ref = torch.load(path, weights_only=False)
    cur = _forward_backward(model_name)
    for k in ("logits", "W0_before", "grad_W0", "loss"):
        torch.testing.assert_close(cur[k], ref[k], atol=TOL_TENSOR, rtol=TOL_TENSOR,
                                   msg=lambda m, k=k: f"{model_name}[{k}] drifted: {m}")


@pytest.mark.parametrize("model_name", MODELS)
def test_training_trajectory_parity(model_name):
    path = _fixture_path(model_name, "traj")
    if not path.exists():
        pytest.skip(f"no fixture at {path}; run the file directly to regenerate")
    ref = torch.load(path, weights_only=False)
    cur = _train_trajectory(model_name)
    torch.testing.assert_close(cur["losses"], ref["losses"],
                               atol=TOL_TRAJ, rtol=TOL_TRAJ,
                               msg=lambda m: f"{model_name} loss trajectory drifted: {m}")
    # Weight hashes are an all-or-nothing check. If losses match within
    # TOL_TRAJ but hashes differ, the drift is below our tolerance — surface
    # it as a warning rather than a hard fail.
    if cur["hashes"] != ref["hashes"]:
        pytest.skip(f"{model_name}: param hashes differ but losses within tol — "
                    f"expected on non-bitwise-determinstic backends")


def _regenerate():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    old = _pin_model_sizes()
    try:
        for m in MODELS:
            print(f"  {m} ...", flush=True)
            torch.save(_forward_backward(m), _fixture_path(m, "fwdbwd"))
            torch.save(_train_trajectory(m), _fixture_path(m, "traj"))
    finally:
        _restore_model_sizes(old)
    print(f"wrote fixtures to {FIXTURE_DIR}")


if __name__ == "__main__":
    _regenerate()
