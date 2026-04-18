"""Per-model forward-pass smoke: shape + finiteness on tiny input.

Catches NaN blow-ups and shape regressions that the training e2e can mask
by retrying with different data.
"""
import pytest
import torch

import models as M
from config import build_net


@pytest.fixture(autouse=True)
def _tiny_model_sizes():
    old = (M.N_IN, M.N_HID, M.N_INH, M.N_OUT, M.HIDDEN_SIZES, M.T_ms, M.T_steps)
    M.N_IN = 8
    M.N_HID = 16
    M.N_INH = 4
    M.N_OUT = 10
    M.HIDDEN_SIZES = [16]
    M.T_ms = 20.0
    M.T_steps = int(M.T_ms / M.dt)
    yield
    (M.N_IN, M.N_HID, M.N_INH, M.N_OUT, M.HIDDEN_SIZES, M.T_ms, M.T_steps) = old


@pytest.mark.parametrize("model_name", ["snntorch-clone", "cuba", "ping"])
def test_forward_returns_finite_logits(model_name):
    net = build_net(model_name, hidden_sizes=[16])
    B = 2
    # Sparse random spike input, shape (T, B, N_IN)
    torch.manual_seed(0)
    spikes = (torch.rand(M.T_steps, B, M.N_IN) < 0.1).float()

    with torch.no_grad():
        out = net(input_spikes=spikes)

    assert out.shape == (B, M.N_OUT), f"{model_name} got {out.shape}"
    assert torch.isfinite(out).all(), f"{model_name} produced non-finite logits"


@pytest.mark.parametrize("model_name", ["snntorch-clone", "cuba", "ping"])
def test_forward_zero_input_is_finite(model_name):
    """Zero input shouldn't blow up (no NaNs from div-by-zero, etc.)."""
    net = build_net(model_name, hidden_sizes=[16])
    B = 2
    spikes = torch.zeros(M.T_steps, B, M.N_IN)
    with torch.no_grad():
        out = net(input_spikes=spikes)
    assert torch.isfinite(out).all()
