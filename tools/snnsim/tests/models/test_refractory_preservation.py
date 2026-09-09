"""Opt-in, read-only comparison against the pre-adoption simulator source.

Run with PINGLAB_REFRACTORY_BANK_CHECK=1. This is a regression check on saved
weights, not a new experimental measurement or a repeat of model training.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import config as current_config
import models as current_model
import numpy as np
import pytest
import torch
from encoders import encode_images_poisson

REPO = Path(__file__).resolve().parents[4]
BASELINE = "255ab3b6e11fe4e92cd5f96c1634a2a58150bb05"
BANK = "exp022-r001-compute"
DIGEST = "sha256:9e3c93df9541809d1d019fe5290afbf7dff7d07ec14b07160fabe7ad79c9a0a8"
RANDOMIZE_INITIAL_STATE = os.environ.get("PINGLAB_REFRACTORY_RANDOMIZED_CHECK") == "1"
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("PINGLAB_REFRACTORY_BANK_CHECK") != "1",
        reason="explicit opt-in required for the retained-bank regression",
    ),
]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def preservation_context(tmp_path_factory):
    from tools.pingstore.stages import source_run

    bank = source_run(
        REPO / ".pingstore", BANK, stage="compute", experiment="exp022",
        reference={"run_id": BANK, "payload_digest": DIGEST},
    )
    root = tmp_path_factory.mktemp("refractory-baseline")
    source_hashes = {}
    for filename in ("models.py", "config.py"):
        content = subprocess.check_output(
            ["git", "show", f"{BASELINE}:tools/snnsim/{filename}"], cwd=REPO,
        )
        (root / filename).write_bytes(content)
        source_hashes[filename] = hashlib.sha256(content).hexdigest()
    old_model = _module("refractory_baseline_models", root / "models.py")
    saved = sys.modules["models"]
    try:
        sys.modules["models"] = old_model
        old_config = _module("refractory_baseline_config", root / "config.py")
    finally:
        sys.modules["models"] = saved
    cells = sorted(
        path.name for path in bank.export.iterdir()
        if path.is_dir() and json.loads((path / "config.json").read_text())["dt"] == 0.1
    )
    assert len(cells) == 90
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    print(json.dumps({"baseline": BASELINE, "source_hashes": source_hashes,
                      "randomize_initial_state": RANDOMIZE_INITIAL_STATE,
                      "source": bank.reference, "reuse_candidates": len(cells)}), flush=True)
    yield bank, cells, old_model, old_config
    bank.check_unchanged()
    torch.set_num_threads(old_threads)


def _configure(module, cfg, duration_ms=200.0):
    for key, value in {
        "dt": 0.1, "T_ms": duration_ms, "T_steps": round(duration_ms / 0.1),
        "N_IN": cfg["n_in"], "N_OUT": cfg["n_out"], "N_INH": cfg["n_inh"],
        "tau_ampa": cfg["tau_ampa_ms"], "tau_gaba": cfg["tau_gaba_ms"],
        "SURROGATE_SLOPE": cfg["surrogate_slope"],
        "V_GRAD_DAMPEN": cfg["v_grad_dampen"],
    }.items():
        setattr(module, key, value)


def _build(builder, module, cfg, explicit):
    _configure(module, cfg)
    kwargs = {
        key: cfg[key] for key in inspect.signature(builder.build_net).parameters
        if key in cfg and cfg[key] is not None
    }
    kwargs.pop("model_name", None)
    kwargs["device"] = torch.device("cpu")
    kwargs["randomize_init"] = True
    kwargs["n_inh_per_layer"] = {1: cfg["n_inh"]}
    if isinstance(kwargs.get("readout_w_init"), dict):
        kwargs["readout_w_init"] = (
            cfg["readout_w_init"]["mean"], cfg["readout_w_init"]["std"],
        )
    if explicit:
        kwargs.update(refractory_e_ms=1.2, refractory_i_ms=0.6,
                      refractory_policy="exact")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    return builder.build_net(cfg["model"], **kwargs)


def _input(cfg, batch=1, duration_ms=200.0):
    from torchvision.datasets import MNIST

    data = MNIST("/tmp/mnist", train=False, download=False)
    index = cfg["seed"] - 42
    pixels = data.data[index:index + batch].reshape(batch, -1).float() / 255.0
    labels = data.targets[index:index + batch]
    rate = cfg.get("input_rate") or 25.0
    spikes = encode_images_poisson(
        pixels, round(duration_ms / 0.1), 0.1, rate,
        generator=torch.Generator().manual_seed(20260415),
    )
    return spikes, labels


def _equal(left, right, label):
    assert left.keys() == right.keys(), label
    for key in left:
        assert torch.equal(left[key], right[key]), f"{label}: {key}"


@pytest.mark.parametrize("checkpoint", ["weights.pth", "weights_final.pth"])
def test_all_retained_cells_match_pre_adoption_forward(preservation_context, checkpoint):
    bank, cells, old_model, old_config = preservation_context
    for name in cells:
        unit = bank.unit(name)
        cfg = json.loads((unit / "config.json").read_text())
        old = _build(old_config, old_model, cfg, False)
        new = _build(current_config, current_model, cfg, True)
        _equal(old.state_dict(), new.state_dict(), f"{name}: initialization")
        weights = torch.load(unit / checkpoint, map_location="cpu", weights_only=True)
        old.load_state_dict(weights, strict=True)
        new.load_state_dict(weights, strict=True)
        spikes, _ = _input(cfg)
        results = []
        for net in (old, new):
            net.eval()
            net.recording = True
            torch.manual_seed(cfg["seed"] + 1000)
            with torch.no_grad():
                output = net(input_spikes=spikes, randomize_init=RANDOMIZE_INITIAL_STATE)
            results.append((output, net.spike_record, torch.get_rng_state()))
        for a, b, label in (
            (results[0][0], results[1][0], "readout"),
            (results[0][2], results[1][2], "RNG state"),
        ):
            assert torch.equal(a, b), f"{name}: {checkpoint}: {label}"
        _equal(results[0][1], results[1][1], f"{name}: {checkpoint}: trajectory")
        assert old.rates == new.rates
        print(json.dumps({"cell": name, "checkpoint": checkpoint, "equal": True,
                          "e_spikes": int(new.spike_record["hid"].sum()),
                          "i_spikes": int(new.spike_record["inh"].sum())}), flush=True)
        del old, new, results, weights
        gc.collect()


@pytest.mark.parametrize("name", [
    "coba__canonical__seed42", "ping__canonical__seed42",
    "ping__rt5hz__seed42", "trainable_ping_init__seed42",
])
def test_retained_training_forward_backward_matches(preservation_context, name):
    bank, _, old_model, old_config = preservation_context
    unit = bank.unit(name)
    cfg = json.loads((unit / "config.json").read_text())
    weights = torch.load(unit / "weights_final.pth", map_location="cpu", weights_only=True)
    # A complete 200-ms, two-image training batch exercises temporal gradients.
    spikes, labels = _input(cfg, batch=2)
    results = []
    for module, builder, explicit in (
        (old_model, old_config, False), (current_model, current_config, True),
    ):
        net = _build(builder, module, cfg, explicit)
        net.load_state_dict(weights, strict=True)
        net.train()
        torch.manual_seed(cfg["seed"] + 2000)
        logits = net(input_spikes=spikes, randomize_init=RANDOMIZE_INITIAL_STATE)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        strength = cfg.get("fr_reg_upper_strength", 0.0)
        if strength:
            target = cfg["fr_reg_upper_target_hz"]
            penalty = sum((counts.mean(dim=1) / 0.2 - target).clamp(min=0).square().mean()
                          for counts in net.last_spike_counts) / len(net.last_spike_counts)
            loss = loss + strength * penalty
        loss.backward()
        gradients = {key: parameter.grad.detach().clone()
                     for key, parameter in net.named_parameters() if parameter.grad is not None}
        assert gradients and all(torch.isfinite(value).all() for value in gradients.values())
        assert any(torch.count_nonzero(value) for value in gradients.values())
        results.append((logits.detach(), loss.detach(), gradients, torch.get_rng_state()))
        del logits, loss, net
        gc.collect()
    for index in (0, 1, 3):
        assert torch.equal(results[0][index], results[1][index]), (name, index)
    _equal(results[0][2], results[1][2], f"{name}: gradients")
