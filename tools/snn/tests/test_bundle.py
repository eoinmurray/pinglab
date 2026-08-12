"""Compatibility gates for the additive snnlang bundle frontend."""

from __future__ import annotations

import json
import os
import subprocess

import config
import models as M
import pytest
import torch
import torch.nn.functional as F
from bundle import (
    BundleCompatibilityError,
    load_graph_bundle,
    load_training_recipe,
    translate_cobanet_v1,
    translate_training_v1,
)
from tool import parse_args

from tools.snnlang.examples.build_examples import ping_classifier


def _write_bundle(tmp_path):
    return ping_classifier().write(tmp_path / "network.bundle")


def _legacy_argv(mode="sim"):
    return [
        mode,
        "--n-hidden",
        "256",
        "--readout",
        "mem-mean",
        "--dt",
        "0.1",
        "--w-in",
        "0.2",
        "0.03",
        "--w-in-initial-zero-fraction",
        "0",
        "--w-ei",
        "0.5",
        "0.05",
        "--w-ie",
        "1.0",
        "0.1",
        "--ei-strength",
        "0.5",
        "--ei-ratio",
        "2",
        "--recurrent-initial-zero-fraction",
        "0",
        "--tau-gaba",
        "9",
    ]


def test_legacy_parse_defaults_are_unchanged():
    args = parse_args(["sim"])
    assert args.bundle is None
    assert args.model == "ping"
    assert args.n_hidden is None
    assert args.dt == pytest.approx(0.25)
    assert args.readout_mode == "rate"


def test_bundle_translates_to_same_structural_arguments_as_legacy(tmp_path):
    root = _write_bundle(tmp_path)
    bundle = parse_args(["sim", "--bundle", str(root)])
    legacy = parse_args(_legacy_argv())
    fields = (
        "model",
        "n_hidden",
        "readout_mode",
        "dt",
        "w_in",
        "w_in_initial_zero_fraction",
        "w_ei",
        "w_ie",
        "ei_strength",
        "ei_ratio",
        "recurrent_initial_zero_fraction",
        "tau_gaba",
    )
    assert {field: getattr(bundle, field) for field in fields} == {
        field: getattr(legacy, field) for field in fields
    }


def test_training_bundle_applies_graph_and_recipe(tmp_path):
    root = _write_bundle(tmp_path)
    args = parse_args(
        [
            "train",
            "--bundle",
            str(root),
            "--max-samples",
            "128",
            "--batch-size",
            "32",
        ]
    )
    assert args.model == "ping"
    assert args.dataset == "mnist"
    assert args.n_hidden == [256]
    assert args.dt == pytest.approx(0.1)
    assert args.readout_mode == "mem-mean"
    assert args.lr == pytest.approx(1e-3)
    assert args.weight_decay == pytest.approx(1e-4)
    assert args.epochs == 20
    assert args.max_samples == 128
    assert args.batch_size == 32


def test_training_recipe_flags_cannot_override_bundle(tmp_path):
    root = _write_bundle(tmp_path)
    with pytest.raises(SystemExit) as error:
        parse_args(["train", "--bundle", str(root), "--epochs", "1"])
    assert error.value.code == 2


def test_training_bundle_requires_authenticated_recipe(tmp_path):
    root = _write_bundle(tmp_path)
    (root / "training.json").unlink()
    with pytest.raises(SystemExit) as error:
        parse_args(["train", "--bundle", str(root)])
    assert error.value.code == 2


def test_training_recipe_rejects_unsupported_parameter_scope(tmp_path):
    root = _write_bundle(tmp_path)
    manifest, graph = load_graph_bundle(root)
    recipe = load_training_recipe(root, manifest, graph)
    recipe["parameter_groups"][0]["parameters"] = ["classifier_projection.weight"]
    recipe["parameter_groups"][1]["parameters"].append("sensory_ping_input.weight")
    with pytest.raises(BundleCompatibilityError, match="input/readout"):
        translate_training_v1(graph, recipe)


def _build_from_args(args):
    M.N_IN = args.n_in
    config.set_sim_dt(args.dt, getattr(args, "t_ms", 1.2))
    return config.build_net(
        args.model,
        w_in=args.w_in,
        w_in_initial_zero_fraction=args.w_in_initial_zero_fraction,
        w_ei=args.w_ei,
        w_ie=args.w_ie,
        ei_strength=args.ei_strength,
        ei_ratio=args.ei_ratio,
        recurrent_initial_zero_fraction=args.recurrent_initial_zero_fraction,
        hidden_sizes=args.n_hidden,
        readout_mode=args.readout_mode,
    )


def test_bundle_and_legacy_build_identical_cobanet(tmp_path):
    root = _write_bundle(tmp_path)
    bundle_args = parse_args(["sim", "--bundle", str(root)])
    legacy_args = parse_args([*_legacy_argv(), "--n-in", "784"])
    torch.manual_seed(17)
    bundle_net = _build_from_args(bundle_args)
    torch.manual_seed(17)
    legacy_net = _build_from_args(legacy_args)
    assert type(bundle_net) is type(legacy_net)
    assert sum(p.numel() for p in bundle_net.parameters()) == sum(
        p.numel() for p in legacy_net.parameters()
    )
    assert bundle_net.state_dict().keys() == legacy_net.state_dict().keys()
    for name, value in bundle_net.state_dict().items():
        torch.testing.assert_close(value, legacy_net.state_dict()[name], rtol=0, atol=0)


def _named_trainable(net):
    return {name: param for name, param in net.named_parameters() if param.requires_grad}


def _assert_tensor_maps_equal(stage, left, right):
    assert left.keys() == right.keys(), (
        f"{stage}: key mismatch "
        f"left_only={sorted(left.keys() - right.keys())} "
        f"right_only={sorted(right.keys() - left.keys())}"
    )
    for name in left:
        l_val = left[name]
        r_val = right[name]
        assert l_val.shape == r_val.shape, (
            f"{stage}: shape mismatch for {name}: {l_val.shape} != {r_val.shape}"
        )
        assert l_val.dtype == r_val.dtype, (
            f"{stage}: dtype mismatch for {name}: {l_val.dtype} != {r_val.dtype}"
        )
        torch.testing.assert_close(
            l_val,
            r_val,
            rtol=0,
            atol=0,
            msg=lambda msg, name=name, stage=stage: (
                f"{stage}: first divergent tensor {name}\n{msg}"
            ),
        )


def test_bundle_and_legacy_one_step_training_are_exactly_equivalent(tmp_path):
    root = _write_bundle(tmp_path)
    bundle_args = parse_args(["train", "--bundle", str(root), "--t-ms", "1.2"])
    legacy_args = parse_args(
        [
            *_legacy_argv("train"),
            "--t-ms",
            "1.2",
            "--lr",
            str(bundle_args.lr),
            "--weight-decay",
            str(bundle_args.weight_decay),
        ]
    )
    legacy_args.n_in = 784

    torch.manual_seed(123)
    bundle_net = _build_from_args(bundle_args)
    torch.manual_seed(123)
    legacy_net = _build_from_args(legacy_args)

    _assert_tensor_maps_equal(
        "initial state_dict",
        bundle_net.state_dict(),
        legacy_net.state_dict(),
    )

    bundle_trainable = _named_trainable(bundle_net)
    legacy_trainable = _named_trainable(legacy_net)
    assert set(bundle_trainable) == {"W_ff.0", "W_ff.1"}
    assert set(bundle_trainable) == set(legacy_trainable)
    assert {
        name for name, param in bundle_net.named_parameters() if not param.requires_grad
    } == {
        name for name, param in legacy_net.named_parameters() if not param.requires_grad
    }

    encoded_spikes = torch.zeros(12, 4, 784)
    encoded_spikes[0::3, :, 0:12] = 1.0
    encoded_spikes[1::3, :, 100:112] = 1.0
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    bundle_logits = bundle_net(input_spikes=encoded_spikes)
    legacy_logits = legacy_net(input_spikes=encoded_spikes)
    torch.testing.assert_close(
        bundle_logits,
        legacy_logits,
        rtol=0,
        atol=0,
        msg=lambda msg: f"forward logits diverged\n{msg}",
    )

    bundle_loss = F.cross_entropy(bundle_logits, labels)
    legacy_loss = F.cross_entropy(legacy_logits, labels)
    torch.testing.assert_close(
        bundle_loss,
        legacy_loss,
        rtol=0,
        atol=0,
        msg=lambda msg: f"cross-entropy loss diverged\n{msg}",
    )

    bundle_loss.backward()
    legacy_loss.backward()
    _assert_tensor_maps_equal(
        "gradients",
        {name: param.grad for name, param in bundle_trainable.items()},
        {name: param.grad for name, param in legacy_trainable.items()},
    )

    bundle_opt = torch.optim.AdamW(
        bundle_trainable.values(),
        lr=bundle_args.lr,
        weight_decay=bundle_args.weight_decay,
    )
    legacy_opt = torch.optim.AdamW(
        legacy_trainable.values(),
        lr=legacy_args.lr,
        weight_decay=legacy_args.weight_decay,
    )
    bundle_opt.step()
    legacy_opt.step()

    _assert_tensor_maps_equal(
        "post-AdamW state_dict",
        bundle_net.state_dict(),
        legacy_net.state_dict(),
    )

    bundle_opt_state = bundle_opt.state_dict()
    legacy_opt_state = legacy_opt.state_dict()
    assert bundle_opt_state["param_groups"] == legacy_opt_state["param_groups"]
    for param_idx, state in bundle_opt_state["state"].items():
        other = legacy_opt_state["state"][param_idx]
        _assert_tensor_maps_equal(
            f"AdamW optimizer state for parameter {param_idx}",
            state,
            other,
        )


def test_bundle_digest_is_authenticated(tmp_path):
    root = _write_bundle(tmp_path)
    graph_path = root / "graph.json"
    graph = json.loads(graph_path.read_text())
    graph["name"] = "tampered"
    graph_path.write_text(json.dumps(graph))
    with pytest.raises(BundleCompatibilityError, match="digest"):
        load_graph_bundle(root)


def test_structural_cli_override_is_rejected(tmp_path):
    root = _write_bundle(tmp_path)
    with pytest.raises(SystemExit) as error:
        parse_args(["sim", "--bundle", str(root), "--dt", "0.5"])
    assert error.value.code == 2


def test_bundle_path_is_not_inherited_from_legacy_load_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"bundle": "stale.bundle"}))
    with pytest.raises(SystemExit) as error:
        parse_args(["sim", "--load-config", str(config_path)])
    assert error.value.code == 2


def test_unsupported_graph_fails_with_capability_error(tmp_path):
    bundle = ping_classifier()
    bundle.graph["populations"].append(
        {
            "id": "extra",
            "size": 4,
            "neuron": {"kind": "lif"},
            "spiking": True,
            "group": None,
        }
    )
    with pytest.raises(BundleCompatibilityError, match="exactly two"):
        translate_cobanet_v1(bundle.graph)


def test_bundle_rejects_input_axis_order_that_backend_cannot_consume():
    bundle = ping_classifier()
    bundle.graph["inputs"][0]["shape"] = ["batch", "time", 784]
    with pytest.raises(BundleCompatibilityError, match="time.*batch.*channels"):
        translate_cobanet_v1(bundle.graph)


def test_bundle_cli_smoke_preserves_artifact_contract(tmp_path):
    root = _write_bundle(tmp_path)
    out = tmp_path / "run"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "tools/snn/tool.py",
            "sim",
            "--bundle",
            str(root),
            "--t-ms",
            "2",
            "--n-batch",
            "2",
            "--out-dir",
            str(out),
            "--wipe-dir",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PINGLAB_NO_COMPILE": "1"},
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "metrics.json").is_file()
    assert (out / "config.json").is_file()
    assert (out / "run.sh").is_file()
    config_data = json.loads((out / "config.json").read_text())
    assert config_data["bundle"] == str(root)
    assert config_data["n_hidden"] == [256]
