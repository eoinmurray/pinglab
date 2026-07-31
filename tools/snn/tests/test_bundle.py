"""Compatibility gates for the additive snnlang bundle frontend."""

from __future__ import annotations

import json
import os
import subprocess

import config
import models as M
import pytest
import torch
from bundle import (
    BundleCompatibilityError,
    load_graph_bundle,
    translate_cobanet_v1,
)
from tool import parse_args

from tools.snnlang.examples.build_examples import ping_classifier


def _write_bundle(tmp_path):
    return ping_classifier().write(tmp_path / "network.bundle")


def _legacy_argv():
    return [
        "sim",
        "--n-hidden",
        "256",
        "--readout",
        "mem-mean",
        "--dt",
        "0.1",
        "--w-in",
        "0.2",
        "0.03",
        "--w-in-sparsity",
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
        "--ei-sparsity",
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
        "w_in_sparsity",
        "w_ei",
        "w_ie",
        "ei_strength",
        "ei_ratio",
        "ei_sparsity",
        "tau_gaba",
    )
    assert {field: getattr(bundle, field) for field in fields} == {
        field: getattr(legacy, field) for field in fields
    }


def _build_from_args(args):
    M.N_IN = args.n_in
    return config.build_net(
        args.model,
        w_in=args.w_in,
        w_in_sparsity=args.w_in_sparsity,
        w_ei=args.w_ei,
        w_ie=args.w_ie,
        ei_strength=args.ei_strength,
        ei_ratio=args.ei_ratio,
        sparsity=args.ei_sparsity,
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
