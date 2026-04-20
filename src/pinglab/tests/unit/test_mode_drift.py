"""Mode-drift: "same inputs → same outputs across modes".

In-process invariants are fast (default). CLI propagation tests spawn
subprocesses and are marked `slow`.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch


# ── In-process invariants (fast) ─────────────────────────────────────────

def _equal_state_dicts(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_build_net_deterministic_snntorch_canonical():
    from config import build_net
    torch.manual_seed(0)
    a = build_net("standard-snn", w_in=(10.0, 1.0), w_in_sparsity=0.0)
    torch.manual_seed(0)
    b = build_net("standard-snn", w_in=(10.0, 1.0), w_in_sparsity=0.0)
    assert _equal_state_dicts(a, b)


def test_build_net_deterministic_ping():
    from config import build_net
    torch.manual_seed(0)
    a = build_net("ping", w_in=(0.3, 0.03), w_in_sparsity=0.95,
                  ei_strength=0.5, ei_ratio=2.0, sparsity=0.2)
    torch.manual_seed(0)
    b = build_net("ping", w_in=(0.3, 0.03), w_in_sparsity=0.95,
                  ei_strength=0.5, ei_ratio=2.0, sparsity=0.2)
    assert _equal_state_dicts(a, b)


def test_encode_images_poisson_deterministic():
    from oscilloscope import encode_images_poisson
    images = torch.rand(4, 64)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = encode_images_poisson(images, T_steps=200, dt=0.25,
                              max_rate_hz=10.0, generator=g1)
    b = encode_images_poisson(images, T_steps=200, dt=0.25,
                              max_rate_hz=10.0, generator=g2)
    assert torch.equal(a, b)


def test_load_dataset_deterministic_scikit():
    from oscilloscope import load_dataset
    a_tr, a_te, ay_tr, ay_te = load_dataset("scikit", max_samples=200, split=True)
    b_tr, b_te, by_tr, by_te = load_dataset("scikit", max_samples=200, split=True)
    assert np.array_equal(a_tr, b_tr)
    assert np.array_equal(a_te, b_te)
    assert np.array_equal(ay_tr, by_tr)
    assert np.array_equal(ay_te, by_te)


def test_train_and_infer_share_test_split_mnist():
    from oscilloscope import load_dataset
    _, train_X_te, _, train_y_te = load_dataset("mnist", max_samples=500, split=True)
    _, infer_X_te, _, infer_y_te = load_dataset("mnist", max_samples=500, split=True)
    assert np.array_equal(train_X_te, infer_X_te)
    assert np.array_equal(train_y_te, infer_y_te)


# ── CLI propagation (slow) ───────────────────────────────────────────────

def _run_cli(*args, timeout=180):
    cmd = ["uv", "run", "python", "src/pinglab/oscilloscope.py", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def _train_probe(tmp_dir, **extra):
    args = ["train", "--model", "standard-snn",
            "--dataset", "mnist", "--max-samples", "100",
            "--epochs", "0", "--dt", "0.25",
            "--w-in", "10", "--w-in-sparsity", "0",
            "--out-dir", str(tmp_dir), "--wipe-dir"]
    for k, v in extra.items():
        args.extend([f"--{k.replace('_', '-')}", str(v)])
    rc, _, _ = _run_cli(*args)
    assert rc == 0
    metrics_path = Path(tmp_dir) / "metrics.json"
    assert metrics_path.exists()
    return json.loads(metrics_path.read_text())


def _read_config(tmp_dir):
    cfg_path = Path(tmp_dir) / "config.json"
    assert cfg_path.exists(), f"no config.json in {tmp_dir}"
    return json.loads(cfg_path.read_text())


@pytest.mark.slow
def test_input_rate_propagates_train(tmp_path):
    expected = 33.0
    metrics = _train_probe(tmp_path / "ir-train", **{"input-rate": expected})
    assert float(metrics["config"]["input_rate"]) == expected


@pytest.mark.slow
def test_input_rate_propagates_image(tmp_path):
    expected = 33.0
    out = tmp_path / "ir-image"
    rc, _, _ = _run_cli("image", "--model", "standard-snn",
                        "--dataset", "mnist", "--digit", "0",
                        "--dt", "0.25", "--w-in", "0.1",
                        "--input-rate", str(expected),
                        "--out-dir", str(out), "--wipe-dir")
    assert rc == 0
    assert float(_read_config(out)["spike_rate"]) == expected


@pytest.mark.slow
def test_input_rate_propagates_video(tmp_path):
    expected = 33.0
    out = tmp_path / "ir-video"
    rc, _, _ = _run_cli("video", "--model", "standard-snn",
                        "--dataset", "mnist", "--digit", "0",
                        "--scan-var", "stim-overdrive",
                        "--scan-min", "1", "--scan-max", "2",
                        "--frames", "2", "--dt", "0.25",
                        "--w-in", "0.1",
                        "--input-rate", str(expected),
                        "--out-dir", str(out), "--wipe-dir")
    assert rc == 0
    assert float(_read_config(out)["spike_rate"]) == expected


@pytest.mark.slow
def test_t_ms_propagates_train(tmp_path):
    expected = 150.0
    metrics = _train_probe(tmp_path / "tms-train", **{"t-ms": expected})
    assert float(metrics["config"]["t_ms"]) == expected


@pytest.mark.slow
def test_t_ms_propagates_image(tmp_path):
    expected = 150.0
    out = tmp_path / "tms-image"
    rc, _, _ = _run_cli("image", "--model", "standard-snn",
                        "--dataset", "mnist", "--digit", "0",
                        "--dt", "0.25", "--w-in", "0.1",
                        "--t-ms", str(expected),
                        "--out-dir", str(out), "--wipe-dir")
    assert rc == 0
    assert float(_read_config(out)["t_ms"]) == expected


@pytest.mark.slow
def test_t_ms_propagates_video(tmp_path):
    expected = 150.0
    out = tmp_path / "tms-video"
    rc, _, _ = _run_cli("video", "--model", "standard-snn",
                        "--dataset", "mnist", "--digit", "0",
                        "--scan-var", "stim-overdrive",
                        "--scan-min", "1", "--scan-max", "2",
                        "--frames", "2", "--dt", "0.25",
                        "--w-in", "0.1",
                        "--t-ms", str(expected),
                        "--out-dir", str(out), "--wipe-dir")
    assert rc == 0
    assert float(_read_config(out)["t_ms"]) == expected


@pytest.mark.slow
def test_train_then_infer_match(tmp_path):
    """infer accuracy on a freshly trained checkpoint == train's last-epoch eval."""
    train_dir = tmp_path / "match-train"
    rc, _, _ = _run_cli("train", "--model", "standard-snn",
                        "--dataset", "mnist", "--max-samples", "200",
                        "--epochs", "2", "--dt", "0.25",
                        "--w-in", "10", "--w-in-sparsity", "0",
                        "--ei-strength", "0",
                        "--out-dir", str(train_dir), "--wipe-dir")
    assert rc == 0
    train_metrics = json.loads((train_dir / "metrics.json").read_text())
    # weights.pth holds the BEST-epoch state_dict, not the last one, so we
    # compare infer's fresh eval against train's best_acc.
    train_best_acc = train_metrics["best_acc"]
    assert train_best_acc is not None

    infer_dir = tmp_path / "match-infer"
    rc, _, _ = _run_cli("infer", "--model", "standard-snn",
                        "--dataset", "mnist", "--max-samples", "200",
                        "--dt", "0.25",
                        "--w-in", "10", "--w-in-sparsity", "0",
                        "--ei-strength", "0",
                        "--load-weights", str(train_dir / "weights.pth"),
                        "--out-dir", str(infer_dir), "--wipe-dir")
    assert rc == 0
    infer_acc = json.loads((infer_dir / "metrics.json").read_text())["best_acc"]
    assert abs(infer_acc - train_best_acc) < 0.01, \
        f"train best={train_best_acc}% infer={infer_acc}%"
