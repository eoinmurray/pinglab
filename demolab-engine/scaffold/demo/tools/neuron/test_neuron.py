"""Tests for the neuron tool's generic primitives (data-in/data-out) and the
manifest contract. No CLI, no files (except the write_output tmp cases)."""

import json

import numpy as np
import pytest

from tools.neuron.tool import (
    simulate_eif,
    simulate_eif_network,
    simulate_lif,
    simulate_network,
    write_output,
)


# --- single neurons ---------------------------------------------------------

def test_lif_shapes():
    t, v, spikes = simulate_lif(i_tonic=2.5, duration=100.0, dt=0.1)
    assert len(t) == len(v) == 1000
    assert isinstance(spikes, list)


def test_lif_suprathreshold_fires():
    _, _, spikes = simulate_lif(i_tonic=2.5)
    assert len(spikes) > 0


def test_lif_subthreshold_is_silent():
    # No drive: steady state sits at rest, never crosses threshold.
    _, v, spikes = simulate_lif(i_tonic=0.0)
    assert spikes == []
    assert np.max(v) < -50.0  # V_thresh


def test_lif_rate_increases_with_current():
    assert len(simulate_lif(i_tonic=5.0)[2]) > len(simulate_lif(i_tonic=2.0)[2])


def test_lif_is_deterministic():
    _, v1, s1 = simulate_lif(i_tonic=2.5)
    _, v2, s2 = simulate_lif(i_tonic=2.5)
    assert np.array_equal(v1, v2) and s1 == s2


def test_eif_fires_and_shapes():
    t, v, spikes = simulate_eif(i_tonic=3.0, duration=100.0, dt=0.1)
    assert len(t) == len(v) == 1000
    assert len(spikes) > 0


# --- networks (seeded) ------------------------------------------------------

def test_network_deterministic_given_seed():
    st1, ids1, is_exc1, *_ = simulate_network(n_neurons=50, duration=100.0, seed=7)
    st2, ids2, is_exc2, *_ = simulate_network(n_neurons=50, duration=100.0, seed=7)
    assert np.array_equal(st1, st2)
    assert np.array_equal(ids1, ids2)
    assert len(is_exc1) == 50
    assert len(st1) > 0


def test_network_seed_changes_output():
    a = simulate_network(n_neurons=50, duration=100.0, seed=1)[0]
    b = simulate_network(n_neurons=50, duration=100.0, seed=2)[0]
    assert not np.array_equal(a, b)


def test_eif_network_deterministic_given_seed():
    a = simulate_eif_network(n_neurons=50, duration=100.0, seed=3)[0]
    b = simulate_eif_network(n_neurons=50, duration=100.0, seed=3)[0]
    assert np.array_equal(a, b)


# --- the manifest contract --------------------------------------------------

def test_write_output_rejects_metric_not_in_output(tmp_path):
    (tmp_path / "fig.png").write_bytes(b"x")
    with pytest.raises(ValueError):
        write_output(
            tmp_path,
            {"present": 1.0},
            {"headline_figure": "fig.png", "headline_metrics": ["absent"]},
        )


def test_write_output_rejects_missing_figure(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_output(
            tmp_path,
            {"rate": 1.0},
            {"headline_figure": "nope.png", "headline_metrics": ["rate"]},
        )


def test_write_output_writes_contract_files(tmp_path):
    (tmp_path / "fig.png").write_bytes(b"x")
    write_output(
        tmp_path,
        {"rate": 90.0},
        {"headline_figure": "fig.png", "headline_metrics": ["rate"]},
    )
    assert json.loads((tmp_path / "output.json").read_text())["rate"] == 90.0
    assert (tmp_path / "manifest.json").exists()
