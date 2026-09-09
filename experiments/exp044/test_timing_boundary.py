"""Whole-trial timing agrees across the public simulator and experiment readers."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from experiments.exp044 import analyse, compute, evidence, inputs
from experiments.exp044.test import lab as lab
from experiments.helpers.operating_point import refractory_args
from experiments.helpers.run_cli import REPO, SNN_TOOL


@pytest.mark.parametrize("dt,steps,e_ref,i_ref", [(0.3, 666, 4, 2), (0.6, 333, 2, 1)])
def test_public_probe_uses_the_same_trial_length_and_rate_denominator_as_collection(
    tmp_path, dt, steps, e_ref, i_ref
):
    output = tmp_path / "probe"
    completed = subprocess.run(
        [
            sys.executable, str(SNN_TOOL), "sim", *refractory_args(),
            "--dt", str(dt), "--t-ms", "200", "--tau-gaba", "6",
            "--n-hidden", "4", "--n-inh", "2", "--n-in", "4",
            "--n-batch", "1", "--input", "synthetic-spikes",
            "--input-rate", "1000", "--private-w-in", "--w-in", "5",
            "--seed", "42", "--outputs", "rasters", "--out-dir", str(output),
        ],
        cwd=REPO,
        env={**os.environ, "PINGLAB_DEVICE": "cpu", "PINGLAB_NO_COMPILE": "1"},
        capture_output=True, text=True, timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    metrics = json.loads((output / "metrics.json").read_text())
    config = json.loads((output / "config.json").read_text())
    for metadata in (config, metrics["config"]):
        assert metadata["nominal_duration_ms"] == 200.0
        assert metadata["duration_steps"] == steps
        assert metadata["realized_duration_ms"] == pytest.approx(199.8)
        assert metadata["refractory_e_steps"] == e_ref
        assert metadata["refractory_i_steps"] == i_ref
    dense = {}
    with np.load(output / "rasters.npz") as raster:
        assert int(raster["T"]) == steps
        for population, width in (("e", 4), ("i", 2)):
            spikes = np.zeros((steps, width), dtype=bool)
            spikes[raster[population + "_t"], raster[population + "_cell"]] = True
            dense["spk_" + population] = spikes
            assert metrics["rate_" + population + "_hz"] == pytest.approx(
                spikes.sum() / (width * 0.1998)
            )
    assert dense["spk_e"].any(), "nonzero spikes are required to test normalization"
    retained = tmp_path / "recording.npz"
    np.savez_compressed(retained, dt=dt, **dense)
    accepted = evidence.snapshot(retained, dt, {"t_ms": 200.0, "n_hidden": 4, "n_inh": 2})
    assert accepted["spk_e"].shape == (steps, 4)


def test_coarse_timestep_analysis_normalizes_raster_counts_by_realized_duration(lab):
    root, bank, _ = lab
    computed = compute.compute(bank)
    identity = analyse.analyse(computed)
    run = inputs.source(root, identity, "analyse")
    results = json.loads((run.export / "results.json").read_text())
    coarse = {row["dt_ms"]: row for row in results["rasters"] if row["dt_ms"] in (0.3, 0.6)}
    assert coarse.keys() == {0.3, 0.6}
    for dt, n_steps, e_events, i_events in ((0.3, 666, 34, 17), (0.6, 333, 17, 9)):
        row = coarse[dt]
        assert row["nominal_t_ms"] == 200.0
        assert row["n_steps"] == n_steps
        assert row["t_ms"] == pytest.approx(199.8)
        assert row["e_rate_hz"] == pytest.approx(e_events / 0.1998)
        assert row["i_rate_hz"] == pytest.approx(i_events / 0.1998)
