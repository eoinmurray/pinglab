"""Validate the frozen control and measure the three retained raster conditions."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
import torch
from experiments.exp169 import recipe
from pingstore.library import file_sha256, source_run, stage_run
from scipy.signal import lfilter
from snnlab.sim import NetworkSnapshot


def metrics(g, target):
    corr = np.corrcoef(g.T)
    return dict(
        mean=float(g.mean()),
        std=float(g.std()),
        per_cell_mean_relative_error_max=float(
            np.max(np.abs(g.mean(0) / target.mean(0) - 1))
        ),
        per_cell_std_relative_error_max=float(
            np.max(np.abs(g.std(0) / target.std(0) - 1))
        ),
        pairwise_correlation_mean=float(corr[np.triu_indices(g.shape[1], 1)].mean()),
        population_mean_temporal_std=float(g.mean(1).std()),
    )


def read_evidence(computed):
    pin = computed.record["inputs"]["bank"]
    bank = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        stage="compute",
        experiment="exp022",
        reference=pin,
    )
    cfg = computed.record["execution"]["configuration"]
    rows = []
    inputs = []
    for name, cell in [("COBA", cfg["coba_cell"]), ("PING", cfg["ping_cell"])]:
        with np.load(bank.file("snapshots", cell, "recording.npz")) as z:
            rows.append(dict(name=name, e=z["spk_e"].copy(), i=z["spk_i"].copy()))
            inputs.append(z["input_spikes"].copy())
            if name == "PING":
                target = z["gi_e_1"].astype(float)
    if not np.array_equal(*inputs):
        raise ValueError("Sensory inputs differ")
    with np.load(computed.file("artificial.npz")) as z:
        e = z["E_spikes"][:, 0, :].copy()
        conductance = z["g_artificial"][:, 0, :].astype(float)
    with np.load(computed.file("calibration.npz")) as z:
        events, amplitudes = z["events"].copy(), z["amplitudes_us"].copy()
    rows.append(dict(name="COBA + asynchronous GABA", e=e, i=events))
    return cfg, bank, rows, target, conductance, amplitudes


def validate(computed):
    cfg, bank, rows, target, conductance, amplitudes = read_evidence(computed)
    snapshots = {
        name: NetworkSnapshot.read(computed.unit(name, "start"), parameters=True)
        for name in ("baseline", "artificial")
    }
    original = torch.load(
        bank.file(cfg["coba_cell"], cfg["checkpoint"]["filename"]),
        map_location="cpu",
        weights_only=True,
    )
    if (
        file_sha256(bank.file(cfg["coba_cell"], cfg["checkpoint"]["filename"]))
        != cfg["checkpoint"]["sha256"]
    ):
        raise ValueError("Checkpoint digest mismatch")
    mapping = {
        "input_to_E.weight": "W_ff.0",
        "E_to_I.weight": "W_ei.1",
        "I_to_E.weight": "W_ie.1",
        "readout_projection.weight": "W_ff.1",
    }
    for snapshot in snapshots.values():
        parameters = snapshot.load_parameters()
        for name, key in mapping.items():
            np.testing.assert_array_equal(
                np.asarray(parameters[name]), original[key].clamp(min=0).numpy()
            )
        np.testing.assert_array_equal(
            np.asarray(parameters["artificial_to_E.weight"]), np.diag(amplitudes)
        )
    events = rows[-1]["i"]
    for row in rows:
        for spikes in (row["e"], row["i"]):
            if not np.isin(spikes, [0, 1]).all():
                raise ValueError("Nonbinary spikes")
        if row["e"].shape != (2000, 1024):
            raise ValueError("Wrong E dimensions")
    delayed = np.vstack([np.zeros((1, 1024)), events[:-1]])
    predicted = (
        lfilter(
            [1.0], [1.0, -np.exp(-cfg["dt_ms"] / cfg["tau_gaba_ms"])], delayed, axis=0
        )
        * amplitudes
    )
    filter_error = float(np.max(np.abs(predicted - conductance)))
    if filter_error > 1e-5:
        raise ValueError("Conductance differs from delayed GABA filter")
    matched = metrics(conductance, target)
    if (
        matched["per_cell_mean_relative_error_max"] > cfg["mean_relative_tolerance"]
        or matched["per_cell_std_relative_error_max"] > cfg["std_relative_tolerance"]
    ):
        raise ValueError("Conductance moments do not match")
    if abs(matched["pairwise_correlation_mean"]) >= 0.1:
        raise ValueError("Excess shared fluctuation")
    with np.load(computed.file("baseline.npz")) as z:
        baseline = z["E_spikes"][:, 0, :]
    np.testing.assert_array_equal(baseline.sum(0), rows[0]["e"].sum(0))
    saved = json.loads(computed.file("summary.json").read_text())
    seconds = cfg["duration_ms"] / 1000
    reference_config = json.loads(
        bank.file(cfg["coba_cell"], "config.json").read_text()
    )
    return dict(
        schema="exp169.analysis/v1",
        configuration=cfg,
        reference_input_rate_hz=reference_config["input_rate"],
        conditions=[
            dict(
                name=row["name"],
                rate_e_hz=float(row["e"].sum() / 1024 / seconds),
                rate_i_or_input_hz=float(row["i"].sum() / row["i"].shape[1] / seconds),
                inhibitory_channels=row["i"].shape[1],
            )
            for row in rows
        ],
        matched=matched,
        ping=metrics(target, target),
        checks=dict(
            frozen_checkpoint_parameters_exact=True,
            private_diagonal_parameters_exact=True,
            native_baseline_counts_exact=True,
            sensory_input_identical=True,
            gaba_filter_max_error_us=filter_error,
            legacy_baseline_mismatched_spike_bins=int(
                np.count_nonzero(baseline != rows[0]["e"])
            ),
        ),
        original_simulation_seconds=saved["simulation_seconds"],
        limitations=saved["limitations"],
    )


def analyse(identity, *, run_id=None):
    started = perf_counter()
    computed = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="compute"
    )
    results = validate(computed)
    with stage_run(
        ROOT,
        recipe.SLUG,
        "analyse",
        inputs={"compute": computed},
        configuration={
            "measurement": "full 200-ms per-cell moments and raw spike events"
        },
        run_id=run_id,
    ) as run:
        (run.export / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        run.record["execution"].update(
            no_simulation=True, elapsed_seconds=perf_counter() - started
        )
        computed.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    analyse(args.source, run_id=args.run_id)
