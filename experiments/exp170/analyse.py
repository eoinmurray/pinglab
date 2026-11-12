"""Authenticate retained inhibition and derive report measurements."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
import torch
from experiments.exp170 import recipe
from pingstore.library import file_sha256, source_run, stage_run
from scipy.signal import lfilter
from snnlab.sim import NetworkSnapshot, load_runtime_state


def measure(events, g):
    rates = events.sum(0) / 0.2
    corr = np.corrcoef(g.T)
    return dict(
        sources=events.shape[1],
        events=int(events.sum()),
        source_rate_mean_hz=float(rates.mean()),
        source_rate_sd_hz=float(rates.std()),
        conductance_mean_us=float(g.mean()),
        mean_per_cell_temporal_sd_us=float(g.std(0).mean()),
        pairwise_conductance_correlation=float(
            corr[np.triu_indices(g.shape[1], 1)].mean()
        ),
        population_mean_conductance_temporal_sd_us=float(g.mean(1).std()),
    )


def read_evidence(computed):
    cfg = computed.details["execution"]["configuration"]
    pin = computed.details["inputs"]["bank"]
    bank = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        experiment="exp022",
        stage="compute",
        reference=pin,
    )
    rows = []
    inputs = []
    for name, cell in [("COBA", "coba__off__seed42"), ("PING", "ping__off__seed42")]:
        with np.load(bank.file("snapshots", cell, "recording.npz")) as z:
            rows.append(dict(name=name, e=z["spk_e"].copy(), i=z["spk_i"].copy()))
            inputs.append(z["input_spikes"].copy())
            if name == "PING":
                target = z["gi_e_1"].astype(float)
    if not np.array_equal(*inputs):
        raise ValueError("Sensory inputs differ")
    with np.load(computed.file("recording.npz")) as z:
        e = z["E_spikes"][:, 0, :].copy()
        g = z["g_artificial"][:, 0, :].astype(float)
    with np.load(computed.file("calibration.npz")) as z:
        events = z["events"].copy()
        amp = z["amplitude_us"].copy()
    label = (
        "COBA + Poisson GABA"
        if cfg.get("inhibition_method") == recipe.POISSON_METHOD
        else "COBA + count-matched GABA"
    )
    rows.append(dict(name=label, e=e, i=events))
    return cfg, bank, rows, target, g, amp


def validate_event_counts(computed, cfg, rows, target, events, amp):
    poisson = cfg.get("inhibition_method") == recipe.POISSON_METHOD
    if poisson:
        if not np.issubdtype(events.dtype, np.integer) or np.any(events < 0):
            raise ValueError("Invalid Poisson event counts")
        validate_poisson_draw(computed, cfg, target, events, amp)
    else:
        count = cfg["target_events_per_source"]
        if not np.all(events.sum(0) == count) or not np.all(
            rows[1]["i"].sum(0) == count
        ):
            raise ValueError("Source counts differ")
        if not np.isin(events, [0, 1]).all():
            raise ValueError("Nonbinary events")


def validate(computed):
    cfg, bank, rows, target, g, amp = read_evidence(computed)
    poisson = cfg.get("inhibition_method") == recipe.POISSON_METHOD
    events = rows[-1]["i"]
    validate_event_counts(computed, cfg, rows, target, events, amp)
    cp = cfg["checkpoint"]
    checkpoint = bank.file("coba__off__seed42", cp["filename"])
    if file_sha256(checkpoint) != cp["sha256"]:
        raise ValueError("Checkpoint digest mismatch")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved = NetworkSnapshot.read(
        computed.unit("execution", "start"), parameters=True
    ).load_parameters()
    for name, key in {
        "input_to_E.weight": "W_ff.0",
        "E_to_I.weight": "W_ei.1",
        "I_to_E.weight": "W_ie.1",
        "readout_projection.weight": "W_ff.1",
    }.items():
        np.testing.assert_array_equal(saved[name], weights[key].clamp(min=0).numpy())
    np.testing.assert_array_equal(saved["artificial_to_E.weight"], np.diag(amp))
    history_steps = round(cfg.get("prehistory_ms", 0) / cfg["dt_ms"])
    with np.load(computed.file("calibration.npz")) as z:
        history = (
            z["prehistory_events"].copy()
            if history_steps
            else np.empty((0, events.shape[1]))
        )
    if (
        history.shape != (history_steps, events.shape[1])
        or not (
            np.isfinite(history).all()
            and np.all(history >= 0)
            and np.all(history == np.floor(history))
        )
        or (not poisson and not np.isin(history, [0, 1]).all())
    ):
        raise ValueError("Invalid inhibitory prehistory")
    full = np.concatenate((history, events))
    filtered = (
        lfilter(
            [1.0],
            [1.0, -np.exp(-cfg["dt_ms"] / cfg["tau_gaba_ms"])],
            np.vstack([np.zeros((1, events.shape[1])), full[:-1]]),
            axis=0,
        )
        * amp
    )
    predicted = filtered[history_steps:]
    if history_steps:
        state = load_runtime_state(computed.unit("artificial-initial-state"))
        np.testing.assert_allclose(
            state.conductances["artificial_to_E"],
            filtered[history_steps - 1 : history_steps],
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            state.input_histories["artificial"], history[-1:, None, :]
        )
        if state.completed_steps or state.reduction_state or state.reduction_tensors:
            raise ValueError("Prehistory advanced the scored trajectory")
        for name, value in state.voltages.items():
            np.testing.assert_array_equal(
                value,
                np.full(
                    value.shape, 0 if name == "readout" else cfg["initial_voltage_mv"]
                ),
            )
        for group in (state.refractory, state.population_histories):
            if any(torch.count_nonzero(value) for value in group.values()):
                raise ValueError("Prehistory evolved neurons")
        if any(
            torch.count_nonzero(value)
            for name, value in state.conductances.items()
            if name != "artificial_to_E"
        ):
            raise ValueError("Prehistory leaked into other conductances")
    filter_error = float(np.max(np.abs(predicted - g)))
    mean_error = float(np.max(np.abs(g.mean(0) / target.mean(0) - 1)))
    sd_error = float(np.max(np.abs(g.std(0) / target.std(0) - 1)))
    if (
        filter_error > 1e-5
        or (not poisson and mean_error > cfg["mean_tolerance"])
        or (not poisson and sd_error > cfg["sd_tolerance"])
    ):
        raise ValueError("GABA matching failed")
    artificial = measure(events, g)
    ping = measure(rows[1]["i"], target)
    if artificial["pairwise_conductance_correlation"] >= 0.1:
        raise ValueError("Shared conductance fluctuations remain too large")
    config = json.loads(bank.file("coba__off__seed42", "config.json").read_text())
    raw = json.loads(computed.file("summary.json").read_text())
    return dict(
        schema="exp170.analysis/v1",
        configuration=cfg,
        ping=ping,
        artificial=artificial,
        stationary_expected=raw.get("stationary_expected"),
        conditions=[
            dict(
                name=r["name"],
                rate_e_hz=float(r["e"].sum() / r["e"].shape[1] / 0.2),
                rate_i_or_input_hz=float(r["i"].sum() / r["i"].shape[1] / 0.2),
            )
            for r in rows
        ],
        checks=dict(
            exact_source_counts=not poisson,
            poisson_counts_unconditioned=poisson,
            frozen_checkpoint_exact=True,
            private_diagonal_exact=True,
            mean_relative_error_max=mean_error,
            sd_relative_error_max=sd_error,
            gaba_filter_max_error_us=filter_error,
        ),
        reference_input_rate_hz=config["input_rate"],
        original_simulation_seconds=raw["simulation_seconds"],
    )


def validate_poisson_draw(computed, cfg, target, events, amp):
    from experiments.exp170.compute import (
        fit_events,
        poisson_expected_moments,
        poisson_parameters,
    )

    rates, fitted_amp = poisson_parameters(target, cfg)
    history_steps = round(cfg["prehistory_ms"] / cfg["dt_ms"])
    with np.load(computed.file("calibration.npz")) as z:
        np.testing.assert_array_equal(z["poisson_rate_hz"], rates)
        np.testing.assert_array_equal(amp, fitted_amp)
        np.testing.assert_array_equal(z["target_mean_us"], target.mean(0))
        np.testing.assert_array_equal(z["target_sd_us"], target.std(0))
        sampled = fit_events(target, cfg)[0]
        np.testing.assert_array_equal(z["prehistory_events"], sampled[:history_steps])
        np.testing.assert_array_equal(events, sampled[history_steps:])
        mean, sd = poisson_expected_moments(rates, amp, cfg)
        np.testing.assert_array_equal(z["expected_mean_us"], mean)
        np.testing.assert_array_equal(z["expected_sd_us"], sd)
        np.testing.assert_allclose(mean, target.mean(0), rtol=1e-6, atol=0)
        np.testing.assert_allclose(sd, target.std(0), rtol=1e-6, atol=0)


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
            "measurement": "Full-trial event counts, conductance moments, pairwise correlations and raw E rates"
        },
        run_id=run_id,
    ) as run:
        (run.export / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        run.details["execution"].update(
            no_simulation=True, elapsed_seconds=perf_counter() - started
        )
        computed.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    args = p.parse_args()
    analyse(args.source, run_id=args.run_id)
