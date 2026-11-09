# ruff: noqa: E402
"""Fit Poisson rates/amplitudes, then execute one frozen native COBA trial."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
import torch
from experiments.exp170 import recipe
from experiments.exp170.analyse import measure
from experiments.exp170.custom.prehistory import filtered_events, initial_state
from pingstore.library import file_sha256, source_run, stage_run
from scipy.signal import lfilter
from snnlab import lang
from snnlab.sim import (
    DenseArrayBinding,
    Performance,
    SnapshotSpec,
    build,
    save_runtime_state,
    simulate,
)


def poisson_parameters(target, cfg):
    """Match stationary moments of the actual discrete filter, not a sampled draw."""
    if target.ndim != 2 or not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("Invalid PING conductance target")
    mu, variance = target.mean(0), target.var(0)
    active = mu > 0
    if np.any(active & (variance == 0)):
        raise ValueError(
            "Constant positive conductance cannot be represented by finite-rate Poisson input"
        )
    dt_s = cfg["dt_ms"] / 1000
    decay = np.exp(-cfg["dt_ms"] / cfg["tau_gaba_ms"])
    amplitude = np.zeros_like(mu, dtype=np.float32)
    rates = np.zeros_like(mu, dtype=np.float64)
    amplitude[active] = ((1 + decay) * variance[active] / mu[active]).astype(np.float32)
    rates[active] = (
        mu[active] ** 2 * (1 - decay) / (dt_s * (1 + decay) * variance[active])
    )
    if (
        not np.isfinite(rates).all()
        or not np.isfinite(amplitude).all()
        or np.any(amplitude[active] <= 0)
    ):
        raise ValueError("Unrepresentable Poisson calibration")
    return rates, amplitude


def poisson_expected_moments(rates, amplitude, cfg):
    decay = np.exp(-cfg["dt_ms"] / cfg["tau_gaba_ms"])
    increments = rates * cfg["dt_ms"] / 1000
    return (
        increments * amplitude / (1 - decay),
        np.sqrt(increments * amplitude.astype(np.float64) ** 2 / (1 - decay**2)),
    )


def fit_events(target, cfg):
    if cfg.get("inhibition_method") != recipe.POISSON_METHOD:
        return fit_count_matched_events(target, cfg)
    rates, amplitude = poisson_parameters(target, cfg)
    history = round(cfg["prehistory_ms"] / cfg["dt_ms"])
    events = (
        np.random.default_rng(cfg["artificial_seed"])
        .poisson(
            rates * cfg["dt_ms"] / 1000, size=(len(target) + history, target.shape[1])
        )
        .astype(np.int32)
    )
    predicted = filtered_events(events, cfg)[history:] * amplitude
    return events, amplitude, predicted, rates, 0


def fit_count_matched_events(target, cfg):
    steps, channels = target.shape
    k = cfg["target_events_per_source"]
    decay = np.exp(-cfg["dt_ms"] / cfg["tau_gaba_ms"])
    mu = target.mean(0)
    cv = target.std(0) / mu
    rng = np.random.default_rng(cfg["artificial_seed"])
    # Keep every event deliverable inside the trial despite the one-bin delay.
    regular = (np.arange(k)[:, None] + rng.random(channels)[None, :]) * (steps - 1) / k
    random = np.empty((k, channels))
    bracketed = np.zeros(channels, dtype=bool)

    history = round(cfg.get("prehistory_ms", 0) / cfg["dt_ms"])
    history_regular = (
        (np.arange(k)[:, None] + rng.random(channels)[None, :]) * (steps - 1) / k
        if history
        else None
    )
    history_random = np.empty((k, channels)) if history else None

    def traces(times, past_times=None):
        def binary(values, *, pending=False):
            bins = np.minimum(
                np.rint(values).astype(int), steps - 1 if pending else steps - 2
            )
            if np.any(np.diff(bins, axis=0) <= 0):
                raise ValueError("Timing interpolation lost distinct event bins")
            events = np.zeros((steps, values.shape[1]), dtype=np.uint8)
            events[bins, np.arange(values.shape[1])[None, :]] = 1
            return events

        events = binary(times)
        if history:
            past = binary(past_times, pending=True)[-history:]
            events = np.concatenate((past, events))
        delayed = np.vstack([np.zeros((1, times.shape[1])), events[:-1]])
        h = lfilter([1.0], [1.0, -decay], delayed, axis=0)[history:]
        return events, h, h.std(0) / h.mean(0)

    _, _, regular_cv = traces(regular, history_regular)
    if not np.all(regular_cv < cv):
        raise ValueError(
            ("regular endpoint exceeds target", float(np.max(regular_cv - cv)))
        )
    bracket_draws = 0
    for _ in range(cfg["bracket_attempts"]):
        indices = np.flatnonzero(~bracketed)
        if not len(indices):
            break
        sampled = np.column_stack(
            [np.sort(rng.choice(steps - 1, k, replace=False)) for j in indices]
        )
        if history:
            history_random[:, indices] = np.column_stack(
                [np.sort(rng.choice(steps - 1, k, replace=False)) for _ in indices]
            )
        _, _, sample_cv = traces(
            sampled, history_random[:, indices] if history else None
        )
        random[:, indices] = sampled
        bracketed[indices] = sample_cv >= cv[indices]
        bracket_draws += len(indices)
    assert bracketed.all(), "Could not bracket all conductance CV targets"
    lo = np.zeros(channels)
    hi = np.ones(channels)
    besterr = np.full(channels, np.inf)
    bestalpha = np.zeros(channels)
    for _ in range(cfg["timing_bisections"]):
        alpha = (lo + hi) / 2
        _, _, candidate_cv = traces(
            regular * (1 - alpha) + random * alpha,
            history_regular * (1 - alpha) + history_random * alpha if history else None,
        )
        error = np.abs(candidate_cv / cv - 1)
        improved = error < besterr
        besterr[improved] = error[improved]
        bestalpha[improved] = alpha[improved]
        lo = np.where(candidate_cv < cv, alpha, lo)
        hi = np.where(candidate_cv >= cv, alpha, hi)
    events, h, _ = traces(
        regular * (1 - bestalpha) + random * bestalpha,
        history_regular * (1 - bestalpha) + history_random * bestalpha
        if history
        else None,
    )
    amplitude = (mu / h.mean(0)).astype(np.float32)
    assert np.all(events[history:].sum(0) == k)
    assert besterr.max() < cfg["sd_tolerance"], float(besterr.max())
    return events, amplitude, h * amplitude, bestalpha, bracket_draws


def compute(*, run_id=None):
    tick = time.perf_counter()
    torch.set_num_threads(1)
    cfg = recipe.configuration()
    old = source_run(
        ROOT / ".pingstore",
        cfg["source_run"],
        experiment="exp169",
        stage="compute",
        reference={"run_id": cfg["source_run"], "payload_digest": cfg["source_digest"]},
    )
    pin = old.record["inputs"]["bank"]
    bank = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        experiment="exp022",
        stage="compute",
        reference=pin,
    )
    original = json.loads(old.file("summary.json").read_text())
    oldcfg = old.record["execution"]["configuration"]
    data = {}
    for key, cell in [("coba", oldcfg["coba_cell"]), ("ping", oldcfg["ping_cell"])]:
        with np.load(bank.file("snapshots", cell, "recording.npz")) as z:
            data[key] = {
                k: z[k].copy() for k in ("spk_e", "spk_i", "gi_e_1", "input_spikes")
            }
    assert np.array_equal(data["coba"]["input_spikes"], data["ping"]["input_spikes"])
    target = data["ping"]["gi_e_1"].astype(float)
    t = time.perf_counter()
    events, amplitude, predicted, rates, _ = fit_events(target, cfg)
    expected_mu, expected_sd = poisson_expected_moments(rates, amplitude, cfg)
    np.testing.assert_allclose(expected_mu, target.mean(0), rtol=1e-6, atol=0)
    np.testing.assert_allclose(expected_sd, target.std(0), rtol=1e-6, atol=0)
    cfg["calibration_seconds"] = time.perf_counter() - t
    print(
        "CALIBRATED",
        cfg["calibration_seconds"],
        "Poisson rate range (Hz)",
        float(rates.min()),
        float(rates.max()),
        flush=True,
    )
    cp = oldcfg["checkpoint"]
    checkpoint = bank.file(oldcfg["coba_cell"], cp["filename"])
    assert cp["sha256"] == file_sha256(checkpoint)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg["checkpoint"] = cp
    cfg["scripts"] = {
        str(p.relative_to(HERE)): file_sha256(p) for p in HERE.rglob("*.py")
    }
    with stage_run(
        ROOT,
        recipe.SLUG,
        "compute",
        inputs={"original_control": old, "bank": bank},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        bundle = recipe.author_network(cfg)
        bundle.write(run.export / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        for p in HERE.rglob("*.py"):
            destination = run.export / p.relative_to(HERE)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(p.read_bytes())
        built = build(
            graph=bundle.graph,
            seed=cfg["seed"],
            device="cpu",
            performance=Performance(compile_mode=None),
        )
        mapping = {
            "input_to_E.weight": "W_ff.0",
            "E_to_I.weight": "W_ei.1",
            "I_to_E.weight": "W_ie.1",
            "readout_projection.weight": "W_ff.1",
        }
        expected = {
            name: weights[key].clamp(min=0).clone() for name, key in mapping.items()
        }
        expected["artificial_to_E.weight"] = torch.diag(torch.from_numpy(amplitude))
        assert set(built.model.parameter_map()) == set(expected)
        with torch.no_grad():
            for key, p in built.model.parameter_map().items():
                p.copy_(expected[key])
                p.requires_grad_(False)
        state = run.scratch / "parameters.pt"
        torch.save(built.model.state_dict(), state)
        drive = torch.from_numpy(data["coba"]["input_spikes"][:, None, :]).float()
        history_steps = round(cfg["prehistory_ms"] / cfg["dt_ms"])
        prehistory, events = events[:history_steps], events[history_steps:]
        established = (
            initial_state(built.model, prehistory, amplitude, cfg)
            if history_steps
            else None
        )
        if established is not None:
            save_runtime_state(run.export / "artificial-initial-state", established)
        t = time.perf_counter()
        result = simulate(
            graph=bundle.graph,
            seed=cfg["seed"],
            device="cpu",
            checkpoint=state,
            runtime_state=established,
            input_bindings=(
                DenseArrayBinding("drive", drive),
                DenseArrayBinding(
                    "artificial", torch.from_numpy(events[:, None, :]).float()
                ),
            ),
            snapshot=SnapshotSpec(
                run.export / "execution/start", stage="start", parameters=True
            ),
            performance=Performance(compile_mode=None),
            diagnostics=False,
            retain_parameters=True,
            retain_runtime_state=False,
            retain_final_state=False,
        )
        sim_seconds = time.perf_counter() - t
        for key, w in expected.items():
            assert torch.equal(result.parameters[key], w), key
        arrays = {k: v.detach().cpu().numpy() for k, v in result.outputs.items()}
        for v in arrays.values():
            assert np.isfinite(v).all()
        g = arrays["g_artificial"][:, 0, :].astype(float)
        filter_error = float(np.max(np.abs(g - predicted)))
        mean_error = float(np.max(np.abs(g.mean(0) / target.mean(0) - 1)))
        sd_error = float(np.max(np.abs(g.std(0) / target.std(0) - 1)))
        assert filter_error < 1e-5
        assert not arrays["I_spikes"].any()
        matched = measure(events, g)
        reference = measure(data["ping"]["spk_i"], target)
        assert abs(matched["pairwise_conductance_correlation"]) < 0.1, matched
        np.savez_compressed(run.export / "recording.npz", **arrays)
        np.savez_compressed(
            run.export / "calibration.npz",
            events=events,
            prehistory_events=prehistory,
            amplitude_us=amplitude,
            poisson_rate_hz=rates,
            expected_mean_us=expected_mu,
            expected_sd_us=expected_sd,
            target_mean_us=target.mean(0),
            target_sd_us=target.std(0),
        )
        summary = dict(
            configuration=cfg,
            ping=reference,
            artificial=matched,
            stationary_expected=dict(
                conductance_mean_us=float(expected_mu.mean()),
                mean_per_cell_temporal_sd_us=float(expected_sd.mean()),
                source_rate_mean_hz=float(rates.mean()),
                source_rate_min_hz=float(rates.min()),
                source_rate_max_hz=float(rates.max()),
            ),
            E_rate_hz=float(arrays["E_spikes"].sum() / 1024 / 0.2),
            checks=dict(
                poisson_counts_unconditioned=True,
                frozen_parameters_exact=True,
                same_sensory_input=True,
                mean_relative_error_max=mean_error,
                sd_relative_error_max=sd_error,
                gaba_filter_max_error_us=filter_error,
            ),
            simulation_seconds=sim_seconds,
            total_seconds=time.perf_counter() - tick,
            original_control_E_rate_hz=original["rate_e_hz"],
        )
        (run.export / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        run.record["scientific_artifacts"] = {
            "snapshot_stages": {"execution": ["start"]}
        }
        run.record["execution"].update(
            no_simulation=False,
            simulation_seconds=sim_seconds,
            elapsed_seconds=summary["total_seconds"],
        )
        print(json.dumps(summary, indent=2), flush=True)
    for source in (old, bank):
        source.check_unchanged()
    print("COMPLETE", run.run_id, flush=True)
    return run.run_id


def import_retained(identity, *, run_id=None):
    """Import immutable scratch evidence without executing its historical code."""
    tick = time.perf_counter()
    source = source_run(
        ROOT / ".pingstore",
        identity,
        experiment="exp169",
        stage="compute",
        phase="scratch-count-matched-inhibition",
    )
    pin = source.record["inputs"]["bank"]
    bank = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        experiment="exp022",
        stage="compute",
        reference=pin,
    )
    cfg = source.record["execution"]["configuration"]
    with stage_run(
        ROOT,
        recipe.SLUG,
        "compute",
        inputs={"retained": source, "bank": bank},
        configuration=cfg,
        operation="import",
        run_id=run_id,
    ) as run:
        for item in source.outputs.iterdir():
            target = run.export / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copyfile(item, target)
        for item in source.outputs.rglob("*"):
            if item.is_file():
                assert file_sha256(item) == file_sha256(
                    run.export / item.relative_to(source.outputs)
                )
        run.record["scientific_artifacts"] = source.record["scientific_artifacts"]
        run.record["migration"] = {
            "source": source.reference,
            "method": "byte-identical copy of retained export; no simulation",
            "original_scientific_execution": source.record.get(
                "scientific_execution", source.record["execution"]
            ),
            "original_configuration_preserved": True,
            "historical_scripts": "Retained root Python files are historical provenance, not experiment entry points.",
        }
        run.record["execution"].update(
            no_simulation=True, elapsed_seconds=time.perf_counter() - tick
        )
        source.check_unchanged()
        bank.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


def record_traces(identity, *, run_id=None):
    """Replay exactly the retained artificial trial, recording one fixed E cell.

    Retain one trace archive (<100 kB), native bundle, diagram and start snapshot
    (approximately 8 MB), not another copy of the source recordings.
    """
    from snnlab.sim import NetworkSnapshot

    started = time.perf_counter()
    torch.set_num_threads(1)
    computed = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="compute"
    )
    pin = computed.record["inputs"]["bank"]
    bank = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        experiment="exp022",
        stage="compute",
        reference=pin,
    )
    cfg = recipe.recorded_configuration(
        computed.record["execution"]["configuration"],
        json.loads(computed.file("network.bundle", "graph.json").read_text()),
    )
    neuron = 512
    cfg.update(
        trace_neuron_index=neuron, trace_selection="fixed index, not output-selected"
    )
    bundle = recipe.author_network(cfg, record_traces=True)
    original = recipe.author_network(cfg)
    with stage_run(
        ROOT,
        recipe.SLUG,
        "compute",
        phase="membrane-traces",
        inputs={"compute": computed, "bank": bank},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        retained = lang.load_bundle(
            computed.restore_bundle("network.bundle", run.scratch / "source.bundle")
        )
        if original.graph != retained.graph:
            raise ValueError("Recipe differs from retained trial")
        bundle.write(run.export / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        built = build(
            graph=bundle.graph,
            seed=cfg["seed"],
            device="cpu",
            performance=Performance(compile_mode=None),
        )
        parameters = NetworkSnapshot.read(
            computed.unit("execution", "start"),
            parameters=True,
        ).load_parameters()
        with torch.no_grad():
            for name, parameter in built.model.parameter_map().items():
                parameter.copy_(torch.as_tensor(parameters[name]))
                parameter.requires_grad_(False)
        checkpoint = run.scratch / "parameters.pt"
        torch.save(built.model.state_dict(), checkpoint)
        with np.load(bank.file("snapshots", "coba__off__seed42", "recording.npz")) as z:
            drive = torch.from_numpy(z["input_spikes"][:, None, :].copy()).float()
        with np.load(computed.file("calibration.npz")) as z:
            events = z["events"].copy()
            established = (
                initial_state(
                    built.model,
                    z["prehistory_events"].copy(),
                    z["amplitude_us"].copy(),
                    cfg,
                )
                if cfg.get("prehistory_ms", 0)
                else None
            )
        t = time.perf_counter()
        result = simulate(
            graph=bundle.graph,
            seed=cfg["seed"],
            device="cpu",
            checkpoint=checkpoint,
            runtime_state=established,
            input_bindings=(
                DenseArrayBinding("drive", drive),
                DenseArrayBinding(
                    "artificial", torch.from_numpy(events[:, None, :]).float()
                ),
            ),
            snapshot=SnapshotSpec(
                run.export / "execution/start", stage="start", parameters=True
            ),
            performance=Performance(compile_mode=None),
            diagnostics=False,
            retain_parameters=True,
            retain_runtime_state=False,
            retain_final_state=False,
        )
        simulation_seconds = time.perf_counter() - t
        arrays = {k: v.detach().cpu().numpy() for k, v in result.outputs.items()}
        with np.load(computed.file("recording.npz")) as z:
            for key in z.files:
                np.testing.assert_array_equal(arrays[key], z[key], err_msg=key)
        for name, parameter in result.parameters.items():
            np.testing.assert_array_equal(parameter.numpy(), parameters[name])
        traces = {
            "voltage_mv": arrays["E_voltage"][:, 0, neuron],
            "g_excitatory_us": arrays["g_excitatory"][:, 0, neuron],
            "g_inhibitory_us": (
                arrays["g_recurrent_inhibitory"] + arrays["g_artificial"]
            )[:, 0, neuron],
        }
        if not all(np.isfinite(value).all() for value in traces.values()):
            raise ValueError("Nonfinite diagnostic recording")
        np.savez_compressed(run.export / "traces.npz", **traces)
        (run.export / "summary.json").write_text(
            json.dumps(
                dict(
                    neuron_index=neuron,
                    original_outputs_exact=True,
                    frozen_parameters_exact=True,
                    simulation_seconds=simulation_seconds,
                ),
                indent=2,
            )
            + "\n"
        )
        run.record["scientific_artifacts"] = {
            "snapshot_stages": {"execution": ["start"]}
        }
        run.record["execution"].update(
            no_simulation=False,
            simulation_seconds=simulation_seconds,
            elapsed_seconds=time.perf_counter() - started,
        )
        computed.check_unchanged()
        bank.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-source")
    parser.add_argument(
        "--record-traces", help="Replay a retained compute trial without recalibration"
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.record_traces:
        record_traces(args.record_traces, run_id=args.run_id)
    elif args.import_source:
        import_retained(args.import_source, run_id=args.run_id)
    else:
        compute(run_id=args.run_id)
