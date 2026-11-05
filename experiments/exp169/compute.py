"""Calibrate exogenous GABA inputs and simulate one frozen COBA probe."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
import torch
from experiments.exp169 import recipe
from experiments.exp169.custom.prehistory import initial_state
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


def calibrate(g, cfg):
    """Fit each independent stream's moments without observing E responses."""
    mu = g.mean(0)
    sd = g.std(0)
    target_cv = sd / mu
    rng = np.random.default_rng(cfg["artificial_seed"])
    decay = np.exp(-cfg["dt_ms"] / cfg["tau_gaba_ms"])
    history_steps = round(cfg.get("prehistory_ms", 0) / cfg["dt_ms"])

    def fit(indices):
        uniforms = rng.random((history_steps + len(g), len(indices)))

        def filtered(p):
            events = (uniforms < p[None, :]).astype(np.float64)
            delayed = np.vstack([np.zeros((1, len(indices))), events[:-1]])
            return events, lfilter([1.0], [1.0, -decay], delayed, axis=0)[
                history_steps:
            ]

        low = np.full(len(indices), cfg["calibration_probability_bounds"][0])
        high = np.full(len(indices), cfg["calibration_probability_bounds"][1])
        best = np.full(len(indices), np.inf)
        bestp = low.copy()
        for _ in range(cfg["calibration_bisections"]):
            p = (low + high) / 2
            _, trace = filtered(p)
            cv = trace.std(0) / np.maximum(trace.mean(0), 1e-20)
            err = np.abs(cv / target_cv[indices] - 1)
            improved = err < best
            bestp[improved] = p[improved]
            best[improved] = err[improved]
            low = np.where(cv > target_cv[indices], p, low)
            high = np.where(cv <= target_cv[indices], p, high)
        events, trace = filtered(bestp)
        amplitudes = mu[indices] / trace.mean(0)
        return events, amplitudes, bestp, trace * amplitudes, best

    events, amplitudes, probabilities, conductance, error = fit(np.arange(g.shape[1]))
    # Resample only streams whose finite-window variance cannot meet the moment
    # target closely with a single amplitude. No postsynaptic activity is used.
    for _attempt in range(cfg["calibration_retries"]):
        indices = np.flatnonzero(error > cfg["calibration_sd_target"])
        if not len(indices):
            break
        ev, amp, prob, trace, err = fit(indices)
        accepted = err < error[indices]
        chosen = indices[accepted]
        events[:, chosen] = ev[:, accepted]
        amplitudes[chosen] = amp[accepted]
        probabilities[chosen] = prob[accepted]
        conductance[:, chosen] = trace[:, accepted]
        error[chosen] = err[accepted]
    return (
        events.astype(np.float32),
        amplitudes.astype(np.float32),
        probabilities,
        conductance,
    )


from experiments.exp169.analyse import metrics, onset_metrics


def compute(*, run_id=None):
    tick = time.perf_counter()
    torch.set_num_threads(1)
    cfg = recipe.configuration()
    source = source_run(
        ROOT / ".pingstore",
        cfg["source_run"],
        stage="compute",
        experiment="exp022",
        reference={"run_id": cfg["source_run"], "payload_digest": cfg["source_digest"]},
    )
    data = {}
    for key, cell in [("coba", cfg["coba_cell"]), ("ping", cfg["ping_cell"])]:
        with np.load(source.file("snapshots", cell, "recording.npz")) as z:
            data[key] = {
                k: z[k].copy() for k in ("spk_e", "spk_i", "input_spikes", "gi_e_1")
            }
        config = json.loads(source.file(cell, "config.json").read_text())
        assert (
            config["dt"] == cfg["dt_ms"] and config["tau_gaba_ms"] == cfg["tau_gaba_ms"]
        )
        assert config["fr_reg_upper_strength"] == 0 and config["max_samples"] == 7000
    assert np.array_equal(data["coba"]["input_spikes"], data["ping"]["input_spikes"])
    cp = source.record["bank_reuse"]["diagnostic_provenance"][cfg["coba_cell"]][
        "checkpoint"
    ]
    checkpoint = source.file(cfg["coba_cell"], cp["filename"])
    assert cp["role"] == cfg["checkpoint_role"] and cp["sha256"] == file_sha256(
        checkpoint
    )
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    target = data["ping"]["gi_e_1"].astype(np.float64)
    ct = time.perf_counter()
    events, amplitudes, probability, expected_g = calibrate(target, cfg)
    calibration_seconds = time.perf_counter() - ct
    proposed = metrics(expected_g, target)
    print("CALIBRATION", json.dumps(proposed), flush=True)
    assert (
        proposed["per_cell_std_relative_error_max"] < cfg["std_relative_tolerance"]
    ), proposed
    cfg.update(
        checkpoint=cp,
        source_script_sha256={
            str(p.relative_to(Path(__file__).parent)): file_sha256(p)
            for p in Path(__file__).parent.rglob("*.py")
        },
        calibration_seconds=calibration_seconds,
    )
    bundle = recipe.author_network(cfg)
    mapping = {
        "input_to_E.weight": "W_ff.0",
        "E_to_I.weight": "W_ei.1",
        "I_to_E.weight": "W_ie.1",
        "readout_projection.weight": "W_ff.1",
    }
    with stage_run(
        ROOT,
        recipe.SLUG,
        "compute",
        inputs={"bank": source},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        bundle.write(run.export / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        for p in Path(__file__).parent.rglob("*.py"):
            destination = run.export / p.relative_to(Path(__file__).parent)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(p.read_bytes())
        built = build(
            graph=bundle.graph,
            seed=cfg["seed"],
            device="cpu",
            performance=Performance(compile_mode=None),
        )
        expected = {
            name: weights[key].clamp(min=0).clone() for name, key in mapping.items()
        }
        expected["artificial_to_E.weight"] = torch.diag(torch.from_numpy(amplitudes))
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
        established = initial_state(built.model, prehistory, amplitudes, cfg)
        save_runtime_state(run.export / "artificial-initial-state", established)
        results = {}
        sim_seconds = 0.0
        for condition, ev in [
            ("baseline", np.zeros_like(events)),
            ("zero_start", events),
            ("artificial", events),
        ]:
            t = time.perf_counter()
            result = simulate(
                graph=bundle.graph,
                seed=cfg["seed"],
                device="cpu",
                checkpoint=state,
                runtime_state=established if condition == "artificial" else None,
                input_bindings=(
                    DenseArrayBinding("drive", drive),
                    DenseArrayBinding("artificial", torch.from_numpy(ev[:, None, :])),
                ),
                snapshot=SnapshotSpec(
                    run.export / condition / "start", stage="start", parameters=True
                ),
                performance=Performance(compile_mode=None),
                diagnostics=False,
                retain_parameters=True,
                retain_runtime_state=False,
                retain_final_state=False,
            )
            elapsed = time.perf_counter() - t
            sim_seconds += elapsed
            for key, w in expected.items():
                assert torch.equal(result.parameters[key], w), key
            arrays = {k: v.detach().cpu().numpy() for k, v in result.outputs.items()}
            for k, v in arrays.items():
                assert np.isfinite(v).all(), k
            print(
                "OUTPUTS",
                condition,
                {k: v.shape for k, v in arrays.items()},
                "seconds",
                elapsed,
                flush=True,
            )
            results[condition] = arrays
            np.savez_compressed(run.export / (condition + ".npz"), **arrays)
        baseline = results["baseline"]["E_spikes"][:, 0, :]
        agreement = float(np.mean(baseline == data["coba"]["spk_e"]))
        count_replay_exact = bool(
            np.array_equal(baseline.sum(0), data["coba"]["spk_e"].sum(0))
        )
        assert count_replay_exact and agreement > 0.9999, (
            "native baseline differs",
            agreement,
        )
        replay_mismatches = int(np.count_nonzero(baseline != data["coba"]["spk_e"]))
        realised = results["artificial"]["g_artificial"][:, 0, :].astype(float)
        match = metrics(realised, target)
        assert (
            match["per_cell_mean_relative_error_max"] < cfg["mean_relative_tolerance"]
        ), match
        assert (
            match["per_cell_std_relative_error_max"] < cfg["std_relative_tolerance"]
        ), match
        assert abs(match["pairwise_correlation_mean"]) < 0.1, match
        assert match["population_mean_temporal_std"] < target.mean(1).std() * 0.2, match
        np.savez_compressed(
            run.export / "calibration.npz",
            events=events.astype(np.uint8),
            amplitudes_us=amplitudes,
            event_probability=probability,
            target_mean_us=target.mean(0),
            target_std_us=target.std(0),
            prehistory_events=prehistory.astype(np.uint8),
        )
        summary = dict(
            configuration=cfg,
            baseline_replay_exact=bool(agreement == 1),
            baseline_per_neuron_counts_exact=count_replay_exact,
            baseline_spike_bin_agreement=agreement,
            baseline_mismatched_spike_bins=replay_mismatches,
            frozen_parameters_exact=True,
            matched=match,
            ping=metrics(target, target),
            rate_e_hz=float(results["artificial"]["E_spikes"].sum() / 1024 / 0.2),
            artificial_event_rate_hz=float(events.mean() / (cfg["dt_ms"] / 1000)),
            simulation_seconds=sim_seconds,
            total_seconds=time.perf_counter() - tick,
            onset_comparison={
                condition: onset_metrics(a, cfg)
                for condition, a in results.items()
                if condition != "baseline"
            },
            limitations="Single illustrative image and calibrated random draw. Same-image PING targets retained. Inhibition-only prehistory removes the zero-conductance ramp; neuronal and sensory onset dynamics may remain. Zero-start comparison uses identical scored events and amplitudes, so its full-window moments need not match PING. No retraining. Native baseline preserves original per-cell counts with small legacy timing differences.",
        )
        (run.export / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        run.record["scientific_artifacts"] = {
            "snapshot_stages": {c: ["start"] for c in results}
        }
        run.record["execution"].update(
            no_simulation=False,
            simulation_seconds=sim_seconds,
            elapsed_seconds=summary["total_seconds"],
        )
        print("SUMMARY", json.dumps(summary), flush=True)
    source.check_unchanged()
    print("COMPLETE", run.run_id, flush=True)
    return run.run_id


def import_retained(identity, *, run_id=None):
    """Import immutable scratch evidence without executing its historical code."""
    tick = time.perf_counter()
    source = source_run(
        ROOT / ".pingstore",
        identity,
        experiment="exp022",
        stage="compute",
        phase="scratch-artificial-inhibition",
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
    cfg = dict(computed.record["execution"]["configuration"])
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
            computed.unit("artificial", "start"),
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
                    z["amplitudes_us"].copy(),
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
        with np.load(computed.file("artificial.npz")) as z:
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--import-source",
        help="Explicit retained scratch compute ID; copies evidence without simulation",
    )
    p.add_argument(
        "--record-traces", help="Replay a retained compute trial without recalibration"
    )
    p.add_argument("--run-id")
    args = p.parse_args()
    if args.record_traces:
        record_traces(args.record_traces, run_id=args.run_id)
    elif args.import_source:
        import_retained(args.import_source, run_id=args.run_id)
    else:
        compute(run_id=args.run_id)
