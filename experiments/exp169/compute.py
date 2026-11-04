"""Calibrate exogenous GABA inputs and simulate one frozen COBA probe."""

import argparse
import csv
import hashlib
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
from pingstore.library import file_sha256, source_run, stage_run
from scipy.signal import lfilter
from snnlab.sim import (
    DenseArrayBinding,
    Performance,
    SnapshotSpec,
    build,
    encode_images_poisson,
    load_dataset,
    simulate,
)


def calibrate(g, cfg):
    """Fit each independent stream's moments without observing E responses."""
    mu = g.mean(0)
    sd = g.std(0)
    target_cv = sd / mu
    rng = np.random.default_rng(cfg["artificial_seed"])
    decay = np.exp(-cfg["dt_ms"] / cfg["tau_gaba_ms"])

    def fit(indices):
        uniforms = rng.random((len(g), len(indices)))

        def filtered(p):
            events = (uniforms < p[None, :]).astype(np.float64)
            delayed = np.vstack([np.zeros((1, len(indices))), events[:-1]])
            return events, lfilter([1.0], [1.0, -decay], delayed, axis=0)

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


from experiments.exp169.analyse import metrics


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
            p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")
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
        for p in Path(__file__).parent.glob("*.py"):
            (run.export / p.name).write_bytes(p.read_bytes())
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
        results = {}
        sim_seconds = 0.0
        for condition, ev in [
            ("baseline", np.zeros_like(events)),
            ("artificial", events),
        ]:
            t = time.perf_counter()
            result = simulate(
                graph=bundle.graph,
                seed=cfg["seed"],
                device="cpu",
                checkpoint=state,
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
            limitations="Native baseline has identical per-neuron counts but 40 differing spike bins versus the saved legacy recording (20 shifted spikes). Single illustrative image and random draw. Event probabilities and weights fitted to PING conductance moments only, not E activity. Independent inputs have incidental finite-sample correlations and a common zero-conductance onset. This does not match voltage-dependent inhibitory current or higher-order temporal statistics.",
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


def classification(*, run_id=None):
    """Evaluate paired official-test images with the three frozen conditions."""
    cfg = recipe.configuration()
    bank = source_run(
        ROOT / ".pingstore",
        cfg["source_run"],
        stage="compute",
        experiment="exp022",
        reference={"run_id": cfg["source_run"], "payload_digest": cfg["source_digest"]},
    )
    print(cfg["classification_retention"], flush=True)
    with stage_run(
        ROOT,
        recipe.SLUG,
        "compute",
        inputs={"bank": bank},
        configuration=cfg,
        phase="classification",
        run_id=run_id,
    ) as run:
        _classification(run, bank, cfg)
    print("COMPLETE", run.run_id, flush=True)
    return run.run_id


def _classification(run, bank, cfg):
    tick = time.perf_counter()
    torch.set_num_threads(1)
    data = run.export
    snapshot_roots = {}
    moment_arrays = []
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    seconds = cfg["duration_ms"] / 1000
    early_steps = round(20.0 / cfg["dt_ms"])
    if cfg["classification_burn_in_ms"] != 0:
        raise ValueError("This protocol preserves zero burn-in")
    conditions = ["COBA", "PING", "COBA + artificial inhibition"]
    mapping = {
        "input_to_E.weight": "W_ff.0",
        "E_to_I.weight": "W_ei.1",
        "I_to_E.weight": "W_ie.1",
        "readout_projection.weight": "W_ff.1",
    }
    params, pins = {}, {}
    for kind in ["coba", "ping"]:
        cell = cfg[kind + "_cell"]
        conf = json.loads(bank.file(cell, "config.json").read_text())
        assert (
            conf["readout_mode"] == "mem-mean"
            and conf["input_rate"] == cfg["classification_input_rate_hz"]
        )
        assert conf["dt"] == cfg["dt_ms"] and conf["t_ms"] == cfg["duration_ms"]
        assert conf["fr_reg_upper_strength"] == 0 and conf["max_samples"] == 7000
        pin = bank.record["bank_reuse"]["diagnostic_provenance"][cell]["checkpoint"]
        path = bank.file(cell, pin["filename"])
        assert (
            pin["role"] == cfg["checkpoint_role"] and file_sha256(path) == pin["sha256"]
        )
        w = torch.load(path, map_location="cpu", weights_only=True)
        params[kind] = {k: w[v].clamp(min=0).clone() for k, v in mapping.items()}
        params[kind]["artificial_to_E.weight"] = torch.zeros(cfg["n_e"], cfg["n_e"])
        pins[kind] = pin
    bundle = recipe.author_network(cfg, record_ping=True)
    bundle.write(data / "network.bundle", visualise=False)
    bundle.visualise(data / "circuit.svg", view="circuit")
    built = build(
        graph=bundle.graph,
        seed=cfg["seed"],
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    assert set(built.model.parameter_map()) == set(params["coba"])
    totals = dict.fromkeys(conditions, 0.0)
    calibration_seconds = 0.0

    # Every simulate call resets state. Only weights change between conditions.
    def infer(kind, drive, events, amplitude, snapshot):
        expected = dict(params[kind])
        if amplitude is not None:
            expected["artificial_to_E.weight"] = torch.diag(torch.from_numpy(amplitude))
        with torch.no_grad():
            for k, p in built.model.parameter_map().items():
                p.copy_(expected[k])
                p.requires_grad_(False)
        state = run.scratch / "execution-parameters.pt"
        torch.save(built.model.state_dict(), state)
        snapshot_roots["snapshots/" + snapshot] = ["start"]
        t = time.perf_counter()
        result = simulate(
            graph=bundle.graph,
            seed=cfg["seed"],
            device="cpu",
            checkpoint=state,
            input_bindings=(
                DenseArrayBinding("drive", drive),
                DenseArrayBinding(
                    "artificial", torch.from_numpy(events[:, None, :]).float()
                ),
            ),
            snapshot=SnapshotSpec(
                data / "snapshots" / snapshot / "start", stage="start", parameters=True
            ),
            performance=Performance(compile_mode=None),
            diagnostics=False,
            retain_parameters=True,
            retain_runtime_state=False,
            retain_final_state=False,
        )
        elapsed = time.perf_counter() - t
        for k, w in expected.items():
            assert torch.equal(result.parameters[k], w), ("changed weight", k)
        arrays = {k: v.detach().cpu().numpy() for k, v in result.outputs.items()}
        assert all(np.isfinite(v).all() for v in arrays.values())
        return arrays, elapsed

    zero = np.zeros((steps, cfg["n_e"]), np.float32)
    replay = {}
    for kind in ["coba", "ping"]:
        with np.load(bank.file("snapshots", cfg[kind + "_cell"], "recording.npz")) as z:
            drive = torch.from_numpy(z["input_spikes"][:, None, :]).float()
            a, sec = infer(kind, drive, zero, None, "replay-" + kind)
            spikes = a["E_spikes"][:, 0, :]
            counts_exact = bool(np.array_equal(spikes.sum(0), z["spk_e"].sum(0)))
            agreement = float((spikes == z["spk_e"]).mean())
            replay[kind] = {
                "per_neuron_counts_exact": counts_exact,
                "spike_bin_agreement": agreement,
                "seconds": sec,
            }
            assert counts_exact and agreement > 0.9999, replay
    print("REPLAY", json.dumps(replay), flush=True)
    _, pixels, _, labels = load_dataset(
        "mnist", split=True, evaluation_only=True, evaluation_split="test"
    )
    rng = np.random.default_rng(cfg["classification_selection_seed"])
    indices = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(labels == d),
                cfg["classification_images_per_digit"],
                replace=False,
            )
            for d in range(10)
        ]
    )
    assert len(np.unique(indices)) == 10 * cfg["classification_images_per_digit"]
    selection = [
        {
            "index": int(idx),
            "label": int(labels[idx]),
            "encoding_seed": cfg["classification_encoding_seed"] + int(idx),
            "artificial_seed": cfg["artificial_seed"] + int(idx),
        }
        for idx in indices
    ]
    (data / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    cfg["checkpoint_pins"] = pins
    cfg["replay"] = replay
    rows = []
    checks = []
    with (data / "predictions.csv").open("w", newline="") as handle:
        fields = [
            "index",
            "label",
            "condition",
            "prediction",
            "correct",
            "top_score_tied",
            "rate_e_hz",
            "early_rate_e_hz",
            "late_rate_e_hz",
            "input_sha256",
        ] + ["score_" + str(i) for i in range(10)]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for number, item in enumerate(selection):
            idx = item["index"]
            label = item["label"]
            drive = encode_images_poisson(
                torch.from_numpy(pixels[idx : idx + 1]),
                steps,
                cfg["dt_ms"],
                cfg["classification_input_rate_hz"],
                generator=torch.Generator().manual_seed(item["encoding_seed"]),
            )
            digest = hashlib.sha256(drive.numpy().tobytes()).hexdigest()
            ping, sec = infer("ping", drive, zero, None, f"{number:03d}-ping")
            totals[conditions[1]] += sec
            target = ping["g_ping"][:, 0, :].astype(np.float64)
            assert np.all(target.mean(0) > 0) and np.all(target.std(0) > 0)
            t = time.perf_counter()
            local_cfg = dict(cfg, artificial_seed=item["artificial_seed"])
            events, amps, prob, _ = calibrate(target, local_cfg)
            calibration_seconds += time.perf_counter() - t
            coba, sec = infer("coba", drive, zero, None, f"{number:03d}-coba")
            totals[conditions[0]] += sec
            artificial, sec = infer(
                "coba", drive, events, amps, f"{number:03d}-artificial"
            )
            totals[conditions[2]] += sec
            realised = artificial["g_artificial"][:, 0, :].astype(np.float64)
            mean_err = float(np.max(np.abs(realised.mean(0) / target.mean(0) - 1)))
            sd_err = float(np.max(np.abs(realised.std(0) / target.std(0) - 1)))
            check = {
                "index": idx,
                "max_mean_relative_error": mean_err,
                "max_sd_relative_error": sd_err,
                "passed": mean_err < cfg["mean_relative_tolerance"]
                and sd_err < cfg["std_relative_tolerance"],
                "event_probability_min": float(prob.min()),
                "event_probability_max": float(prob.max()),
            }
            checks.append(check)
            moment_arrays.append(
                np.stack(
                    [
                        target.mean(0),
                        target.std(0),
                        realised.mean(0),
                        realised.std(0),
                        amps,
                        prob,
                    ]
                )
            )
            for condition, a in zip(conditions, [coba, ping, artificial], strict=True):
                scores = a["scores"][0]
                assert scores.shape == (10,)
                pred = int(scores.argmax())
                e = a["E_spikes"][:, 0, :]
                row = {
                    "index": idx,
                    "label": label,
                    "condition": condition,
                    "prediction": pred,
                    "correct": int(pred == label),
                    "top_score_tied": int((scores == scores.max()).sum() > 1),
                    "rate_e_hz": float(e.sum() / cfg["n_e"] / seconds),
                    "early_rate_e_hz": float(e[:early_steps].sum() / cfg["n_e"] / 0.02),
                    "late_rate_e_hz": float(
                        e[early_steps:].sum() / cfg["n_e"] / (seconds - 0.02)
                    ),
                    "input_sha256": digest,
                }
                row.update({"score_" + str(i): float(scores[i]) for i in range(10)})
                writer.writerow(row)
                rows.append(row)
            handle.flush()
            (data / "calibration.json").write_text(json.dumps(checks, indent=2) + "\n")
            print(
                "PROGRESS",
                number + 1,
                f"/{len(selection)}",
                "correct",
                {
                    c: sum(r["correct"] for r in rows if r["condition"] == c)
                    for c in conditions
                },
                "calibration_pass",
                check["passed"],
                "elapsed_s",
                round(time.perf_counter() - tick, 1),
                flush=True,
            )
    np.savez_compressed(
        data / "calibration-moments.npz", moments=np.stack(moment_arrays)
    )
    cfg["replay"] = replay
    run.record["scientific_artifacts"] = {"snapshot_stages": snapshot_roots}
    run.record["execution"].update(
        no_simulation=False,
        simulation_seconds=sum(totals.values())
        + sum(r["seconds"] for r in replay.values()),
        calibration_seconds=calibration_seconds,
        inference_seconds=totals,
        elapsed_seconds=time.perf_counter() - tick,
        frozen_parameters_exact=True,
    )
    bank.check_unchanged()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--import-source",
        help="Explicit retained scratch compute ID; copies evidence without simulation",
    )
    p.add_argument(
        "--classification",
        action="store_true",
        help="Run the paired 100-image frozen-readout evaluation",
    )
    p.add_argument("--run-id")
    args = p.parse_args()
    if args.import_source and args.classification:
        p.error("--import-source and --classification are mutually exclusive")
    if args.classification:
        classification(run_id=args.run_id)
    elif args.import_source:
        import_retained(args.import_source, run_id=args.run_id)
    else:
        compute(run_id=args.run_id)
