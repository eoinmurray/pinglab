"""Run frozen TR02 (no rate penalty) pool-size conditions locally with native evidence."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
import torch
from experiments.exp173 import recipe
from pingstore.library import file_sha256, source_run, stage_run
from snnlab import sim


def load_source(cfg):
    source = source_run(
        ROOT / ".pingstore",
        cfg["source_run"],
        experiment="exp022",
        stage="compute",
        reference={
            "run_id": cfg["source_run"],
        },
    )
    cell = cfg["source_cell"]
    conf = json.loads(source.file(cell, "config.json").read_text())
    provenance = source.details["bank_reuse"]["diagnostic_provenance"][cell]
    pin = provenance["checkpoint"]
    path = source.file(cell, pin["filename"])
    if (
        pin["role"] != "final_epoch"
        or pin["epoch"] != conf["epochs"]
        or file_sha256(path) != pin["sha256"]
    ):
        raise ValueError("Source checkpoint provenance mismatch")
    expected = dict(
        training_run_id="TR-02",
        fr_reg_upper_strength=0.0,
        seed=42,
        n_hidden=1024,
        n_inh=256,
        n_in=784,
        n_out=10,
        readout_mode="mem-mean",
        dales_law=True,
    )
    if any(conf.get(k) != v for k, v in expected.items()):
        raise ValueError(
            "Source is not the expected TR02 (no rate penalty) architecture"
        )
    if any(
        conf.get(k, False)
        for k in (
            "adaptive_threshold",
            "train_leak",
            "signed_readout",
            "readout_bias",
            "state_clamp",
        )
    ):
        raise ValueError("Unsupported source dynamics")
    diagnostic = provenance["execution_records"]["config.json"]
    if diagnostic["refractory_policy"] != "exact" or diagnostic["dt"] != conf["dt"]:
        raise ValueError("Unsupported diagnostic timing")
    cfg.update(
        source_checkpoint=pin,
        source_training=conf,
        dt_ms=conf["dt"],
        duration_ms=diagnostic["realized_duration_ms"],
        steps=diagnostic["duration_steps"],
        n_input=conf["n_in"],
        n_e=conf["n_hidden"],
        n_output=conf["n_out"],
        input_rate_hz=conf["input_rate"],
        tau_ampa_ms=conf["tau_ampa_ms"],
        tau_gaba_ms=conf["tau_gaba_ms"],
        encoder_seeds=conf["validation_encoder_draws"]["encoder_seeds"],
        readout_tau_ms=2.0,
        readout_threshold=1.0,
        surrogate_slope=conf["surrogate_slope"],
        dynamics_provenance="exp022 retained diagnostic exact refractory settings; canonical COBA-LIF/readout biophysics",
    )
    cfg["neurons"] = {}
    for name, c, leak, ref_key in (
        ("E", 1.0, 0.05, "refractory_e_ms"),
        ("I", 0.5, 0.1, "refractory_i_ms"),
    ):
        cfg["neurons"][name] = dict(
            tau_mem_ms=c / leak,
            capacitance_nf=c,
            leak_us=leak,
            resting_mv=-65.0,
            threshold_mv=-50.0,
            reset_mv=-65.0,
            initial_voltage_mv=-65.0,
            refractory_ms=diagnostic[ref_key],
            voltage_grad_dampen=conf["v_grad_dampen"],
        )
    weights = torch.load(path, map_location="cpu", weights_only=True)
    shapes = {
        "W_ff.0": (784, 1024),
        "W_ff.1": (1024, 10),
        "W_ei.1": (1024, 256),
        "W_ie.1": (256, 1024),
        "W_ee.1": (1024, 1024),
        "W_ii.1": (256, 256),
    }
    if set(weights) != set(shapes):
        raise ValueError("Unexpected checkpoint matrices")
    for key, shape in shapes.items():
        value = weights[key]
        if (
            tuple(value.shape) != shape
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid weights: {key}")
        if key in ("W_ee.1", "W_ii.1") and value.count_nonzero():
            raise ValueError("Unexpected same-population recurrence")
    # The bank executes nonnegative constrained values, including learned matrices.
    return source, {key: value.clamp_min(0) for key, value in weights.items()}


def checkpoint(bundle, parameters, path, performance):
    built = sim.build(bundle=bundle, device="cpu", seed=42, performance=performance)
    with torch.no_grad():
        for name, value in parameters.items():
            parameter = built.model.parameter_map()[name]
            if parameter.shape != value.shape:
                raise ValueError(f"Unexpected native parameter orientation: {name}")
            parameter.copy_(value)
    torch.save(built.model.state_dict(), path)


def valid_counts(values):
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values != np.rint(values))
    ):
        raise ValueError("Invalid population spike counts")
    return values


def execute(cfg, *, run_id=None):
    started = time.perf_counter()
    torch.set_num_threads(cfg["threads"])
    source, weights = load_source(cfg)
    _, pixels, _, labels = sim.load_dataset(
        "mnist", max_samples=cfg["source_training"]["max_samples"], split=True
    )
    _, accuracy_pixels, _, accuracy_labels = sim.load_dataset(
        "mnist", split=True, evaluation_only=True, evaluation_split="test"
    )
    if accuracy_pixels.shape != (10000, cfg["n_input"]) or accuracy_labels.shape != (
        10000,
    ):
        raise ValueError("Official test partition size mismatch")
    sample_ids = recipe.test_sample_indices(
        total_images=len(accuracy_labels),
        samples=cfg["accuracy_images"],
        seed=cfg["accuracy_sample_seed"],
    )
    accuracy_pixels, accuracy_labels = (
        accuracy_pixels[sample_ids],
        accuracy_labels[sample_ids],
    )
    cfg["accuracy_dataset_sha256"] = hashlib.sha256(
        accuracy_pixels.tobytes() + accuracy_labels.tobytes() + sample_ids.tobytes()
    ).hexdigest()
    probe = int(np.flatnonzero(labels == 0)[0])
    pixels, labels = pixels[probe : probe + 1], labels[probe : probe + 1]
    cfg.update(
        evaluation_images=1,
        encoder_seeds=cfg["encoder_seeds"][:1],
        raster_probe_index=0,
        source_validation_index=probe,
        protocol="single_matched_digit_zero",
        rate_aggregation="single-image, single-draw population mean; same trial as raster",
        encoding_policy="one CPU Poisson draw shared exactly by all pool-size conditions",
    )
    cfg["dataset_sha256"] = hashlib.sha256(
        pixels.tobytes() + labels.tobytes()
    ).hexdigest()
    drive = sim.encode_images_poisson(
        torch.from_numpy(pixels),
        cfg["steps"],
        cfg["dt_ms"],
        cfg["input_rate_hz"],
        generator=torch.Generator().manual_seed(cfg["encoder_seeds"][0]),
    )
    cfg["input_sha256"] = hashlib.sha256(drive.numpy().tobytes()).hexdigest()
    performance = sim.Performance(
        compile_mode=None, precision="highest", projection_mode="fast"
    )
    with stage_run(
        ROOT,
        recipe.SLUG,
        "compute",
        inputs={"bank": source},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        run.details["scientific_artifacts"] = {
            "snapshot_stages": {f"i{n}": ["start"] for n in cfg["pool_sizes"]}
        }
        np.savez_compressed(
            run.export / "dataset.npz",
            labels=labels,
            probe_pixels=pixels[0],
            accuracy_labels=accuracy_labels,
            test_sample_indices=sample_ids,
        )
        timings = {}
        reference_accuracy_hashes = None
        for n_i in cfg["pool_sizes"]:
            tick = time.perf_counter()
            directory = run.export / f"i{n_i}"
            directory.mkdir()
            bundle = recipe.author_network(cfg, n_i)
            bundle.write(directory / "network.bundle", visualise=False)
            bundle.visualise(run.export / f"circuit-i{n_i}.svg", view="circuit")
            params, indices = recipe.resize_weights(weights, n_i)
            np.save(directory / "selected_indices.npy", indices)
            checkpoint_path = run.scratch / f"i{n_i}.pth"
            checkpoint(bundle, params, checkpoint_path, performance)
            events = {p: [] for p in ("E", "I")}

            def collect(block, events=events):
                if block.kind == "spike_events":
                    events[block.signal.split(".")[0]].append(
                        block.values.cpu().numpy()
                    )

            with torch.inference_mode():
                result = sim.simulate(
                    bundle=bundle,
                    checkpoint=checkpoint_path,
                    input_bindings=[sim.DenseArrayBinding("drive", drive)],
                    seed=42,
                    device="cpu",
                    performance=performance,
                    snapshot=sim.SnapshotSpec(
                        directory / "start", stage="start", parameters=True
                    ),
                    diagnostics=False,
                    recording=sim.RecordingSpec(
                        signals=tuple(
                            sim.SignalRecording(p + ".spikes", kind="spike_events")
                            for p in events
                        ),
                        sink=collect,
                        sink_id="exp173.raster/v1",
                        block_steps=cfg["steps"],
                    ),
                    retain_parameters=False,
                    retain_final_state=False,
                    retain_runtime_state=False,
                )
            counts, arrays = {}, {}
            for p, blocks in events.items():
                rows = np.concatenate(blocks) if blocks else np.empty((0, 3))
                values = result.outputs[p + "_count"].sum(-1).cpu().numpy()
                counts[p] = valid_counts(values).astype(np.int64).reshape(1, 1)
                if len(rows) != int(counts[p][0, 0]):
                    raise ValueError("Raster and rate counts differ")
                arrays[p + "_time_ms"] = rows[:, 0] * cfg["dt_ms"]
                arrays[p + "_cell"] = rows[:, 2].astype(np.int32)
            np.savez_compressed(directory / "counts.npz", **counts)
            np.savez_compressed(directory / "raster.npz", **arrays)
            if (
                hashlib.sha256(drive.numpy().tobytes()).hexdigest()
                != cfg["input_sha256"]
            ):
                raise ValueError("Shared input changed")
            predictions, batch_hashes = [], []
            generator = torch.Generator().manual_seed(cfg["accuracy_encoder_seed"])
            with torch.inference_mode():
                for start in range(0, len(accuracy_labels), cfg["accuracy_batch_size"]):
                    batch = accuracy_pixels[start : start + cfg["accuracy_batch_size"]]
                    accuracy_drive = sim.encode_images_poisson(
                        torch.from_numpy(batch),
                        cfg["steps"],
                        cfg["dt_ms"],
                        cfg["input_rate_hz"],
                        generator=generator,
                    )
                    batch_hashes.append(
                        hashlib.sha256(accuracy_drive.numpy().tobytes()).hexdigest()
                    )
                    accuracy_result = result.model(
                        {"drive": accuracy_drive},
                        diagnostics=False,
                        retain_parameters=False,
                        retain_final_state=False,
                        retain_runtime_state=False,
                    )
                    scores = accuracy_result.outputs["class_scores"].cpu().numpy()
                    if (
                        scores.shape != (len(batch), cfg["n_output"])
                        or not np.isfinite(scores).all()
                    ):
                        raise ValueError("Invalid classification scores")
                    predictions.append(scores.argmax(axis=1))
            predictions = np.concatenate(predictions)
            np.savez_compressed(
                directory / "accuracy.npz",
                predictions=predictions,
                correct=predictions == accuracy_labels,
            )
            if reference_accuracy_hashes is None:
                reference_accuracy_hashes = batch_hashes
            elif reference_accuracy_hashes != batch_hashes:
                raise ValueError("Accuracy inputs differ between pool sizes")
            timings[str(n_i)] = time.perf_counter() - tick
            print(
                f"I={n_i}: raster + {len(predictions)} test images; accuracy={(predictions == accuracy_labels).mean() * 100:.1f}%; {timings[str(n_i)]:.2f}s",
                flush=True,
            )
        from pingstore.library import write_json_atomic

        write_json_atomic(
            run.export / "accuracy-input-hashes.json",
            {"batch_sha256": reference_accuracy_hashes},
        )
        run.details["execution"].update(
            elapsed_s=time.perf_counter() - started, condition_seconds=timings
        )
    print(run.run_id, flush=True)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    execute(
        recipe.configuration(source_run=args.source, source_digest=args.source_digest),
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
