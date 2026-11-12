"""Replay three frozen unpenalised models on a matched test-subset prefix."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
import torch
from experiments.exp174 import dataset, recipe
from pingstore.library import file_sha256, source_run, stage_run, write_json_atomic
from snnlab import extensions, lang, sim


def load_sources(cfg):
    source = source_run(
        ROOT / ".pingstore",
        cfg["source_run"],
        experiment="exp171",
        stage="compute",
        reference={
            "run_id": cfg["source_run"],
        },
    )
    bank = source_run(
        ROOT / ".pingstore",
        cfg["bank_run"],
        experiment="exp022",
        stage="compute",
        reference={
            "run_id": cfg["bank_run"],
        },
    )
    retained = source.details["consolidation"]["sources"]["exp171-r047-compute"][
        "record"
    ]["execution"]["configuration"]
    if retained["rate_penalty"] != 0 or retained["rate_target_hz"] is not None:
        raise ValueError("Student source is penalised")
    weights, pins = {}, {}
    for kind in ("coba", "ping"):
        cell = kind + "__off__seed42"
        conf = json.loads(bank.file(cell, "config.json").read_text())
        provenance = bank.details["bank_reuse"]["diagnostic_provenance"][cell]
        pin = provenance["checkpoint"]
        expected = retained["source_checkpoints"][kind]
        if (
            conf["fr_reg_upper_strength"] != 0
            or pin["role"] != "final_epoch"
            or pin["epoch"] != 50
            or pin["sha256"] != expected["sha256"]
        ):
            raise ValueError("Baseline checkpoint does not match exp171")
        diagnostic = provenance["execution_records"]["config.json"]
        if (
            diagnostic["refractory_policy"] != "exact"
            or diagnostic["dt"] != cfg["dt_ms"]
        ):
            raise ValueError("Unsupported baseline dynamics")
        path = bank.file(cell, pin["filename"])
        if file_sha256(path) != pin["sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        weights[kind] = {
            name: raw[key].clamp_min(0) for name, key in recipe.MAPPING.items()
        }
        for key in ("W_ee.1", "W_ii.1"):
            if raw[key].count_nonzero():
                raise ValueError("Unexpected recurrent same-population weights")
        pins[kind] = pin
    if any(
        weights["coba"][key].count_nonzero()
        for key in ("E_to_I.weight", "I_to_E.weight")
    ):
        raise ValueError("COBA feedback must be silent")
    cfg["source_checkpoints"] = pins
    cfg["refractory_timing"] = dict(
        coba="following_steps", ping="following_steps", artificial="legacy_coba"
    )
    return source, bank, weights


def checkpoint(bundle, params, path, performance):
    built = sim.build(bundle=bundle, device="cpu", seed=42, performance=performance)
    if set(params) != set(built.model.parameter_map()):
        raise ValueError("Native parameter membership mismatch")
    with torch.no_grad():
        for name, value in built.model.parameter_map().items():
            source = torch.as_tensor(params[name])
            if source.shape != value.shape or not torch.isfinite(source).all():
                raise ValueError("Invalid native parameter shape or value")
            value.copy_(source)
    torch.save(built.model.state_dict(), path)


class PrefixCollector:
    """Reduce bounded native recording blocks; retain no per-neuron rasters."""

    def __init__(self, steps, samples, outputs):
        self.predictions = np.zeros((steps, samples), np.uint8)
        self.counts = {p: np.zeros((steps, samples), np.int32) for p in ("E", "I")}
        self.score_sum = np.zeros((samples, outputs), np.float64)
        self.seen = {
            s: np.zeros((steps, samples), bool)
            for s in ("E.spikes", "I.spikes", "readout.pre_reset_voltage")
        }

    def __call__(self, block):
        if block.kind == "output":
            return
        values = block.values.detach().cpu().numpy()
        start, end, offset = block.start_step, block.end_step, block.batch_offset
        width = values.shape[1]
        columns = slice(offset, offset + width)
        if (
            values.shape[0] != end - start
            or self.seen[block.signal][start:end, columns].any()
        ):
            raise ValueError("Unexpected recording block coordinates")
        self.seen[block.signal][start:end, columns] = True
        if block.signal == "readout.pre_reset_voltage":
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite readout voltage")
            running = (
                values.astype(np.float64).cumsum(0) + self.score_sum[columns][None]
            )
            self.predictions[start:end, columns] = running.argmax(-1)
            self.score_sum[columns] = running[-1]
        else:
            if not np.isin(values, (0, 1)).all():
                raise ValueError("Nonbinary neural spike recording")
            self.counts[block.signal[0]][start:end, columns] = values.sum(-1).astype(
                np.int32
            )

    def validate(self, result, cfg):
        if not all(mask.all() for mask in self.seen.values()):
            raise ValueError("Incomplete temporal recording")
        scores = result.outputs["scores"].detach().cpu().numpy()
        np.testing.assert_array_equal(self.predictions[-1], scores.argmax(-1))
        np.testing.assert_allclose(
            self.score_sum / cfg["steps"], scores, rtol=2e-4, atol=2e-4
        )
        native = result.outputs["e_count"].detach().cpu().numpy().sum(-1)
        np.testing.assert_array_equal(self.counts["E"].sum(0), native)
        if "i_count" in result.outputs:
            np.testing.assert_array_equal(
                self.counts["I"].sum(0),
                result.outputs["i_count"].detach().cpu().numpy().sum(-1),
            )
        return scores


def execute(cfg, *, run_id=None):
    started = time.perf_counter()
    torch.set_num_threads(cfg["threads"])
    source, bank, weights = load_sources(cfg)
    prefix = cfg["test_origin"] + "--off"
    with np.load(source.file(prefix, "dataset.npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    expected_ids = np.random.RandomState(42).choice(10000, 1000, replace=False)
    np.testing.assert_array_equal(arrays["ids"], expected_ids)
    if arrays["features"].shape != (1000, cfg["n_input"]) or arrays["labels"].shape != (
        1000,
    ):
        raise ValueError("Unexpected test dataset")
    if (
        not np.isfinite(arrays["poisson_rate_hz"]).all()
        or (arrays["poisson_rate_hz"] < 0).any()
    ):
        raise ValueError("Invalid frozen inhibition rates")
    cfg["dataset_sha256"] = {k: dataset.sha(v) for k, v in arrays.items()}
    # Check entire original manuscript batches before slicing the prototype prefix.
    original_test = json.loads(
        source.file(cfg["test_origin"] + "--test-results.json").read_text()
    )
    original_off = next(c for c in original_test["cases"] if c["label"] == "off")
    hashes = []
    for start in range(0, cfg["samples"], cfg["manuscript_batch_size"]):
        stop = min(start + cfg["manuscript_batch_size"], 1000)
        sha = dataset.sha(
            dataset.sensory_batch(arrays, np.arange(start, stop), cfg).numpy()
        )
        if sha != original_off["sensory_sha256"][start // cfg["manuscript_batch_size"]]:
            raise ValueError("Sensory replay differs from retained exp171 input")
        hashes.append(sha)
    performance = sim.Performance(
        compile_mode=None, precision="highest", projection_mode="fast"
    )
    extensions.register_encoder("pinglab.exp174/v1", dataset.encode)
    with stage_run(
        ROOT,
        recipe.SLUG,
        "compute",
        inputs={"exp171": source, "bank": bank},
        configuration=cfg,
        operation="prototype-test-prefix",
        run_id=run_id,
    ) as run:
        dataset_path = run.export / "dataset.npz"
        np.savez_compressed(dataset_path, **arrays)
        write_json_atomic(
            run.export / "sensory-hashes.json", {"original_full_batch_sha256": hashes}
        )
        timings = {}
        for kind in recipe.KINDS:
            tick = time.perf_counter()
            out = run.export / kind
            out.mkdir()
            if kind == "artificial":
                bundle = lang.load_bundle(
                    source.restore_bundle(
                        cfg["test_origin"] + "/off/network.bundle",
                        run.scratch / "original.bundle",
                    )
                )
                original_start = source.restore_bundle(
                    cfg["test_origin"] + "/off/student/start",
                    run.scratch / "original-start",
                )
                original_metadata = json.loads(
                    (original_start / "snapshot.json").read_text()
                )
                if (
                    original_metadata["execution"]["checkpoint"]["completed_updates"]
                    != 250
                ):
                    raise ValueError(
                        "Student snapshot is not the final 10-epoch checkpoint"
                    )
                params = sim.NetworkSnapshot.read(
                    original_start, parameters=True
                ).load_parameters()
                if bundle.graph["refractory_timing"] != cfg["refractory_timing"][kind]:
                    raise ValueError("Student refractory timing differs from source")
            else:
                bundle = recipe.author_network(cfg, kind)
                params = weights[kind]
            bundle.write(out / "network.bundle", visualise=False)
            bundle.visualise(run.export / ("circuit-" + kind + ".svg"), view="circuit")
            checkpoint_path = run.scratch / (kind + ".pth")
            checkpoint(bundle, params, checkpoint_path, performance)
            collector = PrefixCollector(cfg["steps"], cfg["samples"], cfg["n_output"])
            encoder = sim.DatasetEncoder(
                "custom",
                definition="pinglab.exp174/v1",
                rng_policy="sample_v1",
                duration_ms=cfg["duration_ms"],
                config={
                    "kind": kind,
                    "recipe": cfg,
                    "signature": sim.runtime_state_signature(
                        sim.plan_graph(bundle.graph)
                    ),
                },
            )
            with torch.inference_mode():
                result = sim.simulate(
                    bundle=bundle,
                    checkpoint=checkpoint_path,
                    input_bindings=[
                        sim.DatasetSnapshotBinding(
                            path=dataset_path,
                            input_id="drive",
                            dataset_id="exp171-off-test",
                            split="test",
                            encoder=encoder,
                            feature_key="features",
                            label_key="labels",
                            sample_ids_key="ids",
                            sample_indices=tuple(range(cfg["samples"])),
                        )
                    ],
                    batch_size=cfg["batch_size"],
                    seed=cfg["seed"],
                    device=cfg["device"],
                    diagnostics=False,
                    performance=performance,
                    snapshot=sim.SnapshotSpec(
                        out / "start", stage="start", parameters=True
                    ),
                    recording=sim.RecordingSpec(
                        signals=tuple(sim.SignalRecording(s) for s in collector.seen),
                        sink=collector,
                        sink_id="exp174.prefix-population-reduction/v1",
                        block_steps=cfg["recording_block_steps"],
                    ),
                    retain_parameters=False,
                    retain_final_state=False,
                    retain_runtime_state=False,
                )
            scores = collector.validate(result, cfg)
            extra_events = np.zeros((cfg["steps"], cfg["samples"]), np.int32)
            if kind == "artificial":
                for start in range(0, cfg["samples"], cfg["batch_size"]):
                    selected = range(
                        start, min(start + cfg["batch_size"], cfg["samples"])
                    )
                    _, _, _, counts = dataset.private_batch(arrays, selected, cfg)
                    extra_events[:, start : start + len(selected)] = counts
            np.savez_compressed(
                out / "temporal.npz",
                predictions=collector.predictions,
                e_counts=collector.counts["E"],
                i_counts=collector.counts["I"],
                artificial_events=extra_events,
                endpoint_scores=scores,
                ids=arrays["ids"][: cfg["samples"]],
                labels=arrays["labels"][: cfg["samples"]],
            )
            timings[kind] = time.perf_counter() - tick
            write_json_atomic(out / "performance.json", result.metrics["performance"])
            print(
                f"{kind}: {(collector.predictions[-1] == arrays['labels'][: cfg['samples']]).mean():.1%}; {timings[kind]:.2f}s",
                flush=True,
            )
        run.details["scientific_artifacts"] = {
            "snapshot_stages": {k: ["start"] for k in recipe.KINDS}
        }
        run.details["execution"].update(
            elapsed_seconds=time.perf_counter() - started,
            condition_seconds=timings,
            no_training=True,
        )
        write_json_atomic(
            run.export / "timings.json",
            {
                "conditions_seconds": timings,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=int, default=recipe.configuration()["samples"]
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    execute(
        recipe.configuration(samples=args.samples, batch_size=args.batch_size),
        run_id=args.run_id,
    )
