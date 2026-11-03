"""Import authenticated native evidence, or explicitly simulate the two-state recipe."""

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp168 import recipe, dataset
from pingstore.library import source_run, stage_run, load_json, file_sha256


def recorded_configuration(source, label):
    spec = recipe.IMPORTS[label]
    numbers = load_json(source.file(spec["numbers"]))
    original = (
        next(r for r in numbers["trials"] if r["trial"] == spec["trial"])[
            "configuration"
        ]
        if label == "AI"
        else numbers["configuration"]
    )
    expected = recipe.configuration(label)
    for key in (
        "seed",
        "n_e",
        "n_i",
        "n_external",
        "dt_ms",
        "duration_ms",
        "measurement_ms",
        "external_rate_hz",
        "external_seed_offset",
        "connection_probability",
        "input_connection_probability",
        "initializer_parent_means_us",
        "weight_relative_sd",
        "capacitance_nf",
        "leak_us",
        "resting_mv",
        "threshold_mv",
        "reset_mv",
        "excitatory_reversal_mv",
        "inhibitory_reversal_mv",
        "refractory_e_ms",
        "refractory_i_ms",
        "tau_synapse_ms",
        "excitatory_delay_ms",
        "inhibitory_delay_ms",
    ):
        if original[key] != expected[key]:
            raise ValueError(
                f"{label}: source setting differs from preserved recipe: {key}"
            )
    gain = original["control"]["final_gain"]
    if original["control"]["baseline_gain"] != gain or gain != expected["input_i_gain"]:
        raise ValueError("source input-to-I gain is not the expected constant")
    cfg = deepcopy(original)
    cfg["input_i_gain"] = gain  # Explicit alias derived from the executed modulation.
    return cfg


def import_native(store, *, run_id=None):
    started = perf_counter()
    sources = {
        label: source_run(
            Path(store),
            spec["run_id"],
            stage="compute",
            reference={
                "run_id": spec["run_id"],
                "payload_digest": spec["payload_digest"],
            },
        )
        for label, spec in recipe.IMPORTS.items()
    }
    configs = {
        label: recorded_configuration(source, label)
        for label, source in sources.items()
    }
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        operation="import",
        configuration={"conditions": configs},
        run_id=run_id,
    ) as run:
        # Isolated source IDs are scoped to their original store. Preserve the
        # authenticated records in an archive rather than colliding with global IDs.
        evidence = {
            "kind": "authenticated-native-import",
            "source_store": str(Path(store).resolve()),
            "sources": {},
        }
        with ZipFile(
            run.export / "source-provenance.zip", "w", compression=ZIP_DEFLATED
        ) as archive:
            visited = set()

            def archive_ancestry(source):
                if source.record["run_id"] in visited:
                    return
                visited.add(source.record["run_id"])
                archive.write(
                    source.directory / "run.json",
                    "records/" + source.record["run_id"] + "/run.json",
                )
                for pin in source.record["inputs"].values():
                    archive_ancestry(
                        source_run(Path(store), pin["run_id"], reference=pin)
                    )

            for label, source in sources.items():
                spec = recipe.IMPORTS[label]
                archive_ancestry(source)
                target = run.export / label
                target.mkdir()
                source.restore_bundle(
                    spec["prefix"] + "/network.bundle", target / "network.bundle"
                )
                (target / "start").mkdir()
                copied = {}
                for origin, name in (
                    (source.file(spec["prefix"], "events.npz"), "events.npz"),
                    (
                        source.file(spec["prefix"], "start", "snapshot.json"),
                        "start/snapshot.json",
                    ),
                    (
                        source.file(spec["prefix"], "start", "parameters.npz"),
                        "start/parameters.npz",
                    ),
                ):
                    shutil.copyfile(origin, target / name)
                    digest = file_sha256(origin)
                    assert digest == file_sha256(target / name)
                    copied[name] = digest
                graph = source.file(spec["prefix"], "network.bundle", "graph.json")
                diagram_graph = source.file(
                    spec["diagram_prefix"], "network.bundle", "graph.json"
                )
                if file_sha256(graph) != file_sha256(diagram_graph):
                    raise ValueError("diagram is not from the selected compiled graph")
                if label == "AI":
                    shutil.copyfile(
                        source.file(spec["diagram_prefix"], "circuit.svg"),
                        run.export / "circuit.svg",
                    )
                for file in source.export.iterdir():
                    if file.is_file() and file.suffix == ".py":
                        archive.write(file, label + "/" + file.name)
                evidence["sources"][label] = {
                    **source.reference,
                    "prefix": spec["prefix"],
                    "original_execution": source.record["execution"],
                    "copied_sha256": copied,
                }
        run.record["execution"].update(
            evidence=evidence, elapsed_s=perf_counter() - started
        )
        run.record["scientific_artifacts"] = {
            "snapshot_stages": {label: ["start"] for label in recipe.CONDITIONS}
        }
        for source in sources.values():
            source.check_unchanged()
    return run.run_id


def execute(cfg, directory, diagram):
    from snnlab.sim import (
        DenseArrayBinding,
        Performance,
        RecordingSpec,
        SignalRecording,
        SnapshotSpec,
        simulate,
    )

    directory.mkdir(parents=True)
    bundle = recipe.author_network(cfg)
    bundle.write(directory / "network.bundle", visualise=False)
    if not diagram.exists():
        bundle.visualise(diagram, view="circuit")
    drive = dataset.encode_drive(cfg).to(cfg["device"])
    events = {"E": [], "I": []}

    def collect(block):
        if block.kind == "spike_events":
            events[block.signal.split(".")[0]].append(block.values.numpy())

    simulate(
        bundle=bundle,
        input_bindings=(DenseArrayBinding("X", drive),),
        seed=cfg["seed"],
        device=cfg["device"],
        performance=Performance(**cfg["performance"]),
        diagnostics=False,
        recording=RecordingSpec(
            signals=tuple(
                SignalRecording(p + ".spikes", kind="spike_events") for p in ("E", "I")
            ),
            sink=collect,
            sink_id="exp168.sparse-spikes/v1",
            block_steps=cfg["recording_block_steps"],
        ),
        snapshot=SnapshotSpec(directory / "start", stage="start", parameters=True),
        retain_outputs=False,
        retain_parameters=True,
        retain_final_state=False,
        retain_runtime_state=False,
    )
    arrays = {}
    for pop, blocks in events.items():
        rows = np.concatenate(blocks) if blocks else np.empty((0, 3))
        arrays[pop + "_time_ms"] = rows[:, 0] * cfg["dt_ms"]
        arrays[pop + "_cell"] = rows[:, 2]
    np.savez_compressed(directory / "events.npz", **arrays)


def simulate_conditions(*, seed=43, device="cpu", run_id=None):
    import torch
    from importlib.metadata import version

    started = perf_counter()
    configs = {
        label: recipe.configuration(label, seed=seed, device=device)
        for label in recipe.CONDITIONS
    }
    torch.set_num_threads(configs["AI"]["torch_threads"])
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        configuration={"conditions": configs},
        run_id=run_id,
    ) as run:
        for label, cfg in configs.items():
            execute(cfg, run.export / label, run.export / "circuit.svg")
        run.record["scientific_artifacts"] = {
            "snapshot_stages": {label: ["start"] for label in recipe.CONDITIONS}
        }
        run.record["execution"].update(
            elapsed_s=perf_counter() - started, snnlab_version=version("snnlab")
        )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--import-store", type=Path)
    mode.add_argument("--simulate", action="store_true")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.import_store:
        if args.seed != 43 or args.device != "cpu":
            parser.error("import preserves executed seed and device")
        import_native(args.import_store, run_id=args.run_id)
    else:
        simulate_conditions(seed=args.seed, device=args.device, run_id=args.run_id)
