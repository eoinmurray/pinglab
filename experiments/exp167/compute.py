"""Import historical outputs, or explicitly execute a fresh two-condition run."""

import argparse
import shutil
import sys
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp167 import dataset, recipe
from experiments.exp167.analyse import validate_metrics
from pingstore.library import file_sha256, load_json, stage_run
from snnlab.sim import (
    DenseArrayBinding,
    Performance,
    RecordingSpec,
    SignalRecording,
    SnapshotSpec,
    simulate,
)


def import_outputs(directory, *, run_id=None):
    """Preserve the exact legacy files; this does not validate missing simulation evidence."""
    started = perf_counter()
    directory = Path(directory).expanduser().resolve()
    files = ["run.py", "metrics.json", "AI/raster.png", "PING/raster.png"]
    values = validate_metrics(load_json(directory / "metrics.json"))
    for name in files:
        if not (directory / name).is_file():
            raise ValueError("missing historical file: " + name)
    # Capture original dependencies as provenance, never as runtime imports.
    dependencies = [
        REPO / "experiments/exp148/recipe.py",
        REPO / "experiments/exp148/dataset.py",
    ]
    evidence = dict(
        kind="historical-scratch-import",
        original_source=str(directory),
        raw_spikes_available=False,
        native_execution_evidence_available=False,
        configuration_origin="reconstructed from retained scratch source and current exp148 source; execution not independently verified",
        missing=[
            "events.npz",
            "executed native bundles",
            "simulation-start snapshots",
            "original runtime version and timings",
        ],
        file_sha256={name: file_sha256(directory / name) for name in files},
    )
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        operation="import",
        configuration=recipe.configuration(),
        run_id=run_id,
    ) as run:
        for name in files[1:]:
            target = run.export / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / name, target)
        with ZipFile(
            run.export / "historical-source.zip", "w", compression=ZIP_DEFLATED
        ) as archive:
            archive.write(directory / "run.py", "scratch/run.py")
            for path in dependencies:
                archive.write(path, str(path.relative_to(REPO)))
        for name in files[1:]:
            if file_sha256(run.export / name) != evidence["file_sha256"][name]:
                raise ValueError("import byte verification failed")
        assert values == load_json(run.export / "metrics.json")
        run.record["execution"].update(
            evidence=evidence, elapsed_s=perf_counter() - started
        )
    return run.run_id


def execute(cfg, directory):
    directory.mkdir(parents=True)
    bundle = recipe.author_network(cfg)
    bundle.write(directory / "network.bundle", visualise=False)
    bundle.visualise(directory / "circuit.svg", view="circuit")
    external = dataset.encode_drive(cfg).to(cfg["device"])
    events = {"E": [], "I": []}

    def collect(block):
        if block.kind == "spike_events":
            events[block.signal.split(".")[0]].append(block.values.numpy())

    simulate(
        bundle=bundle,
        input_bindings=(DenseArrayBinding("X", external),),
        seed=cfg["seed"],
        device=cfg["device"],
        performance=Performance(**cfg["performance"]),
        diagnostics=False,
        recording=RecordingSpec(
            signals=tuple(
                SignalRecording(f"{p}.spikes", kind="spike_events") for p in ("E", "I")
            ),
            sink=collect,
            sink_id="exp167.sparse-spikes/v1",
            block_steps=cfg["recording_block_steps"],
        ),
        snapshot=SnapshotSpec(directory / "start", stage="start", parameters=True),
        retain_outputs=False,
        retain_parameters=True,
        retain_final_state=False,
        retain_runtime_state=False,
    )
    arrays = {}
    for population, blocks in events.items():
        rows = np.concatenate(blocks)
        arrays[population + "_time_ms"] = rows[:, 0] * cfg["dt_ms"]
        arrays[population + "_cell"] = rows[:, 2]
    np.savez_compressed(directory / "events.npz", **arrays)


def compute(*, seed=42, device="cpu", run_id=None):
    from importlib.metadata import version

    import torch

    started = perf_counter()
    cfg = recipe.configuration(seed=seed, device=device)
    torch.set_num_threads(cfg["torch_threads"])
    with stage_run(
        REPO, recipe.SLUG, "compute", configuration=cfg, run_id=run_id
    ) as run:
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={label: ["start"] for label in cfg["conditions"]}
        )
        run.record["execution"]["snnlab_version"] = version("snnlab")
        for label in cfg["conditions"]:
            execute(recipe.condition_configuration(cfg, label), run.export / label)
        run.record["execution"]["elapsed_s"] = perf_counter() - started
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--import-from", type=Path)
    mode.add_argument("--simulate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.import_from:
        if args.seed != 42 or args.device != "cpu":
            parser.error("import preserves historical seed/device")
        import_outputs(args.import_from, run_id=args.run_id)
    else:
        compute(seed=args.seed, device=args.device, run_id=args.run_id)
