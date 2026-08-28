"""Approved, scoped Gold-2 import. No neural simulation, publication or archive writes."""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp082 import (
    evidence,
    historical,
    inference,
    inputs,
    measurements,
    recipe,
)
from experiments.helpers.checkpoints import public_provenance, resolve_checkpoint
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)

REMOTE = "r2:pinglab/campaigns/gold-2"
HEADERS = {
    "run.json": (
        1103,
        "d4d067148589ae8469a37ee765c77282c9ab081eff22dfda0a24d596cbba913c",
    ),
    "inventory.json": (
        2101788,
        "c7b9455968e34ac3be2df46a57ab4fc0ffcd94dc88799bf11b36c9b673a88f68",
    ),
    "lineage.json": (
        1876,
        "d3e128a730ec4f85011e73cafdb0119683b4af25be7daced6d5eace550658e97",
    ),
}
DATA = "derived/artifacts/data/exp082/"
BASE = "state/checkpoints/base-production-exp022/"
STREAM = "state/experiments/exp082/stream/ping__variable_rate__seed42__best_validation__6af7b0c4e3e0/"
DATASET_HASHES = {
    "t10k-images-idx3-ubyte": "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
    "t10k-labels-idx1-ubyte": "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2",
}


def checked(path, size, sha):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise PingstoreError(f"source uses a symlink: {path}")
    if not path.is_file() or path.stat().st_size != size or file_sha256(path) != sha:
        raise PingstoreError(f"source checksum differs: {path}")


def live_metadata(archive):
    result = {}
    for name, (size, sha) in HEADERS.items():
        checked(archive / name, size, sha)
        command = [
            "rclone",
            "cat",
            f"{REMOTE}/{name}",
            "--retries",
            "1",
            "--low-level-retries",
            "1",
        ]
        data = subprocess.run(
            command, check=True, capture_output=True, timeout=60
        ).stdout
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            raise PingstoreError(f"live R2 metadata differs: {name}")
        result[name] = {"command": command, "size_bytes": size, "sha256": sha}
    return result


def selection(archive):
    for name, (size, sha) in HEADERS.items():
        checked(archive / name, size, sha)
    all_files = load_json(archive / "inventory.json")["files"]
    excluded = {
        DATA + name
        for name in (
            "duration_rate_summary.png",
            "matched_stream.png",
            "variable_stream.png",
            "psychometric_200ms.svg",
        )
    }
    excluded.update(STREAM + name + "/rasters.npz" for name in ("matched", "variable"))
    extras = {
        "lineage.json",
        BASE + "campaign.json",
        "provenance/source-records/base/run.json",
        "provenance/source-records/base/collection-plan.json",
    }
    extras.update(
        BASE + "cells/" + recipe.training_cell_name(s) + "/" + f
        for s in recipe.SEEDS
        for f in ("config.json", "metrics.json", "attempt.json")
    )
    files = [
        dict(x)
        for x in all_files
        if (
            ("exp082" in x["path"] and x["path"] not in excluded) or x["path"] in extras
        )
    ]
    files.extend(
        {"path": n, "size_bytes": HEADERS[n][0], "sha256": HEADERS[n][1]}
        for n in ("run.json", "inventory.json")
    )
    if len(files) != 199 or sum(x["size_bytes"] for x in files) != 6079619:
        raise PingstoreError("selection differs from approved 199-file plan")
    omitted = [x for x in all_files if x["path"] in excluded]
    if len(omitted) != 6 or sum(x["size_bytes"] for x in omitted) != 672858:
        raise PingstoreError("exclusions differ from approved plan")
    for item in [*files, *omitted]:
        checked(archive / item["path"], item["size_bytes"], item["sha256"])
    return files, omitted


def scientific_view(obj):
    # These are the only metadata differences between the base and repaired bank.
    ignored = {
        "campaign_id",
        "campaign_manifest_sha256",
        "campaign_repository_commit",
        "imported_cell_provenance",
    }

    def visit(value, parent=None):
        if isinstance(value, dict):
            return {
                k: visit(v, k)
                for k, v in value.items()
                if k not in ignored and not (parent == "arguments" and k == "--out-dir")
            }
        if isinstance(value, list):
            return [visit(v) for v in value]
        return value

    return visit(obj)


def verify_bank(archive, bank, old):
    contract = evidence.training_contract(bank.export)
    if contract["checkpoints"] != old["checkpoint_provenance"]:
        raise PingstoreError("selected checkpoint lineage differs")
    for seed in recipe.SEEDS:
        name = recipe.training_cell_name(seed)
        source = archive / BASE / "cells" / name
        for filename in ("config.json", "metrics.json"):
            if scientific_view(load_json(source / filename)) != scientific_view(
                load_json(bank.export / name / filename)
            ):
                raise PingstoreError(
                    f"bank scientific record differs: {name}/{filename}"
                )
        for role in ("best_validation", "final_epoch"):
            if public_provenance(resolve_checkpoint(source, role)) != public_provenance(
                resolve_checkpoint(bank.export / name, role)
            ):
                raise PingstoreError("bank checkpoint roles differ")
    return contract


def reconstruct(archive, mnist, old):
    for name, sha in DATASET_HASHES.items():
        if file_sha256(mnist / name) != sha:
            raise PingstoreError("official MNIST reconstruction source differs")
    images = (
        np.frombuffer(
            (mnist / "t10k-images-idx3-ubyte").read_bytes(), dtype=np.uint8, offset=16
        )
        .reshape(10000, 784)
        .astype(np.float32)
        / 255.0
    )
    labels = np.frombuffer(
        (mnist / "t10k-labels-idx1-ubyte").read_bytes(), dtype=np.uint8, offset=8
    ).astype(np.int64)
    arrays = evidence.arrays(archive / DATA / "measurements.npz")
    if set(arrays) != {
        f"{n}_{k}"
        for n in ("matched", "variable")
        for k in ("spikes_e", "spikes_i", "spikes_out", "probabilities")
    }:
        raise PingstoreError("historical measurement arrays differ")
    result, proof = {}, {}
    for name, seed in (("matched", 82), ("variable", 83)):
        meta = old[name + "_stream"]
        pixels, chosen = inference.pick_digits(images, labels, 5, seed)
        if chosen.tolist() != meta["labels"]:
            raise PingstoreError("reconstructed labels differ")
        encoded = inference.encode_stream(
            pixels, meta["conditions"], torch.Generator().manual_seed(seed + 1)
        ).numpy()
        original = evidence.arrays(archive / STREAM / name / "input.npz")
        reset = np.zeros(len(encoded), dtype=bool)
        reset[meta["boundaries"][:-1]] = True
        if not np.array_equal(encoded, original["input_spikes"]) or not np.array_equal(
            reset, original["readout_reset"]
        ):
            raise PingstoreError(
                "reconstructed input differs from historical simulation"
            )
        raw = {
            k: arrays[name + "_" + k] for k in ("spikes_e", "spikes_i", "spikes_out")
        }
        raw["pixels"] = pixels
        rasters = evidence.arrays(archive / STREAM / name / "rasters.npz")
        for pop in ("e", "i", "out"):
            dense = np.zeros_like(raw["spikes_" + pop])
            if np.any(rasters[pop + "_trial"] != 0):
                raise PingstoreError("unexpected raster trial")
            dense[rasters[pop + "_t"], rasters[pop + "_cell"]] = 1
            if not np.array_equal(dense, raw["spikes_" + pop]):
                raise PingstoreError("excluded raster is not redundant")
        replay = measurements.stream_result(raw, meta)
        for key in ("predictions", "correct", "output_activity"):
            if replay[key] != meta[key]:
                raise PingstoreError("historical stream measurement differs")
        if not np.array_equal(replay["probabilities"], arrays[name + "_probabilities"]):
            raise PingstoreError("historical count shares differ")
        result[name] = raw
        proof[name] = {
            "selection_seed": seed,
            "encoding_seed": seed + 1,
            "pixels": inference.array_record(pixels),
            "encoded_input": inference.array_record(encoded),
            "input_replay_exact": True,
            "raster_equivalence_exact": True,
        }
    return result, {
        "dataset_sha256": DATASET_HASHES,
        "streams": proof,
        "neural_simulation_executed": False,
    }


def import_run(archive, bank_id, mnist):
    archive, mnist = archive.absolute(), mnist.absolute()
    before = live_metadata(archive)
    files, excluded = selection(archive)
    bank = inputs.source(REPO, bank_id, "compute", experiment="exp022")
    old = load_json(archive / DATA / "numbers.json")
    cfg = recipe.configuration()
    if old["config"] != {
        k: v
        for k, v in cfg.items()
        if k not in ("schema", "profile", "checkpoint_policy")
    }:
        raise PingstoreError("historical evaluation recipe differs")
    contract = verify_bank(archive, bank, old)
    recordings, reconstruction = reconstruct(archive, mnist, old)
    conditions = {}
    for item in files:
        if "/exp082/conditions/" in item["path"]:
            row = load_json(archive / item["path"])
            identity = recipe.condition_job_id(
                row["seed"], row["duration_ms"], row["rate_hz"]
            )
            if identity in conditions:
                raise PingstoreError("duplicate condition source")
            conditions[identity] = item["path"]
    if set(conditions) != {j["id"] for j in recipe.jobs(cfg)}:
        raise PingstoreError("historical grid incomplete")
    targets = {v: "export/jobs/" + k + "/condition.json" for k, v in conditions.items()}
    targets[DATA + "measurements.npz"] = "export/historical/measurements.npz"
    for name in ("matched", "variable"):
        targets[STREAM + name + "/input.npz"] = f"export/historical/{name}_input.npz"
    with inputs.execution(
        REPO,
        "compute",
        sources={"bank": bank},
        configuration=cfg,
        operation="historical-import",
    ) as run:
        run.record["historical_import"] = {
            "schema": historical.SCHEMA,
            "source": REMOTE,
            "producer_commit": historical.PRODUCER,
            "producer_campaign": "ggs-exp082-repair-20260820-73f0883e",
            "producer_origin": "hpc",
            "producer_host": "gpu-q-26",
            "slurm_array": "34021105",
            "slurm_aggregate": "34021106",
            "base_campaign": "ggs-production-20260818-4ad223d3",
            "repair": old["repair_run_provenance"],
            "source_manifest_discrepancy": "Original presentation says host=local; retained Slurm logs establish HPC production.",
            "limitations": "Grid retains original aggregates, not per-decision labels or counts. Ten image pixels reconstructed and input-replay verified; no neural simulation.",
        }
        mapped = []
        for item in files:
            target = targets.get(item["path"], "provenance/archive/" + item["path"])
            dest = run.directory / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive / item["path"], dest)
            checked(dest, item["size_bytes"], item["sha256"])
            mapped.append(
                {
                    "source": item["path"],
                    "target": target,
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                }
            )
        for name, raw in recordings.items():
            folder = run.export / "streams" / name
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "recordings.npz", **raw)
            write_json_atomic(
                folder / "stream.json",
                {
                    k: old[name + "_stream"][k]
                    for k in ("labels", "boundaries", "conditions")
                },
            )
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp082.compute/v1",
                "recipe": cfg,
                "training_contract": contract,
                "jobs": recipe.jobs(cfg),
                "condition_evidence": "historical-aggregate/v1",
            },
        )
        write_json_atomic(
            run.provenance / "import.json",
            {
                "schema": historical.SCHEMA,
                "source_files": len(files),
                "source_bytes": sum(f["size_bytes"] for f in files),
                "files": mapped,
                "excluded": excluded,
                "upstream": bank.reference,
                "reconstruction": reconstruction,
                "live_metadata_before": before,
                "live_metadata_after": live_metadata(archive),
            },
        )
        selection(archive)
        historical.validate_import(run, cfg)
        evidence.validate_compute(run.export, cfg, historical=True)
        shutil.copyfile(
            Path(__file__).with_name("IMPORT_PLAN.md"),
            run.provenance / "approved-import-plan.md",
        )
        (run.directory / "README.md").write_text(
            "# Historical exp082 import\n\nLocal import of corrected HPC inference, not a new simulation or training run.\nOriginal producer: ggs-exp082-repair-20260820-73f0883e; Slurm array 34021105, aggregate 34021106.\nThe repaired operational exp022 bank is referenced because both checkpoint roles match the original base bank exactly. Original base records and lineage are retained.\n\nAll 132 historical aggregate conditions and both complete illustrative streams are retained. Per-decision grid counts were not archived. Ten image thumbnails were reconstructed from hash-verified official test data and exactly reproduce the original encoded inputs.\nSee provenance/import.json for the approved selection, source hashes, exact mappings, live R2 checks and reconstruction evidence; run.json separates the local importer from its HPC producer.\n"
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--mnist-raw", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(import_run(args.archive, args.bank, args.mnist_raw))
    except (
        PingstoreError,
        OSError,
        ValueError,
        KeyError,
        subprocess.SubprocessError,
    ) as exc:
        parser.exit(1, f"exp082 historical import: {exc}\n")


if __name__ == "__main__":
    main()
