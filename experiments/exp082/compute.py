"""Compute exp082 streaming inference from a pinned v4 bank, with six shards."""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp082 import evidence, inputs, recipe
from experiments.exp082.inference import Inference
from experiments.helpers import concurrent_compute
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)
from pingstore.stages import _capture_code, reserve_stage


def _run_jobs(bank, directory, jobs, contract):
    cfg = recipe.environment_configuration()
    worker = Inference(bank, directory, cfg)
    for job in jobs:
        if (directory / "export" / job["path"]).exists() or (
            directory / ".scratch/simulations" / job["path"]
        ).exists():
            raise PingstoreError(
                "incomplete job exists; explicit recovery or a fresh run required"
            )
        worker.condition(job)
        write_json_atomic(
            directory / ".scratch/simulations" / job["path"] / "dataset.json",
            worker.dataset,
        )
        retained = evidence.counts(
            directory / "export" / job["path"] / "counts.npz", cfg
        )
        if not (retained["labels"] == worker.image_stream_labels).all():
            raise PingstoreError("condition labels differ from shared image bank")


def _job_inventory(directory, jobs):
    files = {}
    for job in jobs:
        for prefix in ("export", ".scratch/simulations"):
            folder = directory / prefix / job["path"]
            if not folder.is_dir() or folder.is_symlink():
                raise PingstoreError("missing or linked job evidence")
            for path in folder.rglob("*"):
                if path.is_symlink() or not (path.is_file() or path.is_dir()):
                    raise PingstoreError("unsupported job evidence entry")
                if path.is_file():
                    files[str(path.relative_to(directory))] = file_sha256(path)
    return files


def _shard_paths(repo, run_id, index, count):
    return concurrent_compute.working_directory(
        repo,
        run_id,
        index,
        count,
        experiment=recipe.SLUG,
        expected_count=recipe.SHARDS,
    )


_compute_lock = concurrent_compute.compute_lock


def shard(identity, *, run_id, index, count=recipe.SHARDS):
    """Compute-only recovery: each shard owns an isolated lock and completion record."""
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    contract = evidence.training_contract(bank.export)
    cfg = recipe.environment_configuration()
    directory = _shard_paths(REPO, run_id, index, count)
    job_list = recipe.jobs(cfg)[index::count]

    def check_inputs():
        for ancestor in inputs.lineage(REPO, identity, bank.reference).values():
            ancestor.check_unchanged()

    return concurrent_compute.execute_shard(
        repo=REPO,
        experiment=recipe.SLUG,
        run_id=run_id,
        index=index,
        count=count,
        expected_count=recipe.SHARDS,
        inputs={"bank": bank.reference},
        configuration=cfg,
        items=job_list,
        run_items=lambda: _run_jobs(bank, directory, job_list, contract),
        inventory=lambda: _job_inventory(directory, job_list),
        check_inputs=check_inputs,
        capture_code=_capture_code,
    )


def compute(identity, *, run_id=None, collect=False):
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    contract = evidence.training_contract(bank.export)
    cfg = recipe.environment_configuration()
    if collect and not run_id:
        raise PingstoreError("collection requires an explicit compute reservation")
    run_id = run_id or reserve_stage(REPO / ".pingstore", recipe.SLUG, "compute")
    with concurrent_compute.collect_shards(
        repo=REPO,
        experiment=recipe.SLUG,
        run_id=run_id,
        count=recipe.SHARDS,
        inputs={"bank": bank.reference},
        configuration=cfg,
        items_for=lambda index: recipe.jobs(cfg)[index :: recipe.SHARDS],
        inventory_for=lambda index: _job_inventory(
            _shard_paths(REPO, run_id, index, recipe.SHARDS),
            recipe.jobs(cfg)[index :: recipe.SHARDS],
        ),
        collect=collect,
    ) as (_directory, shard_records):
        with inputs.execution(
            REPO, "compute", sources={"bank": bank}, run_id=run_id, configuration=cfg
        ) as run:
            if collect:
                concurrent_compute.retain_worker_provenance(run, shard_records)
            environment = {
                "PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0",
                **{
                    k: os.environ[k]
                    for k in (
                        "PINGLAB_EXP082_STREAMS_PER_CELL",
                        "PINGLAB_EXP082_DIGITS_PER_STREAM",
                        "PINGLAB_EXP082_STREAM_BATCH_SIZE",
                    )
                    if k in os.environ
                },
            }
            run.record["execution"]["environment"] = environment
            if not collect:
                _run_jobs(bank, run.directory, recipe.jobs(cfg), contract)
            worker = Inference(bank, run.directory, cfg)
            for job in recipe.jobs(cfg):
                if (
                    load_json(
                        run.scratch / "simulations" / job["path"] / "dataset.json"
                    )
                    != worker.dataset
                ):
                    raise PingstoreError(
                        "inference workers used different dataset bytes"
                    )
            for name in ("matched", "variable"):
                worker.stream(name)
            write_json_atomic(run.scratch / "dataset.json", worker.dataset)
            write_json_atomic(
                run.export / "evidence.json",
                {
                    "schema": "exp082.compute/v2",
                    "recipe": cfg,
                    "training_contract": contract,
                    "jobs": recipe.jobs(cfg),
                    "image_stream_bank": worker.image_stream_bank,
                },
            )
            evidence.validate_compute(run.export, cfg)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="explicit completed exp022 compute ID"
    )
    parser.add_argument("--run-id", help="unused v4 reservation")
    parser.add_argument(
        "--shard-index", type=int, help="compute worker index (six shards)"
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="complete this compute run from its six shards",
    )
    args = parser.parse_args()
    try:
        if args.shard_index is not None:
            if not args.run_id or args.collect:
                raise PingstoreError(
                    "shard workers require --run-id and cannot --collect"
                )
            shard(args.source, run_id=args.run_id, index=args.shard_index)
        else:
            compute(args.source, run_id=args.run_id, collect=args.collect)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp082 compute: {exc}\n")


if __name__ == "__main__":
    main()
