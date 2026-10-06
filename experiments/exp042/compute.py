"""Compute the reduced interventions from an explicit v4 bank; never analyse or present."""

import argparse
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp042 import inputs, recipe, simulation
from experiments.helpers.hpc import concurrent_compute
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)
from pingstore.stages import _capture_code, reserve_stage


def _job_path(export, job):
    return export / (job["id"] + "--metrics.json")


def _run_jobs(bank, cfg, data, root, export, jobs, requests):
    for job, metrics in simulation.evaluate_jobs(bank, cfg, data, root, jobs, requests):
        record = {"job": job, "metrics": metrics}
        canonical = recipe.replay_job(job)
        if canonical != job:
            record["replay_of"] = canonical["id"]
        write_json_atomic(_job_path(export, job), record)


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


def _shard(identity, *, run_id, index, count=recipe.SHARDS):
    """Compute-only recovery: each shard owns an isolated lock and completion record."""
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    inputs.bank_evidence(bank)
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    directory = _shard_paths(REPO, run_id, index, count)
    job_list = recipe.jobs(cfg)[index::count]

    data = simulation.load_evaluation(cfg)
    requests = []
    request_path = directory / ".scratch/shards" / str(index) / "requests.json"

    def run_items():
        _run_jobs(
            bank,
            cfg,
            data,
            directory / ".baseline-scratch",
            directory / "export",
            job_list,
            requests,
        )
        write_json_atomic(request_path, requests)

    def inventory():
        return {
            **{
                job["id"]: file_sha256(_job_path(directory / "export", job))
                for job in job_list
            },
            "requests": file_sha256(request_path),
        }

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
        configuration={"recipe": cfg, **data["protocol"]},
        items=job_list,
        run_items=run_items,
        inventory=inventory,
        check_inputs=check_inputs,
        capture_code=_capture_code,
    )


@contextmanager
def _threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def shard(identity, *, run_id, index, count=recipe.SHARDS):
    with _threads():
        return _shard(identity, run_id=run_id, index=index, count=count)


def compute(identity, *, run_id=None, collect=False):
    with _threads():
        return _compute(identity, run_id=run_id, collect=collect)


def _compute(identity, *, run_id=None, collect=False):
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    evidence = inputs.bank_evidence(bank)
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    if collect and not run_id:
        raise PingstoreError("collection requires an explicit compute reservation")
    run_id = run_id or reserve_stage(REPO / ".pingstore", recipe.SLUG, "compute")
    directory = _shard_paths(REPO, run_id, 0, recipe.SHARDS)
    scratch = directory / ".scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-", dir=scratch) as tmp:
        data = simulation.load_evaluation(cfg)
        return _complete(bank, evidence, cfg, data, Path(tmp), run_id, collect)


def _collection_protocol(directory, cfg, data):
    """Authenticate frozen GPU requests without pretending the CPU collector is a worker."""
    frozen = load_json(directory / ".scratch/shards/0/completed.json")["configuration"]
    if set(frozen) != {"recipe", *data["protocol"]}:
        raise PingstoreError("collected request envelope changed")
    if (
        frozen.get("recipe") != cfg
        or frozen.get("dataset_sha256") != data["protocol"]["dataset_sha256"]
    ):
        raise PingstoreError("collected recipe or dataset changed")
    expected = dict(data["protocol"]["runtime"])
    actual = dict(frozen.get("runtime", {}))
    for key in ("device", "device_name"):
        expected.pop(key, None)
        if not isinstance(actual.pop(key, None), str):
            raise PingstoreError("missing recorded worker device identity")
    if actual != expected:
        raise PingstoreError("collected execution request or runtime source changed")
    requested = actual["requested_device"]
    resolved = frozen["runtime"]["device"]
    if requested != "auto" and resolved != requested:
        raise PingstoreError("worker device differs from explicit requested device")
    return frozen


def _complete(bank, evidence, cfg, data, root, run_id, collect):
    directory = _shard_paths(REPO, run_id, 0, recipe.SHARDS)
    collection_protocol = (
        _collection_protocol(directory, cfg, data)
        if collect
        else {"recipe": cfg, **data["protocol"]}
    )
    with concurrent_compute.collect_shards(
        repo=REPO,
        experiment=recipe.SLUG,
        run_id=run_id,
        count=recipe.SHARDS,
        inputs={"bank": bank.reference},
        configuration=collection_protocol,
        items_for=lambda index: recipe.jobs(cfg)[index :: recipe.SHARDS],
        inventory_for=lambda index: {
            "requests": file_sha256(
                _shard_paths(REPO, run_id, index, recipe.SHARDS)
                / ".scratch/shards"
                / str(index)
                / "requests.json"
            ),
            **{
                job["id"]: file_sha256(
                    _job_path(
                        _shard_paths(REPO, run_id, index, recipe.SHARDS) / "export", job
                    )
                )
                for job in recipe.jobs(cfg)[index :: recipe.SHARDS]
            },
        },
        collect=collect,
    ) as (_directory, shard_records):
        with inputs.execution(
            REPO, "compute", sources={"bank": bank}, run_id=run_id, configuration=cfg
        ) as run:
            if collect:
                concurrent_compute.retain_worker_provenance(run, shard_records)
            environment = {
                "PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0",
                **(
                    {"PINGLAB_DEVICE": os.environ["PINGLAB_DEVICE"]}
                    if os.environ.get("PINGLAB_DEVICE")
                    else {}
                ),
            }
            run.record["execution"]["environment"] = environment
            run.record["execution"].update(
                executor=cfg["executor"],
                **data["protocol"],
                parameter_binding=dict(recipe.CHECKPOINT_PARAMETERS),
            )
            requests = []
            if collect:
                for index in range(recipe.SHARDS):
                    requests.extend(
                        load_json(run.scratch / "shards" / str(index) / "requests.json")
                    )
            if not collect:
                _run_jobs(bank, cfg, data, root, run.export, recipe.jobs(cfg), requests)
            train_dir = bank.unit(recipe.cell_name(cfg["raster"]["seed"]))
            for name, snapshot in simulation.recordings(
                train_dir,
                cfg,
                data,
                bank.reference,
                requests,
            ):
                np.savez_compressed(run.export / f"{name}.npz", **snapshot)
            run.record["execution"]["requests"] = requests
            shared_baseline = run.directory / ".baseline-scratch"
            if shared_baseline.exists():
                shutil.rmtree(shared_baseline)
            write_json_atomic(
                run.export / "evidence.json",
                {
                    "schema": "exp042.compute/v4",
                    "recipe": cfg,
                    "bank_evidence": evidence,
                    "jobs": recipe.jobs(cfg),
                    "recordings": ["cycle.npz", "cell.npz"],
                },
            )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="explicit completed exp022 compute ID"
    )
    parser.add_argument("--run-id", help="unused v4 reservation")
    parser.add_argument(
        "--shard-index", type=int, help="compute worker index (eight shards)"
    )
    args = parser.parse_args()
    try:
        if args.shard_index is not None:
            if not args.run_id:
                raise PingstoreError("shard workers require --run-id")
            shard(args.source, run_id=args.run_id, index=args.shard_index)
        else:
            compute(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, ValueError) as exc:
        parser.exit(1, f"exp042 compute: {exc}\n")


if __name__ == "__main__":
    main()
