"""Compute exp037 perturbations from a pinned v4 bank; includes six-shard execution."""

import argparse
import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp037 import evidence, inputs, recipe
from experiments.helpers import concurrent_compute
from experiments.helpers.run_cli import run_cli
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)
from pingstore.stages import _capture_code, reserve_stage


def _run_jobs(bank, directory, jobs, contract):
    for job in jobs:
        if job.get("level_units"):
            calibration = {
                **job,
                "id": f"calibration__{job['model']}__seed{job['seed']}__drop__0",
                "kind": "calibration",
                "mode": "drop",
                "level": 0.0,
            }
            calibration.pop("level_units")
            m = evidence.metric(
                directory / "export/jobs" / calibration["id"] / "metrics.json",
                contract["configs"][job["cell_name"]],
                calibration,
            )
            job = recipe.resolve_job(
                job,
                float(m["rates_hz"]["hid"]),
                contract["configs"][job["cell_name"]]["dt"],
            )
        output = directory / "export" / job["path"]
        attachments = directory / ".scratch/simulations" / job["path"]
        if output.exists() or attachments.exists():
            raise PingstoreError(
                "incomplete job already exists; explicit recovery or a fresh run is required"
            )
        attachments.mkdir(parents=True)
        train = bank.unit(job["cell_name"])
        shutil.copyfile(train / "config.json", attachments / "training-config.json")
        with tempfile.TemporaryDirectory(prefix=".job-", dir=directory) as tmp:
            scratch = Path(tmp) / "output"
            args = recipe.inference_args(train, train / "weights.pth", scratch, job)
            write_json_atomic(
                attachments / "command.json", {"job": job, "arguments": args}
            )
            print(f"[infer] {job['id']}", flush=True)
            with (
                (attachments / "stdout.log").open("w") as stdout,
                (attachments / "stderr.log").open("w") as stderr,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                run_cli(args, no_sync=True)
            cfg = contract["configs"][job["cell_name"]]
            evidence.inference_config(load_json(scratch / "config.json"), cfg, job)
            evidence.recordings(scratch, cfg, job)
            output.mkdir(parents=True)
            if job["kind"] == "raster":
                (scratch / "recording.npz").rename(output / "recording.npz")
            else:
                (scratch / "metrics.json").rename(output / "metrics.json")
            for p in scratch.iterdir():
                if p.name not in ("recording.npz", "metrics.json"):
                    p.rename(attachments / p.name)
                elif job["kind"] == "raster" and p.name == "metrics.json":
                    p.rename(attachments / p.name)
            evidence.recordings(output, cfg, job)


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
    evidence.histories(bank.export, contract)
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    directory = _shard_paths(REPO, run_id, index, count)
    job_list = recipe.shard_jobs(cfg, index)

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
    evidence.histories(bank.export, contract)
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
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
        items_for=lambda index: recipe.shard_jobs(cfg, index),
        inventory_for=lambda index: _job_inventory(
            _shard_paths(REPO, run_id, index, recipe.SHARDS),
            recipe.shard_jobs(cfg, index),
        ),
        collect=collect,
    ) as (_directory, shard_records):
        with inputs.execution(
            REPO, "compute", sources={"bank": bank}, run_id=run_id, configuration=cfg
        ) as run:
            if collect:
                concurrent_compute.retain_worker_provenance(run, shard_records)
            environment = {"PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0"}
            run.record["execution"]["environment"] = environment
            if not collect:
                _run_jobs(bank, run.directory, recipe.jobs(cfg), contract)
            resolved = evidence.resolved_jobs(
                cfg, contract, lambda j: run.export / j["path"] / "metrics.json"
            )
            for job in resolved:
                train = contract["configs"][job["cell_name"]]
                evidence.inference_config(
                    load_json(
                        run.scratch / "simulations" / job["path"] / "config.json"
                    ),
                    train,
                    job,
                )
                evidence.recordings(run.export / job["path"], train, job)
            write_json_atomic(
                run.export / "evidence.json",
                {
                    "schema": "exp037.compute/v2",
                    "recipe": cfg,
                    "training_contract": contract,
                    "jobs": resolved,
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
        parser.exit(1, f"exp037 compute: {exc}\n")


if __name__ == "__main__":
    main()
