"""Compute exp082 streaming inference from a pinned v3 bank, with six shards."""

import argparse
import contextlib
import fcntl
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp082 import evidence, inputs, recipe
from experiments.exp082.inference import Inference
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    run_root,
    write_json_atomic,
)
from pingstore.stages import _capture_code, reserve_stage, stage_reservation, utc_now


def _run_jobs(bank, directory, jobs, contract):
    cfg = recipe.environment_configuration()
    worker = Inference(bank, directory, cfg)
    for job in jobs:
        if (directory / "export" / job["path"]).exists() or (
            directory / "export/evidence/simulations" / job["path"]
        ).exists():
            raise PingstoreError(
                "incomplete job exists; explicit recovery or a fresh run required"
            )
        worker.condition(job)
        write_json_atomic(
            directory / "export/evidence/simulations" / job["path"] / "dataset.json",
            worker.dataset,
        )
        evidence.counts(directory / "export" / job["path"] / "counts.npz", cfg)


def _job_inventory(directory, jobs):
    files = {}
    for job in jobs:
        for prefix in ("export", "export/evidence/simulations"):
            folder = directory / prefix / job["path"]
            if not folder.is_dir() or folder.is_symlink():
                raise PingstoreError("missing or linked job evidence")
            for path in folder.rglob("*"):
                if path.is_symlink() or not (path.is_file() or path.is_dir()):
                    raise PingstoreError("unsupported job evidence entry")
                if path.is_file():
                    files[str(path.relative_to(directory))] = file_sha256(path)
    return files


def _verify_shard(directory, record, jobs):
    if record.get("jobs") != [job["id"] for job in jobs]:
        raise PingstoreError("shard job identity differs")
    if record.get("files") != _job_inventory(directory, jobs):
        raise PingstoreError("shard payload changed or is incomplete")


def _shard_paths(repo, run_id, index, count):
    if count != recipe.SHARDS or not 0 <= index < count:
        raise PingstoreError("exp082 requires six shards and an index in [0, 6)")
    destination = run_root(repo / ".pingstore", run_id)
    directory = destination.with_name(f".{run_id}.tmp")
    record = stage_reservation(directory)
    if (
        record["experiment"] != recipe.SLUG
        or record["stage"] != "compute"
        or record["run_id"] != run_id
        or destination.exists()
        or (directory / "run.json").exists()
    ):
        raise PingstoreError("shards require an unused exp082 v3 compute reservation")
    return directory


@contextlib.contextmanager
def _compute_lock(directory, *, exclusive):
    path = directory / "export/evidence/compute.lock"
    if any(p.is_symlink() for p in (directory, *directory.parents, path.parent, path)):
        raise PingstoreError("compute working paths must not use symlinks")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as handle:
        try:
            fcntl.flock(
                handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            raise PingstoreError("compute reservation is busy") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def shard(identity, *, run_id, index, count=recipe.SHARDS):
    """Compute-only recovery: each shard owns an isolated lock and completion record."""
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    contract = evidence.training_contract(bank.export)
    cfg = recipe.environment_configuration()
    directory = _shard_paths(REPO, run_id, index, count)
    with _compute_lock(directory, exclusive=False):
        folder = directory / "export/evidence" / "shards" / str(index)
        folder.mkdir(parents=True, exist_ok=True)
        lock = folder / "writer.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise PingstoreError(
                "shard is busy; interrupted locks need explicit recovery"
            ) from exc
        os.close(descriptor)
        try:
            # Recheck after locking so collection cannot race a newly started worker.
            _shard_paths(REPO, run_id, index, count)
            code = _capture_code(REPO, directory)
            # The campaign requires a frozen checkout; do not permit mixed worker code.
            if code.get("code_dirty"):
                raise PingstoreError(
                    "distributed exp082 compute requires committed execution code"
                )
            job_list = recipe.jobs(cfg)[index::count]
            expected = {
                "run_id": run_id,
                "bank": bank.reference,
                "recipe": cfg,
                "index": index,
                "count": count,
                "source": code,
                "jobs": [job["id"] for job in job_list],
            }
            marker = folder / "completed.json"
            if marker.exists():
                previous = load_json(marker)
                if any(previous.get(k) != v for k, v in expected.items()):
                    raise PingstoreError(
                        "shard source or recipe changed; reserve a fresh compute run"
                    )
                _verify_shard(directory, previous, job_list)
                return previous
            started = utc_now()
            _run_jobs(bank, directory, job_list, contract)
            for ancestor in inputs.lineage(REPO, identity, bank.reference).values():
                ancestor.check_unchanged()
            record = {
                **expected,
                "started_at": started,
                "completed_at": utc_now(),
                "command": [sys.executable, *sys.argv],
                "scheduler": {
                    k: os.environ[k]
                    for k in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID")
                    if k in os.environ
                },
                "files": _job_inventory(directory, job_list),
            }
            write_json_atomic(marker, record)
            return record
        finally:
            lock.unlink()


def compute(identity, *, run_id=None, collect=False):
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    contract = evidence.training_contract(bank.export)
    cfg = recipe.environment_configuration()
    if collect and not run_id:
        raise PingstoreError("collection requires an explicit compute reservation")
    run_id = run_id or reserve_stage(REPO / ".pingstore", recipe.SLUG, "compute")
    directory = _shard_paths(REPO, run_id, 0, recipe.SHARDS)
    with _compute_lock(directory, exclusive=True):
        _shard_paths(REPO, run_id, 0, recipe.SHARDS)
        if not collect and (directory / "export/evidence/shards").exists():
            raise PingstoreError("sharded work requires explicit --collect")
        if collect:
            directory = _shard_paths(REPO, run_id, 0, recipe.SHARDS)
            if list((directory / "export/evidence/shards").glob("*/writer.lock")):
                raise PingstoreError("compute shards are still running")
            for index in range(recipe.SHARDS):
                record = load_json(
                    directory / "export/evidence/shards" / str(index) / "completed.json"
                )
                if (
                    record.get("run_id") != run_id
                    or record.get("bank") != bank.reference
                    or record.get("recipe") != cfg
                    or record.get("index") != index
                    or record.get("count") != recipe.SHARDS
                ):
                    raise PingstoreError("shard bank, recipe or identity mismatch")
                _verify_shard(
                    directory, record, recipe.jobs(cfg)[index :: recipe.SHARDS]
                )
        with inputs.execution(
            REPO, "compute", sources={"bank": bank}, run_id=run_id, configuration=cfg
        ) as run:
            if collect:
                for index in range(recipe.SHARDS):
                    marker = load_json(
                        run.evidence / "shards" / str(index) / "completed.json"
                    )
                    if marker["source"] != run.record["provenance"]:
                        raise PingstoreError(
                            "worker and collector execution code differ"
                        )
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
                        run.evidence / "simulations" / job["path"] / "dataset.json"
                    )
                    != worker.dataset
                ):
                    raise PingstoreError(
                        "inference workers used different dataset bytes"
                    )
            for name in ("matched", "variable"):
                worker.stream(name)
            write_json_atomic(run.evidence / "dataset.json", worker.dataset)
            evidence.validate_compute(run.export, cfg)
            write_json_atomic(
                run.export / "evidence.json",
                {
                    "schema": "exp082.compute/v1",
                    "recipe": cfg,
                    "training_contract": contract,
                    "jobs": recipe.jobs(cfg),
                },
            )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="explicit completed exp022 compute ID"
    )
    parser.add_argument("--run-id", help="unused v3 reservation")
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
