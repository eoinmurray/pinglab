"""Compute the reduced interventions from an explicit v4 bank; never analyse or present."""

import argparse
import contextlib
import fcntl
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp042 import inputs, recipe
from experiments.exp042.simulation import Simulator
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    run_root,
    write_json_atomic,
)
from pingstore.stages import _capture_code, reserve_stage, stage_reservation, utc_now


def _job_path(export, job):
    return export / "jobs" / (job["id"] + ".json")


def _run_jobs(simulator, bank, export, jobs):
    for job in jobs:
        metrics = simulator.evaluate(bank.export / job["cell"], job)
        record = {"job": job, "metrics": metrics}
        canonical = recipe.replay_job(job)
        if canonical != job:
            record["replay_of"] = canonical["id"]
        write_json_atomic(_job_path(export, job), record)


def _shard_paths(repo, run_id, index, count):
    if count != recipe.SHARDS or not 0 <= index < count:
        raise PingstoreError("exp042 requires eight shards and an index in [0, 8)")
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
        raise PingstoreError("shards require an unused exp042 v4 compute reservation")
    return directory


@contextlib.contextmanager
def _compute_lock(directory, *, exclusive):
    path = directory / ".scratch/compute.lock"
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
    inputs.bank_evidence(bank)
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    directory = _shard_paths(REPO, run_id, index, count)
    with _compute_lock(directory, exclusive=False):
        folder = directory / ".scratch" / "shards" / str(index)
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
                    "distributed exp042 compute requires committed execution code"
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
            with tempfile.TemporaryDirectory(
                prefix=f".scratch-{index}-", dir=directory
            ) as tmp:
                simulator = Simulator(
                    Path(tmp),
                    folder / "commands",
                    cfg,
                    baseline_root=directory / ".baseline-scratch",
                )
                _run_jobs(simulator, bank, directory / "export", job_list)
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
                "files": {
                    job["id"]: file_sha256(_job_path(directory / "export", job))
                    for job in job_list
                },
            }
            write_json_atomic(marker, record)
            return record
        finally:
            lock.unlink()


def _verify_shard(directory, record, jobs):
    if set(record.get("files", {})) != {job["id"] for job in jobs}:
        raise PingstoreError("incomplete shard inventory")
    for job in jobs:
        path = _job_path(directory / "export", job)
        if (
            path.is_symlink()
            or not path.is_file()
            or file_sha256(path) != record["files"][job["id"]]
            or load_json(path).get("job") != job
        ):
            raise PingstoreError("shard payload changed or is incomplete")


def compute(identity, *, run_id=None, collect=False):
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    evidence = inputs.bank_evidence(bank)
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    if collect and not run_id:
        raise PingstoreError("collection requires an explicit compute reservation")
    run_id = run_id or reserve_stage(REPO / ".pingstore", recipe.SLUG, "compute")
    directory = _shard_paths(REPO, run_id, 0, recipe.SHARDS)
    with _compute_lock(directory, exclusive=True):
        _shard_paths(REPO, run_id, 0, recipe.SHARDS)
        if not collect and (directory / ".scratch/shards").exists():
            raise PingstoreError("sharded work requires explicit --collect")
        if collect:
            directory = _shard_paths(REPO, run_id, 0, recipe.SHARDS)
            if list((directory / ".scratch/shards").glob("*/writer.lock")):
                raise PingstoreError("compute shards are still running")
            for index in range(recipe.SHARDS):
                record = load_json(
                    directory / ".scratch/shards" / str(index) / "completed.json"
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
                        run.scratch / "shards" / str(index) / "completed.json"
                    )
                    if marker["source"] != run.record["provenance"]:
                        raise PingstoreError(
                            "worker and collector execution code differ"
                        )
            environment = {"PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0"}
            run.record["execution"]["environment"] = environment
            with tempfile.TemporaryDirectory(
                prefix=".scratch-", dir=run.directory
            ) as tmp:
                simulator = Simulator(Path(tmp), run.scratch / "commands", cfg)
                if not collect:
                    _run_jobs(simulator, bank, run.export, recipe.jobs(cfg))
                raster = cfg["raster"]
                seed, sigma = raster["seed"], raster["sigma_ms"]
                train_dir = bank.export / recipe.cell_name(seed)
                for name, condition, offset in (
                    ("cycle", f"jitter_sigma_{sigma:g}", seed + int(sigma)),
                    ("cell", f"cell_jitter_sigma_{sigma:g}", seed + int(sigma * 13)),
                ):
                    snapshot = simulator.recording(train_dir, condition, offset)
                    # Lossless, compact raw spikes only; voltage/current tensors stay scratch.
                    np.savez_compressed(run.export / f"{name}.npz", **snapshot)
                simulator.cache.clear()
            shared_baseline = run.directory / ".baseline-scratch"
            if shared_baseline.exists():
                shutil.rmtree(shared_baseline)
            write_json_atomic(
                run.export / "evidence.json",
                {
                    "schema": "exp042.compute/v1",
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
    parser.add_argument(
        "--collect",
        action="store_true",
        help="complete this compute run from its eight shards",
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
    except (PingstoreError, OSError, ValueError) as exc:
        parser.exit(1, f"exp042 compute: {exc}\n")


if __name__ == "__main__":
    main()
