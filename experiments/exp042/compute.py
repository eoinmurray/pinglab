"""Compute the reduced interventions from an explicit v4 bank; never analyse or present."""

import argparse
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
from experiments.helpers.hpc import concurrent_compute
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)
from pingstore.stages import _capture_code, reserve_stage


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
    inputs.bank_evidence(bank)
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    directory = _shard_paths(REPO, run_id, index, count)
    job_list = recipe.jobs(cfg)[index::count]

    def run_items():
        folder = directory / ".scratch/shards" / str(index)
        folder.mkdir(parents=True, exist_ok=True)
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

    def inventory():
        return {
            job["id"]: file_sha256(_job_path(directory / "export", job))
            for job in job_list
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
        configuration=cfg,
        items=job_list,
        run_items=run_items,
        inventory=inventory,
        check_inputs=check_inputs,
        capture_code=_capture_code,
    )


def compute(identity, *, run_id=None, collect=False):
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    evidence = inputs.bank_evidence(bank)
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
        items_for=lambda index: recipe.jobs(cfg)[index :: recipe.SHARDS],
        inventory_for=lambda index: {
            job["id"]: file_sha256(
                _job_path(
                    _shard_paths(REPO, run_id, index, recipe.SHARDS) / "export", job
                )
            )
            for job in recipe.jobs(cfg)[index :: recipe.SHARDS]
        },
        collect=collect,
    ) as (_directory, shard_records):
        with inputs.execution(
            REPO, "compute", sources={"bank": bank}, run_id=run_id, configuration=cfg
        ) as run:
            if collect:
                concurrent_compute.retain_worker_provenance(run, shard_records)
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
