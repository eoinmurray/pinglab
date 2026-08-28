"""Bounded downstream inference shards for Wilkes campaign execution."""

from __future__ import annotations

import importlib
from typing import Any

SHARD_COUNTS: dict[str, int] = {
    "exp037": 6,
    "exp042": 8,
    "exp082": 6,
}

PRODUCTION_CONTRACTS: dict[str, dict[str, int]] = {
    "exp037": {"condition_jobs": 204, "simulator_launches_max": 204},
    "exp042": {"condition_jobs": 66, "simulator_launches_max": 66},
    "exp082": {
        "condition_jobs": 132,
        "simulator_launches_max": 1_058,
        "classified_presentations": 26_400,
    },
}

SMOKE_CONTRACTS: dict[str, dict[str, int]] = {
    "exp037": {"condition_jobs": 54, "simulator_launches_max": 54},
    "exp042": {"condition_jobs": 39, "simulator_launches_max": 39},
    "exp082": {
        "condition_jobs": 18,
        "simulator_launches_max": 20,
        "classified_presentations": 162,
    },
}


def shard_count(slug: str) -> int:
    return SHARD_COUNTS.get(slug, 1)


def workload_contract(slug: str, *, smoke: bool) -> dict[str, int] | None:
    contract = (SMOKE_CONTRACTS if smoke else PRODUCTION_CONTRACTS).get(slug)
    return dict(contract) if contract is not None else None


def _runner(slug: str) -> Any:
    if slug not in SHARD_COUNTS:
        raise ValueError(f"experiment does not declare sharded inference: {slug}")
    module = importlib.import_module(f"experiments.{slug}")
    for name in ("infer_jobs", "job_is_done", "run_infer_job"):
        if not callable(getattr(module, name, None)):
            raise TypeError(f"{slug} must provide callable {name}")
    return module


def jobs_for_shard(slug: str, index: int, count: int) -> list[str]:
    if count != shard_count(slug):
        raise ValueError(
            f"{slug} requires {shard_count(slug)} shards, received {count}"
        )
    if not 0 <= index < count:
        raise ValueError(f"shard index {index} outside [0, {count})")
    if slug == "exp042":
        import os

        from experiments.exp042 import recipe
        cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
        jobs = [job["id"] for job in recipe.jobs(cfg)]
    else:
        jobs = list(_runner(slug).infer_jobs())
    if len(jobs) != len(set(jobs)):
        raise ValueError(f"{slug} inference job IDs must be unique")
    return jobs[index::count]


def execute_shard(
    slug: str, index: int, count: int, *, smoke: bool
) -> dict[str, object]:
    if count != shard_count(slug):
        raise ValueError(
            f"{slug} requires {shard_count(slug)} shards, received {count}"
        )
    if not 0 <= index < count:
        raise ValueError(f"shard index {index} outside [0, {count})")
    if slug == "exp042":
        raise ValueError("exp042 shards require the staged adapter and an explicit v3 bank")
    runner = _runner(slug)
    all_jobs = list(runner.infer_jobs())
    contract = workload_contract(slug, smoke=smoke)
    if contract is not None and len(all_jobs) != contract["condition_jobs"]:
        raise RuntimeError(
            f"{slug} planned {len(all_jobs)} inference jobs; "
            f"reviewed contract requires {contract['condition_jobs']}"
        )
    jobs = all_jobs[index::count]
    completed: list[str] = []
    reused: list[str] = []
    for job_id in jobs:
        if runner.job_is_done(job_id):
            reused.append(job_id)
            continue
        runner.run_infer_job(job_id)
        if not runner.job_is_done(job_id):
            raise RuntimeError(f"{slug} inference job did not produce output: {job_id}")
        completed.append(job_id)
    return {
        "slug": slug,
        "shard_index": index,
        "shard_count": count,
        "workload_contract": contract,
        "jobs": jobs,
        "completed": completed,
        "reused": reused,
    }
