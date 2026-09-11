"""Prepare and review one exp112 Slurm array; submission requires --live."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp112 import recipe
from experiments.helpers.hpc.slurm_submit import submit_pipeline
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import _capture_code, reserve_stage, stage_reservation


def _resolved_cases() -> list[dict]:
    return [recipe.configuration(case) for case in recipe.CASES]


def check_plan(plan: dict) -> None:
    if plan.get("schema") != "exp112.hpc/v2" or plan.get("repo") != str(REPO):
        raise PingstoreError("plan belongs to another schema or checkout")
    code = _capture_code(REPO, REPO)
    if code.get("code_dirty") or code != plan.get("code"):
        raise PingstoreError("HPC execution requires the exact frozen plan source")
    work_items = _resolved_cases()
    if plan.get("work_items") != work_items or plan.get("partitions") != [
        [case["condition"]["id"]] for case in work_items
    ]:
        raise PingstoreError("HPC cell configuration or allocation changed")
    if not (Path(plan["mnist_cache"]) / "MNIST").is_dir():
        raise PingstoreError("persistent MNIST cache is missing")
    runs = plan.get("runs")
    if not isinstance(runs, list) or len(runs) != len(recipe.CASES):
        raise PingstoreError("HPC plan requires one reserved run per condition")
    for run_id in runs:
        reservation = stage_reservation(REPO / ".pingstore/runs" / f".{run_id}.tmp")
        if (
            reservation["experiment"] != recipe.SLUG
            or reservation["stage"] != "compute"
        ):
            raise PingstoreError("condition reservation differs from reviewed plan")


def prepare(args: argparse.Namespace) -> None:
    if args.plan.exists():
        raise PingstoreError("plan already exists; review it instead of overwriting")
    code = _capture_code(REPO, REPO)
    if code.get("code_dirty"):
        raise PingstoreError("commit execution code before preparing HPC work")
    work_items = _resolved_cases()
    plan = {
        "schema": "exp112.hpc/v2",
        "repo": str(REPO),
        "code": code,
        "work_items": work_items,
        "partitions": [[case["condition"]["id"]] for case in work_items],
        "runs": [
            reserve_stage(REPO / ".pingstore", recipe.SLUG, "compute", origin="slurm")
            for _case in recipe.CASES
        ],
        "account": args.account,
        "partition": args.partition,
        "mnist_cache": str(args.mnist_cache.resolve()),
        "walltime": args.walltime,
        "cpus": args.cpus,
        "memory_gb": args.memory_gb,
        "gpus": 1,
        "concurrency": len(recipe.CASES),
    }
    check_plan(plan)
    write_json_atomic(args.plan, plan)
    print(json.dumps(plan, indent=2))


def command(plan: dict, path: Path, dependency: str | None = None) -> list[str]:
    del dependency
    logs = path.parent / "logs"
    return [
        "sbatch",
        "--parsable",
        f"--account={plan['account']}",
        f"--partition={plan['partition']}",
        "--nodes=1",
        "--job-name=exp112-compute",
        f"--time={plan['walltime']}",
        f"--cpus-per-task={plan['cpus']}",
        f"--mem={plan['memory_gb']}G",
        f"--gres=gpu:{plan['gpus']}",
        f"--array=0-{len(recipe.CASES) - 1}%{plan['concurrency']}",
        f"--output={logs}/%x-%A_%a.out",
        f"--error={logs}/%x-%A_%a.err",
        "--export=NONE",
        f"--chdir={REPO}",
        str(REPO / "experiments/helpers/hpc/slurm-stage.sbatch"),
        str(path),
        "compute",
        str(REPO),
        "gpu",
        recipe.SLUG,
    ]


def review(args: argparse.Namespace) -> None:
    plan = load_json(args.plan)
    check_plan(plan)
    if args.live or args.test_only:
        (args.plan.parent / "logs").mkdir(exist_ok=True)
    submit_pipeline(
        steps=("compute",),
        command_for=lambda _step, dependency: command(plan, args.plan, dependency),
        receipt=args.plan.with_suffix(".submitted.json"),
        live=args.live,
        test_only=args.test_only,
        context={"experiment": recipe.SLUG, "plan": str(args.plan)},
        runner=subprocess.run,
    )


def worker(args: argparse.Namespace) -> None:
    plan = load_json(args.plan)
    check_plan(plan)
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= index < len(recipe.CASES):
        raise PingstoreError("Slurm array index is outside the frozen allocation")
    from experiments.exp112 import compute

    compute.compute(index, run_id=plan["runs"][index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--plan", type=Path, required=True)
    prepare_parser.add_argument("--account", required=True)
    prepare_parser.add_argument("--mnist-cache", type=Path, required=True)
    prepare_parser.add_argument("--partition", default="ampere")
    prepare_parser.add_argument("--walltime", default="01:00:00")
    prepare_parser.add_argument("--cpus", type=int, default=4)
    prepare_parser.add_argument("--memory-gb", type=int, default=32)
    review_parser = sub.add_parser("review")
    review_parser.add_argument("plan", type=Path)
    group = review_parser.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true")
    group.add_argument("--test-only", action="store_true")
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("plan", type=Path)
    worker_parser.add_argument("stage", choices=("compute",))
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    try:
        {"prepare": prepare, "review": review, "worker": worker}[args.action](args)
    except (
        KeyError,
        OSError,
        ValueError,
        PingstoreError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(1, f"exp112 HPC: {exc}\n")


if __name__ == "__main__":
    main()
