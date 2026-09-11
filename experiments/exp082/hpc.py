"""Prepare and review one exp082 Slurm run; submission requires --live."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp082 import evidence, inputs, recipe
from experiments.helpers.hpc.slurm_submit import submit_pipeline
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import _capture_code, reserve_stage, stage_reservation

STAGES = ("compute", "analyse", "present")


def _allocation(cfg):
    jobs = recipe.jobs(cfg)
    return [job["id"] for job in jobs], [
        [job["id"] for job in jobs[index :: recipe.SHARDS]]
        for index in range(recipe.SHARDS)
    ]


def check_plan(plan):
    if plan.get("schema") != "exp082.hpc/v1" or plan.get("repo") != str(REPO):
        raise PingstoreError("plan belongs to another schema or checkout")
    code = _capture_code(REPO, REPO)
    if code.get("code_dirty") or code != plan["code"]:
        raise PingstoreError("HPC execution requires the exact frozen plan source")
    cfg = recipe.configuration(smoke=plan["profile"] == "smoke")
    items, partitions = _allocation(cfg)
    if (
        plan["recipe"] != cfg
        or plan.get("work_items") != items
        or plan.get("partitions") != partitions
        or sorted(item for shard in partitions for item in shard) != sorted(items)
    ):
        raise PingstoreError("HPC recipe or work-item allocation changed")
    bank = inputs.source(
        REPO,
        plan["bank"]["run_id"],
        "compute",
        experiment="exp022",
        reference=plan["bank"],
    )
    contract = evidence.training_contract(bank.export)
    showcase = inputs.source(
        REPO,
        plan["showcase"]["run_id"],
        "compute",
        reference=plan["showcase"],
    )
    showcase_bank, _record = evidence.showcase_evidence(REPO, showcase)
    if showcase_bank.reference != bank.reference:
        raise PingstoreError("showcase and evaluation use different training banks")
    if not contract["cells"]:
        raise PingstoreError("training contract has no cells")
    if not (Path(plan["mnist_cache"]) / "MNIST/raw/t10k-images-idx3-ubyte").is_file():
        raise PingstoreError("persistent MNIST cache is missing")
    for stage, run_id in plan["runs"].items():
        reservation = stage_reservation(REPO / ".pingstore/runs" / f".{run_id}.tmp")
        if reservation["experiment"] != recipe.SLUG or reservation["stage"] != stage:
            raise PingstoreError("reservation differs from reviewed plan")
    return bank, showcase


def prepare(args):
    if args.plan.exists():
        raise PingstoreError("plan already exists; review it instead of overwriting")
    code = _capture_code(REPO, REPO)
    if code.get("code_dirty"):
        raise PingstoreError("commit execution code before preparing HPC work")
    bank = inputs.source(REPO, args.source, "compute", experiment="exp022")
    showcase = inputs.source(REPO, args.showcase, "compute")
    showcase_bank, _record = evidence.showcase_evidence(REPO, showcase)
    if showcase_bank.reference != bank.reference:
        raise PingstoreError("showcase and evaluation use different training banks")
    cfg = recipe.configuration(smoke=args.profile == "smoke")
    items, partitions = _allocation(cfg)
    plan = {
        "schema": "exp082.hpc/v1",
        "repo": str(REPO),
        "code": code,
        "bank": bank.reference,
        "showcase": showcase.reference,
        "profile": args.profile,
        "recipe": cfg,
        "work_items": items,
        "partitions": partitions,
        "runs": {
            stage: reserve_stage(
                REPO / ".pingstore", recipe.SLUG, stage, origin="slurm"
            )
            for stage in STAGES
        },
        "account": args.account,
        "cpu_account": args.cpu_account,
        "partition": args.partition,
        "cpu_partition": args.cpu_partition,
        "mnist_cache": str(args.mnist_cache.resolve()),
        "walltime": args.walltime,
        "collector_walltime": args.collector_walltime,
        "cpus": args.cpus,
        "memory_gb": args.memory_gb,
    }
    check_plan(plan)
    write_json_atomic(args.plan, plan)
    print(json.dumps(plan, indent=2))


def command(plan, path, stage, dependency=None):
    gpu = stage in ("compute", "collect")
    args = [
        "sbatch",
        "--parsable",
        f"--account={plan['account'] if gpu else plan['cpu_account']}",
        f"--partition={plan['partition'] if gpu else plan['cpu_partition']}",
        "--nodes=1",
        f"--job-name=exp082-{stage}",
        f"--time={plan['walltime'] if stage == 'compute' else plan['collector_walltime'] if stage == 'collect' else '00:30:00'}",
        f"--cpus-per-task={plan['cpus']}",
        f"--mem={plan['memory_gb']}G",
        f"--output={path.parent / 'logs'}/%x-%A_%a.out",
        f"--error={path.parent / 'logs'}/%x-%A_%a.err",
        "--export=NONE",
        f"--chdir={REPO}",
    ]
    if gpu:
        args.append("--gres=gpu:1")
    if stage == "compute":
        args.append(f"--array=0-{recipe.SHARDS - 1}%{recipe.SHARDS}")
    if dependency:
        args.append(f"--dependency=afterok:{dependency}")
    return [
        *args,
        str(REPO / "experiments/helpers/hpc/slurm-stage.sbatch"),
        str(path),
        stage,
        str(REPO),
        "gpu" if gpu else "cpu",
        recipe.SLUG,
    ]


def review(args):
    plan = load_json(args.plan)
    check_plan(plan)
    if args.live or args.test_only:
        (args.plan.parent / "logs").mkdir(exist_ok=True)
    submit_pipeline(
        steps=("compute", "collect", "analyse", "present"),
        command_for=lambda stage, dependency: command(
            plan, args.plan, stage, dependency
        ),
        receipt=args.plan.with_suffix(".submitted.json"),
        live=args.live,
        test_only=args.test_only,
        context={"experiment": recipe.SLUG, "plan": str(args.plan)},
        runner=subprocess.run,
    )


def worker(args):
    plan = load_json(args.plan)
    check_plan(plan)
    os.environ["PINGLAB_SMOKE"] = "1" if plan["profile"] == "smoke" else "0"
    for name in (
        "PINGLAB_EXP082_STREAMS_PER_CELL",
        "PINGLAB_EXP082_DIGITS_PER_STREAM",
        "PINGLAB_EXP082_STREAM_BATCH_SIZE",
    ):
        os.environ.pop(name, None)
    from experiments.exp082 import analyse, compute, present

    runs = plan["runs"]
    if args.stage == "compute":
        compute.shard(
            plan["bank"]["run_id"],
            run_id=runs["compute"],
            index=int(os.environ["SLURM_ARRAY_TASK_ID"]),
        )
    elif args.stage == "collect":
        compute.compute(plan["bank"]["run_id"], run_id=runs["compute"], collect=True)
    elif args.stage == "analyse":
        analyse.analyse(
            runs["compute"], plan["showcase"]["run_id"], run_id=runs["analyse"]
        )
    else:
        present.present(runs["analyse"], run_id=runs["present"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepared = sub.add_parser("prepare")
    prepared.add_argument("--source", required=True)
    prepared.add_argument("--showcase", required=True)
    prepared.add_argument("--plan", type=Path, required=True)
    prepared.add_argument("--account", required=True)
    prepared.add_argument("--cpu-account", required=True)
    prepared.add_argument("--mnist-cache", type=Path, required=True)
    prepared.add_argument(
        "--profile", choices=("smoke", "production"), default="production"
    )
    prepared.add_argument("--partition", default="ampere")
    prepared.add_argument("--cpu-partition", default="icelake")
    prepared.add_argument("--walltime", default="04:00:00")
    prepared.add_argument("--collector-walltime", default="01:00:00")
    prepared.add_argument("--cpus", type=int, default=4)
    prepared.add_argument("--memory-gb", type=int, default=32)
    reviewed = sub.add_parser("review")
    reviewed.add_argument("plan", type=Path)
    group = reviewed.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true")
    group.add_argument("--test-only", action="store_true")
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("plan", type=Path)
    worker_parser.add_argument(
        "stage", choices=("compute", "collect", "analyse", "present")
    )
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
        parser.exit(1, f"exp082 HPC: {exc}\n")


if __name__ == "__main__":
    main()
