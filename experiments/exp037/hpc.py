"""Prepare and review one exp037 Slurm run; submission requires --live."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp037 import evidence, inputs, recipe
from experiments.helpers.slurm_submit import submit_pipeline
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import _capture_code, reserve_stage, stage_reservation

STAGES = ("compute", "analyse", "present")


def check_plan(plan):
    if plan.get("schema") != "exp037.hpc/v1" or plan.get("repo") != str(REPO):
        raise PingstoreError("plan belongs to another schema or checkout")
    code = _capture_code(REPO, REPO)
    if code.get("code_dirty") or code != plan["code"]:
        raise PingstoreError("HPC execution requires the exact frozen plan source")
    if plan["recipe"] != recipe.configuration(smoke=plan["profile"] == "smoke"):
        raise PingstoreError("HPC recipe changed")
    cfg = plan["recipe"]
    expected_items = [job["id"] for job in recipe.jobs(cfg)]
    expected_partitions = [
        [job["id"] for job in recipe.shard_jobs(cfg, index)]
        for index in range(recipe.SHARDS)
    ]
    if (
        plan.get("work_items") != expected_items
        or plan.get("partitions") != expected_partitions
        or sorted(item for shard in expected_partitions for item in shard)
        != sorted(expected_items)
    ):
        raise PingstoreError("HPC work-item allocation changed")
    source = inputs.source(
        REPO,
        plan["bank"]["run_id"],
        "compute",
        experiment="exp022",
        reference=plan["bank"],
    )
    contract = evidence.training_contract(source.export)
    evidence.histories(source.export, contract)
    if not (Path(plan["mnist_cache"]) / "MNIST/raw/t10k-images-idx3-ubyte").is_file():
        raise PingstoreError("persistent MNIST cache is missing")
    return source


def prepare(args):
    if args.plan.exists():
        raise PingstoreError("plan already exists; review it instead of overwriting")
    code = _capture_code(REPO, REPO)
    if code.get("code_dirty"):
        raise PingstoreError("commit execution code before preparing HPC work")
    source = inputs.source(REPO, args.source, "compute", experiment="exp022")
    cfg = recipe.configuration(smoke=args.profile == "smoke")
    plan = {
        "schema": "exp037.hpc/v1",
        "repo": str(REPO),
        "code": code,
        "bank": source.reference,
        "profile": args.profile,
        "recipe": cfg,
        "work_items": [job["id"] for job in recipe.jobs(cfg)],
        "partitions": [
            [job["id"] for job in recipe.shard_jobs(cfg, index)]
            for index in range(recipe.SHARDS)
        ],
        "account": args.account,
        "cpu_account": args.cpu_account,
        "cpu_partition": "icelake",
        "partition": "ampere",
        "mnist_cache": str(args.mnist_cache.resolve()),
        "walltime": args.walltime,
        "cpus": 4,
        "memory_gb": 32,
    }
    check_plan(plan)
    identities = {
        stage: reserve_stage(REPO / ".pingstore", recipe.SLUG, stage, origin="slurm")
        for stage in STAGES
    }
    plan["runs"] = identities
    # Bind the reserved identities to the reviewed bank before submission.
    for stage, identity in identities.items():
        path = REPO / ".pingstore/runs" / f".{identity}.tmp/.reservation.json"
        record = load_json(path)
        record["inputs"] = {"bank": source.reference}
        write_json_atomic(path, record)
    write_json_atomic(args.plan, plan)
    print(json.dumps(plan, indent=2))


def command(plan, path, stage, dependency=None):
    logs = path.parent / "logs"
    args = [
        "sbatch",
        "--parsable",
        f"--account={plan['account'] if stage == 'compute' else plan['cpu_account']}",
        f"--partition={plan['partition'] if stage == 'compute' else plan['cpu_partition']}",
        "--nodes=1",
        f"--job-name=exp037-{stage}",
        f"--time={plan['walltime'] if stage == 'compute' else '00:30:00'}",
        f"--cpus-per-task={plan['cpus']}",
        f"--mem={plan['memory_gb']}G",
        f"--output={logs}/%x-%A_%a.out",
        f"--error={logs}/%x-%A_%a.err",
        "--export=NONE",
        f"--chdir={REPO}",
    ]
    if stage == "compute":
        args += ["--gres=gpu:1", "--array=0-5%6"]
    if dependency:
        args += [f"--dependency=afterok:{dependency}"]
    args += [str(REPO / "experiments/exp037/slurm.sbatch"), str(path), stage, str(REPO)]
    return args


def review(args):
    plan = load_json(args.plan)
    check_plan(plan)
    for stage, identity in plan["runs"].items():
        reservation = stage_reservation(REPO / ".pingstore/runs" / f".{identity}.tmp")
        if reservation["stage"] != stage or reservation.get("inputs") != {
            "bank": plan["bank"]
        }:
            raise PingstoreError("reservation differs from reviewed plan")
    receipt = args.plan.with_suffix(".submitted.json")
    if args.live or args.test_only:
        (args.plan.parent / "logs").mkdir(exist_ok=True)
    submit_pipeline(
        steps=("compute", "collect", "analyse", "present"),
        command_for=lambda stage, dependency: command(
            plan, args.plan, stage, dependency
        ),
        receipt=receipt,
        live=args.live,
        test_only=args.test_only,
        context={"experiment": recipe.SLUG, "plan": str(args.plan)},
        runner=subprocess.run,
    )


def worker(args):
    plan = load_json(args.plan)
    check_plan(plan)
    os.environ["PINGLAB_SMOKE"] = "1" if plan["profile"] == "smoke" else "0"
    from experiments.exp037 import analyse, compute, present

    ids = plan["runs"]
    if args.stage == "compute":
        import torch

        if not torch.cuda.is_available():
            raise PingstoreError("HPC compute requires a visible CUDA GPU")
        compute.shard(
            plan["bank"]["run_id"],
            run_id=ids["compute"],
            index=int(os.environ["SLURM_ARRAY_TASK_ID"]),
        )
    elif args.stage == "collect":
        compute.compute(plan["bank"]["run_id"], run_id=ids["compute"], collect=True)
    elif args.stage == "analyse":
        analyse.analyse(ids["compute"], run_id=ids["analyse"])
    else:
        present.present(ids["analyse"], run_id=ids["present"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--account", required=True)
    p.add_argument("--cpu-account", required=True)
    p.add_argument("--mnist-cache", type=Path, required=True)
    p.add_argument("--profile", choices=("smoke", "production"), default="production")
    p.add_argument("--walltime", default="02:00:00")
    p = sub.add_parser("review")
    p.add_argument("plan", type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true")
    group.add_argument("--test-only", action="store_true")
    p = sub.add_parser("worker")
    p.add_argument("plan", type=Path)
    p.add_argument("stage", choices=("compute", "collect", "analyse", "present"))
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    try:
        {"prepare": prepare, "review": review, "worker": worker}[args.action](args)
    except (PingstoreError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"exp037 HPC: {exc}\n")


if __name__ == "__main__":
    main()
