"""Prepare and review one exp022 Slurm bank; submission requires --live."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp022 import compute, recipe, reuse
from experiments.helpers.hpc.slurm_submit import submit_pipeline
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import _capture_code


def check_plan(plan):
    if plan.get("schema") != "exp022.hpc/v1" or plan.get("repo") != str(REPO):
        raise PingstoreError("plan belongs to another schema or checkout")
    code = _capture_code(REPO, REPO)
    if code.get("code_dirty") or code != plan["code"]:
        raise PingstoreError("HPC execution requires the exact frozen plan source")
    manifest_path = Path(plan["manifest"])
    manifest = compute._checked_bank_manifest(manifest_path)
    items = manifest["cells"]
    if (
        plan.get("work_items") != items
        or plan.get("partitions") != [[cell["name"]] for cell in items]
        or manifest.get("pingstore_run_id") != plan.get("run_id")
    ):
        raise PingstoreError("HPC cell configuration or allocation changed")
    if plan.get("mode") == "exp110-coba-damping-replacement":
        status = reuse.status(REPO, plan["run_id"])
        if (
            set(status["new_cells"]) != {row["name"] for row in items}
            or len(items) != 18
            or status["source"] != plan.get("source")
        ):
            raise PingstoreError("replacement-bank source or 84/18 partition changed")
    if not (Path(plan["mnist_cache"]) / "MNIST").is_dir():
        raise PingstoreError("persistent MNIST cache is missing")
    if type(plan.get("concurrency")) is not int or plan["concurrency"] < 1:
        raise PingstoreError("HPC concurrency must be a positive integer")
    return manifest


def prepare(args):
    if args.plan.exists():
        raise PingstoreError("plan already exists; review it instead of overwriting")
    if args.root.exists():
        raise PingstoreError("bank working root already exists")
    code = _capture_code(REPO, REPO)
    if code.get("code_dirty"):
        raise PingstoreError("commit execution code before preparing HPC work")
    args.root.mkdir(parents=True)
    if args.replacement == "exp110-coba-damping":
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO / "experiments/exp022/compute.py"),
                "--reuse-reserve",
                "--execution-origin",
                "slurm-wilkes",
            ],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
        run_id = completed.stdout.strip().splitlines()[-1]
        manifest_path = (
            REPO
            / ".pingstore/runs"
            / f".{run_id}.tmp/.scratch/reuse/campaign.json"
        )
        source = reuse.status(REPO, run_id)["source"]
        mode = "exp110-coba-damping-replacement"
    else:
        command = [
            sys.executable,
            str(REPO / "experiments/exp022/compute.py"),
            "--bank-create",
            str(args.root / "bank"),
            "--execution-origin",
            "slurm-wilkes",
        ]
        subprocess.run(command, cwd=REPO, check=True)
        manifest_path = args.root / "bank/bank.json"
        source = None
        mode = "complete-bank"
    manifest = compute._checked_bank_manifest(manifest_path)
    items = manifest["cells"]
    plan = {
        "schema": "exp022.hpc/v1",
        "mode": mode,
        "repo": str(REPO),
        "code": code,
        "manifest": str(manifest_path.resolve()),
        "run_id": manifest["pingstore_run_id"],
        **({"source": source} if source is not None else {}),
        "work_items": items,
        "partitions": [[cell["name"]] for cell in items],
        "account": args.account,
        "partition": args.partition,
        "mnist_cache": str(args.mnist_cache.resolve()),
        "walltime": args.walltime,
        "collector_walltime": args.collector_walltime,
        "cpus": args.cpus,
        "memory_gb": args.memory_gb,
        "concurrency": args.concurrency,
    }
    check_plan(plan)
    write_json_atomic(args.plan, plan)
    print(json.dumps(plan, indent=2))


def command(plan, path, stage, dependency=None):
    args = [
        "sbatch",
        "--parsable",
        f"--account={plan['account']}",
        f"--partition={plan['partition']}",
        "--nodes=1",
        f"--job-name=exp022-{stage}",
        f"--time={plan['walltime'] if stage == 'compute' else plan['collector_walltime']}",
        f"--cpus-per-task={plan['cpus']}",
        f"--mem={plan['memory_gb']}G",
        "--gres=gpu:1",
        f"--output={path.parent / 'logs'}/%x-%A_%a.out",
        f"--error={path.parent / 'logs'}/%x-%A_%a.err",
        "--export=NONE",
        f"--chdir={REPO}",
    ]
    if stage == "compute":
        args.append(f"--array=0-{len(plan['partitions']) - 1}%{plan['concurrency']}")
    if dependency:
        args.append(f"--dependency=afterok:{dependency}")
    return [
        *args,
        str(REPO / "experiments/helpers/hpc/slurm-stage.sbatch"),
        str(path),
        stage,
        str(REPO),
        "gpu",
        recipe.SLUG,
    ]


def review(args):
    plan = load_json(args.plan)
    check_plan(plan)
    if args.live or args.test_only:
        (args.plan.parent / "logs").mkdir(exist_ok=True)
    submit_pipeline(
        steps=("compute", "collect"),
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
    manifest = Path(plan["manifest"])
    if args.stage == "compute":
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        try:
            cell_name = plan["partitions"][index][0]
        except (IndexError, TypeError) as exc:
            raise PingstoreError(
                "Slurm array index is outside the frozen allocation"
            ) from exc
        if plan.get("mode") == "exp110-coba-damping-replacement":
            reuse.train_cell(REPO, plan["run_id"], cell_name)
        else:
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "experiments/exp022/compute.py"),
                    "--bank-train-cell",
                    cell_name,
                    "--bank",
                    str(manifest),
                ],
                cwd=REPO,
                check=True,
            )
    else:
        if plan.get("mode") == "exp110-coba-damping-replacement":
            reuse.finalize(REPO, plan["run_id"])
        else:
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "experiments/exp022/compute.py"),
                    "--bank-finalize",
                    str(manifest),
                ],
                cwd=REPO,
                check=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepared = sub.add_parser("prepare")
    prepared.add_argument("--root", type=Path, required=True)
    prepared.add_argument("--plan", type=Path, required=True)
    prepared.add_argument(
        "--replacement",
        choices=("exp110-coba-damping",),
        help="prepare the exact 84-reused/18-retrained exp110 replacement bank",
    )
    prepared.add_argument("--account", required=True)
    prepared.add_argument("--mnist-cache", type=Path, required=True)
    prepared.add_argument("--partition", default="ampere")
    prepared.add_argument("--walltime", required=True)
    prepared.add_argument("--collector-walltime", default="02:00:00")
    prepared.add_argument("--concurrency", type=int, required=True)
    prepared.add_argument("--cpus", type=int, default=4)
    prepared.add_argument("--memory-gb", type=int, default=32)
    reviewed = sub.add_parser("review")
    reviewed.add_argument("plan", type=Path)
    group = reviewed.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true")
    group.add_argument("--test-only", action="store_true")
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("plan", type=Path)
    worker_parser.add_argument("stage", choices=("compute", "collect"))
    args = parser.parse_args()
    if hasattr(args, "plan"):
        args.plan = args.plan.resolve()
    if hasattr(args, "root"):
        args.root = args.root.resolve()
    if hasattr(args, "mnist_cache"):
        args.mnist_cache = args.mnist_cache.resolve()
    try:
        {
            "prepare": prepare,
            "review": review,
            "worker": worker,
        }[args.action](args)
    except (
        KeyError,
        OSError,
        ValueError,
        PingstoreError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(1, f"exp022 HPC: {exc}\n")


if __name__ == "__main__":
    main()
