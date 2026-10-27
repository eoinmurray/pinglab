"""Deploy frozen experiment sources to CSD3 and submit one explicit stage.

Run ``uv run python -m tools.hpc deploy --help`` for the public interface.
Deployment previews are local; only --live uploads sources and submits Slurm
work. Receipts retain uncertain submissions for manual scheduler reconciliation.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from pingstore.library import PingstoreError, load_json, source_run, write_json_atomic

REPO = Path(__file__).resolve().parents[2]


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def source_inputs(root, identities):
    """Authenticate selected inputs and their complete, explicit ancestry."""
    sources, active = {}, set()

    def visit(identity, reference=None):
        if identity in active:
            raise PingstoreError("cycle in source ancestry")
        source = source_run(root, identity, reference=reference)
        if identity in sources:
            if source.reference != sources[identity].reference:
                raise PingstoreError("conflicting input references")
            return
        active.add(identity)
        for pin in source.record["inputs"].values():
            visit(pin["run_id"], pin)
        active.remove(identity)
        sources[identity] = source

    for identity in identities:
        visit(identity)
    return sources


def plan(args, repo=REPO):
    """Resolve a shell-safe remote deployment without contacting CSD3."""
    if not re.fullmatch(r"exp[0-9]{3,}", args.experiment):
        raise ValueError("experiment must be an expXXX identifier")
    entrypoint = f"experiments/{args.experiment}/{args.stage}.py"
    if not (repo / entrypoint).is_file():
        raise ValueError(f"missing entrypoint: {entrypoint}")
    remote = PurePosixPath(args.remote_dir)
    if not remote.is_absolute() or ".." in remote.parts or str(remote) == "/":
        raise ValueError("--remote-dir must be an absolute, new deployment directory")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", args.host):
        raise ValueError("invalid SSH host")
    if args.cpus < 1 or args.memory_gb < 1 or args.gpus < 0:
        raise ValueError("cpus/memory must be positive and gpus nonnegative")
    if not re.fullmatch(r"(?:[0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]", args.walltime):
        raise ValueError("walltime must be [days-]hours:minutes:seconds")
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    commit = run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    account, qos = {
        "SL2": ("OLEARY-SL2-GPU", "gpu1"),
        "SL3": ("OLEARY-SL3-GPU", "gpu2"),
    }[args.service]
    command = [
        "sbatch",
        "--parsable",
        f"--account={account}",
        "--partition=ampere",
        f"--qos={qos}",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name={args.experiment}-{args.stage}",
        f"--time={args.walltime}",
        f"--cpus-per-task={args.cpus}",
        f"--mem={args.memory_gb}G",
        "--export=NONE",
        f"--chdir={remote / 'repo'}",
        f"--output={remote}/logs/%x-%j.out",
        f"--error={remote}/logs/%x-%j.err",
    ]
    if args.gpus:
        command.append(f"--gres=gpu:{args.gpus}")
    command.append(str(remote / "job.sbatch"))
    return dict(
        schema="pinglab.hpc-deployment/v1",
        host=args.host,
        remote_dir=str(remote),
        commit=commit,
        experiment=args.experiment,
        stage=args.stage,
        entrypoint=entrypoint,
        arguments=arguments,
        cpus=args.cpus,
        gpus=args.gpus,
        inputs=args.input,
        command=command,
    )


def job_script(payload):
    """Install locked dependencies and execute only inside the allocation."""
    directory = shlex.quote(payload["remote_dir"] + "/repo")
    return f"""#!/bin/bash
set -euo pipefail
cd {directory}
source tools/hpc/load-wilkes-modules.sh
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS={payload["cpus"]}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export SNNLAB_DEVICE={"cuda" if payload["gpus"] else "cpu"}
export PINGLAB_NO_SYNC=1
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
exec .venv/bin/python -m tools.hpc worker ../deployment.json
"""


def deploy(args, repo=REPO):
    payload = plan(args, repo)
    print(json.dumps(payload, indent=2), flush=True)
    if not args.live:
        print(
            "Preview only. No upload or jobs submitted. Commit all deployment sources before --live."
        )
        return payload
    # Include untracked files in this check: a new experiment omitted from a
    # committed snapshot would otherwise silently disappear during deployment.
    dirty = run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError(
            "commit deployment sources first; the checkout has uncommitted changes"
        )
    if args.receipt.exists():
        raise ValueError(
            "deployment already attempted; inspect receipt and Slurm before recovery"
        )
    sources = source_inputs(repo / ".pingstore", args.input)
    payload["input_refs"] = [source.reference for source in sources.values()]
    with tempfile.TemporaryDirectory(prefix="pinglab-hpc-") as temporary:
        staging = Path(temporary)
        run(
            ["git", "bundle", "create", str(staging / "source.bundle"), "HEAD"],
            cwd=repo,
        )
        (staging / "deployment.json").write_text(json.dumps(payload, indent=2) + "\n")
        (staging / "job.sbatch").write_text(job_script(payload))
        archive = staging / "upload.tar"
        with tarfile.open(archive, "w") as tar:
            for name in ("source.bundle", "deployment.json", "job.sbatch"):
                tar.add(staging / name, arcname=name)
            for identity, source in sources.items():
                tar.add(source.directory, arcname=f"inputs/{identity}")
        for source in sources.values():
            source.check_unchanged()
        directory = shlex.quote(payload["remote_dir"])
        parent = shlex.quote(str(PurePosixPath(payload["remote_dir"]).parent))
        # mkdir without -p claims a fresh destination; never overwrite a checkout.
        upload = (
            f"mkdir -p {parent} && mkdir {directory} && cd {directory} && tar -xf -"
        )
        receipt = dict(payload, status="uploading", job_id=None)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents two local invocations racing to submit.
        with args.receipt.open("x") as handle:
            json.dump(receipt, handle, indent=2)
        with archive.open("rb") as handle:
            run(["ssh", payload["host"], upload], stdin=handle)
        setup = (
            f"cd {directory} && git clone source.bundle repo && "
            f"git -C repo checkout --detach {shlex.quote(payload['commit'])} && "
            "mkdir logs && "
            "if test -d inputs; then mkdir -p repo/.pingstore/runs && "
            "mv inputs/* repo/.pingstore/runs/; fi"
        )
        checkout = payload["remote_dir"] + "/repo"
        run(["ssh", payload["host"], setup])
        command = payload["command"]
        receipt.update(status="submitting", command=command, checkout=checkout)
        write_json_atomic(args.receipt, receipt)
        result = run(
            ["ssh", payload["host"], shlex.join(command)],
            capture_output=True,
            text=True,
        )
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise ValueError(
                "ambiguous sbatch response; inspect receipt and scheduler before recovery"
            )
        receipt.update(status="submitted", job_id=job_id)
        write_json_atomic(args.receipt, receipt)
        print(
            f"Submitted {job_id}; receipt: {args.receipt}; outputs: {checkout}/.pingstore",
            flush=True,
        )
        return receipt


def worker(path):
    payload = load_json(path)
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("worker requires a Slurm allocation")
    commit = run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    if commit != payload["commit"]:
        raise ValueError("deployed commit differs from submission")
    dirty = run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    if dirty:
        raise ValueError("deployed checkout has uncommitted changes")
    for reference in payload["input_refs"]:
        source_run(Path(".pingstore"), reference["run_id"], reference=reference)
    run([sys.executable, payload["entrypoint"], *payload["arguments"]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="action", required=True)
    launch = commands.add_parser(
        "deploy", help="preview or upload and submit an experiment", allow_abbrev=False
    )
    launch.add_argument("--experiment", required=True)
    launch.add_argument(
        "--stage", choices=("compute", "analyse", "present"), default="compute"
    )
    launch.add_argument("--host", default="csd3")
    launch.add_argument(
        "--remote-dir", required=True, help="new absolute deployment directory on CSD3"
    )
    launch.add_argument("--service", choices=("SL2", "SL3"), default="SL3")
    launch.add_argument("--walltime", required=True)
    launch.add_argument("--cpus", type=int, default=4)
    launch.add_argument("--memory-gb", type=int, default=32)
    launch.add_argument("--gpus", type=int, default=1)
    launch.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="RUN_ID",
        help="transfer a validated local run and its ancestry; repeatable",
    )
    launch.add_argument(
        "--receipt", type=Path, default=REPO / ".pingstore/hpc-deployment.json"
    )
    launch.add_argument("--live", action="store_true")
    launch.add_argument(
        "arguments", nargs=argparse.REMAINDER, help="experiment arguments after --"
    )
    execute = commands.add_parser("worker", help=argparse.SUPPRESS)
    execute.add_argument("plan", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "worker":
            worker(args.plan)
        else:
            deploy(args)
    except (OSError, ValueError, PingstoreError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"HPC: {exc}\n")


if __name__ == "__main__":
    main()
