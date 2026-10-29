"""Run `uv run python -m libraries.modal_dispatcher --help` from the repository root."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .request import execution_environment, validate_dispatch

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Durable Modal compute for any Pingstore experiment.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser(
        "dispatch",
        help="launch paid compute (requires Modal authentication)",
        allow_abbrev=False,
    )
    launch.add_argument("--experiment", required=True)
    launch.add_argument(
        "--gpu", default="A10G", help="Modal GPU type/count, or none for CPU"
    )
    launch.add_argument("--cpu", type=int, default=4)
    launch.add_argument("--memory", type=int, default=16384, help="RAM in MiB")
    launch.add_argument(
        "--timeout", type=int, default=86400, help="worker timeout in seconds"
    )
    launch.add_argument("--root", type=Path, default=REPO / ".pingstore")
    launch.add_argument("--input", action="append", default=[], metavar="ROLE=RUN_ID")
    launch.add_argument(
        "--smoke",
        action="store_true",
        help="set PINGLAB_SMOKE for experiments supporting it",
    )
    launch.add_argument(
        "arguments", nargs=argparse.REMAINDER, help="experiment arguments after --"
    )
    status = commands.add_parser(
        "check", help="check status and optionally collect results", allow_abbrev=False
    )
    status.add_argument("--run-id", required=True)
    status.add_argument("--collect", action="store_true")
    status.add_argument("--root", type=Path, default=REPO / ".pingstore")
    args = parser.parse_args(argv)
    if args.command == "check":
        from .jobs import check

        print(
            json.dumps(
                check(
                    args.root.expanduser().absolute(), args.run_id, collect=args.collect
                )
            )
        )
        return
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    environment = execution_environment(args.gpu, args.cpu, args.smoke)
    inputs = {}
    for item in args.input:
        role, separator, identity = item.partition("=")
        if not separator or not role or not identity or role in inputs:
            parser.error("each --input must be a distinct ROLE=RUN_ID")
        inputs[role] = identity
    validate_dispatch(
        REPO,
        args.experiment,
        arguments,
        inputs,
        environment,
        args.cpu,
        args.memory,
        args.timeout,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "run",
            "--detach",
            str(Path(__file__).with_name("app.py")),
            "--experiment",
            args.experiment,
            "--gpu",
            args.gpu,
            "--cpu",
            str(args.cpu),
            "--memory",
            str(args.memory),
            "--timeout",
            str(args.timeout),
            "--store",
            str(args.root.expanduser().absolute()),
            "--arguments",
            json.dumps(arguments),
            "--inputs",
            json.dumps(inputs),
            "--environment",
            json.dumps(environment),
        ],
        cwd=REPO,
        check=True,
    )


if __name__ == "__main__":
    main()
