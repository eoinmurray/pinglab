"""Narrow Pingstore discovery, backup and pruning commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .discovery import discover_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pingstore",
        description="Validate immutable Pingstore runs, back them up and prune unused, unprotected runs.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser(
        "discover", help="emit validated runs as Demolab discovery JSON"
    )
    discover.add_argument(
        "--source",
        type=Path,
        help="runs directory (default: DEMOLAB_PREVIEW_SOURCE, then .pingstore/runs)",
    )
    backup = commands.add_parser(
        "backup", help="append verified complete runs to an R2 backup using rclone"
    )
    backup.add_argument(
        "--source",
        type=Path,
        default=Path(".pingstore/runs"),
        help="runs directory (default: .pingstore/runs in the current directory)",
    )
    remote = os.environ.get("PINGLAB_R2_REMOTE", "r2")
    bucket = os.environ.get("PINGLAB_R2_BUCKET", "pinglab")
    backup.add_argument(
        "--destination",
        default=f"{remote}:{bucket}/pingstore/runs",
        help="rclone remote ending in /pingstore/runs (default: r2:pinglab/pingstore/runs; PINGLAB_R2_REMOTE and PINGLAB_R2_BUCKET override)",
    )
    backup_action = backup.add_mutually_exclusive_group(required=True)
    backup_action.add_argument(
        "--dry-run",
        action="store_true",
        help="inspect the exact backup plan without uploading",
    )
    backup_action.add_argument(
        "--confirm", metavar="PLAN_HASH", help="apply an unchanged dry-run plan"
    )
    prune = commands.add_parser(
        "prune",
        help="remove unused runs while retaining report and manually protected ancestry",
    )
    prune.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="lab directory (default: current directory)",
    )
    scope = prune.add_mutually_exclusive_group()
    scope.add_argument(
        "--all", action="store_true", help="consider all experiments (the default)",
    )
    scope.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        metavar="EXPNNN",
        help="limit pruning to this experiment (repeatable)",
    )
    prune.add_argument(
        "--stage",
        action="append",
        dest="stages",
        choices=("compute", "analyse", "present"),
        help="limit pruning to this stage (repeatable); other stages stay protected",
    )
    action = prune.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--dry-run", action="store_true", help="print the exact immutable prune plan"
    )
    action.add_argument(
        "--confirm", metavar="PLAN_HASH", help="apply an unchanged dry-run plan"
    )
    args = parser.parse_args(argv)
    if args.command == "backup":
        from .backup import backup_runs

        source = args.source.expanduser().absolute()
        if not source.is_dir():
            backup.error("source must be an existing Pingstore runs directory")
        destination = args.destination.rstrip("/")
        if ":" not in destination or not destination.endswith("/pingstore/runs"):
            backup.error("destination must be an rclone remote ending in /pingstore/runs")
        return backup_runs(
            source, destination, dry_run=args.dry_run, confirm=args.confirm
        )
    if args.command == "prune":
        from .prune import apply_plan, build_plan, render_plan

        try:
            if args.dry_run:
                print(
                    render_plan(
                        build_plan(
                            args.root,
                            args.experiments,
                            args.stages,
                        )
                    )
                )
            else:
                plan = apply_plan(
                    args.root,
                    args.confirm,
                    args.experiments,
                    args.stages,
                )
                reclaimed = sum(row["bytes"] for row in plan["prune"])
                print(
                    f"Pruned {len(plan['prune'])} runs ({reclaimed / 2**30:.2f} GiB)."
                )
            return 0
        except (OSError, ValueError) as exc:
            print(f"pingstore prune: {exc}", file=sys.stderr)
            return 1
    source = args.source or Path(
        os.environ.get("DEMOLAB_PREVIEW_SOURCE") or ".pingstore/runs"
    )
    try:
        records = discover_runs(source)
    except (OSError, ValueError) as exc:
        print(f"pingstore discover: {exc}", file=sys.stderr)
        return 1
    # A single JSON document, only after all candidates passed. No partial output.
    print(json.dumps(records, ensure_ascii=False))
    return 0
