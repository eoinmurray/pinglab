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
        "--all",
        action="store_true",
        help="consider all experiments (the default)",
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
    papers = commands.add_parser(
        "backup-papers", help="retired: public PDF uploads are disabled"
    )
    papers.add_argument("--source", type=Path, default=Path("papers"))
    papers.add_argument("--destination", default=f"{remote}:{bucket}/papers")
    paper_action = papers.add_mutually_exclusive_group(required=True)
    paper_action.add_argument("--dry-run", action="store_true")
    paper_action.add_argument("--confirm", metavar="PLAN_HASH")
    evict = commands.add_parser(
        "evict", help="remove verified local backup copies, including report-used runs"
    )
    evict.add_argument("--root", type=Path, default=Path("."))
    evict.add_argument("--destination", default=f"{remote}:{bucket}/pingstore/runs")
    evict.add_argument("--asset-base", default="https://assets.eoinmurray.info")
    evict_scope = evict.add_mutually_exclusive_group(required=True)
    evict_scope.add_argument("--all", action="store_true")
    evict_scope.add_argument("--run", action="append", dest="run_ids")
    evict_action = evict.add_mutually_exclusive_group(required=True)
    evict_action.add_argument("--dry-run", action="store_true")
    evict_action.add_argument("--confirm", metavar="PLAN_HASH")
    args = parser.parse_args(argv)
    if args.command == "evict":
        import subprocess

        from .evict import apply_plan as apply_eviction
        from .evict import build_plan as plan_eviction
        from .evict import render_plan as render_eviction

        try:
            if args.dry_run:
                print(
                    render_eviction(
                        plan_eviction(
                            args.root, args.destination, args.run_ids, args.asset_base
                        )
                    )
                )
            else:
                result = apply_eviction(
                    args.root,
                    args.destination,
                    args.confirm,
                    args.run_ids,
                    args.asset_base,
                )
                print(
                    f"Evicted {len(result['evict'])} local copies; R2 backups retained."
                )
            return 0
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            print(f"pingstore evict: {exc}", file=sys.stderr)
            return 1
    if args.command == "backup-papers":
        import subprocess

        from .papers import apply_plan as apply_papers
        from .papers import build_plan as plan_papers

        try:
            if args.dry_run:
                print(json.dumps(plan_papers(args.source, args.destination), indent=2))
            else:
                result = apply_papers(args.source, args.destination, args.confirm)
                print(f"Archived and verified {len(result['files'])} PDFs.")
            return 0
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            print(f"pingstore backup-papers: {exc}", file=sys.stderr)
            return 1
    if args.command == "backup":
        from .backup import backup_runs

        source = args.source.expanduser().absolute()
        if not source.is_dir():
            backup.error("source must be an existing Pingstore runs directory")
        destination = args.destination.rstrip("/")
        if ":" not in destination or not destination.endswith("/pingstore/runs"):
            backup.error(
                "destination must be an rclone remote ending in /pingstore/runs"
            )
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
