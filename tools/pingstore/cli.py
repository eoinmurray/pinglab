"""Command interface for Pingstore's core storage operations.

Run ``pingstore --help`` for commands. Mutating maintenance commands require a
separate dry-run and its complete confirmation hash; experiment execution uses
``pingstore.library`` instead.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import core


def _parser() -> argparse.ArgumentParser:
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
    remote_prune = commands.add_parser(
        "prune-remote", help="delete unused R2 runs after current evidence is backed up"
    )
    remote_prune.add_argument("--root", type=Path, default=Path("."))
    remote_prune.add_argument(
        "--destination", default=f"{remote}:{bucket}/pingstore/runs"
    )
    remote_prune.add_argument(
        "--experiment", action="append", dest="experiments", required=True
    )
    remote_action = remote_prune.add_mutually_exclusive_group(required=True)
    remote_action.add_argument("--dry-run", action="store_true")
    remote_action.add_argument("--confirm", metavar="PLAN_HASH")
    discover.set_defaults(handler=_discover)
    backup.set_defaults(handler=_backup, argument_error=backup.error)
    prune.set_defaults(handler=_prune)
    papers.set_defaults(handler=_papers)
    evict.set_defaults(handler=_evict)
    remote_prune.set_defaults(handler=_prune_remote)
    return parser


def _discover(args) -> int:
    source = args.source or Path(
        os.environ.get("DEMOLAB_PREVIEW_SOURCE") or ".pingstore/runs"
    )
    records = core.discover_runs(source)
    # A single JSON document, only after all candidates passed. No partial output.
    print(json.dumps(records, ensure_ascii=False))
    return 0


def _backup(args) -> int:
    source = args.source.expanduser().absolute()
    if not source.is_dir():
        args.argument_error("source must be an existing Pingstore runs directory")
    destination = args.destination.rstrip("/")
    if ":" not in destination or not destination.endswith("/pingstore/runs"):
        args.argument_error(
            "destination must be an rclone remote ending in /pingstore/runs"
        )
    return core.backup_runs(
        source, destination, dry_run=args.dry_run, confirm=args.confirm
    )


def _prune(args) -> int:
    if args.dry_run:
        print(
            core.render_prune_plan(
                core.prune_plan(args.root, args.experiments, args.stages)
            )
        )
    else:
        plan = core.apply_prune(args.root, args.confirm, args.experiments, args.stages)
        reclaimed = sum(row["bytes"] for row in plan["prune"])
        print(f"Pruned {len(plan['prune'])} runs ({reclaimed / 2**30:.2f} GiB).")
    return 0


def _papers(args) -> int:
    if args.dry_run:
        print(
            json.dumps(core.paper_backup_plan(args.source, args.destination), indent=2)
        )
    else:
        result = core.apply_paper_backup(args.source, args.destination, args.confirm)
        print(f"Archived and verified {len(result['files'])} PDFs.")
    return 0


def _evict(args) -> int:
    if args.dry_run:
        print(
            core.render_eviction_plan(
                core.eviction_plan(
                    args.root, args.destination, args.run_ids, args.asset_base
                )
            )
        )
    else:
        result = core.apply_eviction(
            args.root, args.destination, args.confirm, args.run_ids, args.asset_base
        )
        print(f"Evicted {len(result['evict'])} local copies; R2 backups retained.")
    return 0


def _prune_remote(args) -> int:
    if args.dry_run:
        print(
            core.render_remote_prune_plan(
                core.remote_prune_plan(args.root, args.destination, args.experiments)
            )
        )
    else:
        result = core.apply_remote_prune(
            args.root, args.destination, args.experiments, args.confirm
        )
        print(f"Pruned {len(result['prune'])} unused remote runs.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"pingstore {args.command}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
