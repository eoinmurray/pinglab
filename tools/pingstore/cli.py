"""Narrow Pingstore discovery, presentation and pruning commands."""

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
        description="Read-only integration with immutable Pingstore runs.",
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
    presentation = commands.add_parser(
        "presentation-inputs",
        help="prepare validated JSON for Pinglab's Typst run view",
    )
    presentation.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="lab directory (default: current working directory)",
    )
    prune = commands.add_parser(
        "prune",
        help="remove superseded runs while retaining HPC and latest visible lineages",
    )
    prune.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="lab directory (default: current directory)",
    )
    action = prune.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--dry-run", action="store_true", help="print the exact immutable prune plan"
    )
    action.add_argument(
        "--confirm", metavar="PLAN_HASH", help="apply an unchanged dry-run plan"
    )
    args = parser.parse_args(argv)
    if args.command == "presentation-inputs":
        from .presentation_inputs import prepare

        return prepare(args.root)
    if args.command == "prune":
        from .prune import apply_plan, build_plan, render_plan

        try:
            if args.dry_run:
                print(render_plan(build_plan(args.root)))
            else:
                plan = apply_plan(args.root, args.confirm)
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
