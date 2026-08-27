"""Narrow, read-only Pingstore integration commands."""

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
    args = parser.parse_args(argv)
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
