"""Command-line entrypoint for collection planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .plan import build_plan


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="gamma-gated-sparsity")
    commands = root.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="print the dependency and path plan")
    plan.add_argument("--campaign-root", type=Path, required=True)
    plan.add_argument("--campaign-id", required=True)
    plan.add_argument("--json", action="store_true")
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    payload = build_plan(args.campaign_root, args.campaign_id)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"collection: {payload['collection']}")
    print(f"campaign: {payload['campaign_id']}")
    print(f"root: {payload['campaign_root']}")
    for stage in payload["stages"]:
        names = ", ".join(row["slug"] for row in stage["experiments"])
        print(f"stage {stage['index']}: {names}")
    print(f"executable: {str(payload['executable']).lower()}")
    if not payload["executable"]:
        print("blocked: runners remain to be integrated; see issue #70")


if __name__ == "__main__":
    main()
