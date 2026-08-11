"""Command-line entrypoint for collection planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .execution import (
    campaign_status,
    initialize_campaign,
    run_local,
    validate_campaign,
)
from .plan import build_plan


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="gamma-gated-sparsity")
    commands = root.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="print the dependency and path plan")
    plan.add_argument("--campaign-root", type=Path, required=True)
    plan.add_argument("--campaign-id", required=True)
    plan.add_argument("--json", action="store_true")
    init = commands.add_parser("init", help="initialize an isolated campaign")
    init.add_argument("--campaign-root", type=Path, required=True)
    init.add_argument("--campaign-id", required=True)
    init.add_argument("--smoke", action="store_true")
    run = commands.add_parser("run", help="run or resume locally in dependency order")
    run.add_argument("--campaign-root", type=Path, required=True)
    status = commands.add_parser("status", help="report validated campaign state")
    status.add_argument("--campaign-root", type=Path, required=True)
    status.add_argument("--json", action="store_true")
    validate = commands.add_parser("validate", help="require all planned outputs")
    validate.add_argument("--campaign-root", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "init":
        payload = initialize_campaign(
            args.campaign_root, args.campaign_id, smoke=args.smoke
        )
        print(Path(payload["campaign_root"]) / "collection-plan.json")
        return
    if args.command == "run":
        run_local(args.campaign_root)
        return
    if args.command in {"status", "validate"}:
        payload = (
            validate_campaign(args.campaign_root)
            if args.command == "validate"
            else campaign_status(args.campaign_root)
        )
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"campaign: {payload['campaign_id']}")
            for row in payload["experiments"]:
                print(
                    f"{row['experiment']}: {row['state']} "
                    f"outputs_valid={str(row['outputs_valid']).lower()}"
                )
        return
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
