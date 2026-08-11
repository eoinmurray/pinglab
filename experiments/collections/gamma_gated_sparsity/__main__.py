"""Command-line entrypoint for collection planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .execution import (
    aggregate_exp022,
    build_publication,
    campaign_status,
    finalize_campaign,
    initialize_campaign,
    run_experiment,
    run_local,
    validate_campaign,
)
from .plan import build_plan
from .slurm import resume_campaign, slurm_status, submit_campaign


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
    aggregate = commands.add_parser(
        "aggregate-exp022", help="run the scheduler's exp022 aggregation step"
    )
    aggregate.add_argument("--campaign-root", type=Path, required=True)
    experiment = commands.add_parser(
        "run-experiment", help="run one scheduler-managed downstream step"
    )
    experiment.add_argument("--campaign-root", type=Path, required=True)
    experiment.add_argument("--slug", required=True)
    status = commands.add_parser("status", help="report validated campaign state")
    status.add_argument("--campaign-root", type=Path, required=True)
    status.add_argument("--json", action="store_true")
    validate = commands.add_parser("validate", help="require all planned outputs")
    validate.add_argument("--campaign-root", type=Path, required=True)
    finalize = commands.add_parser(
        "finalize", help="validate outputs and freeze the runstore inventory"
    )
    finalize.add_argument("--campaign-root", type=Path, required=True)
    build = commands.add_parser(
        "build", help="promote into a separate checkout and build the publication"
    )
    build.add_argument("--campaign-root", type=Path, required=True)
    build.add_argument("--checkout", type=Path, required=True)
    submit = commands.add_parser(
        "submit", help="plan or submit the production campaign to Slurm"
    )
    submit.add_argument("--campaign-root", type=Path, required=True)
    submit.add_argument("--resources", type=Path, required=True)
    submit.add_argument("--live", action="store_true")
    resume = commands.add_parser("resume", help="plan or submit missing Slurm work")
    resume.add_argument("--campaign-root", type=Path, required=True)
    resume.add_argument("--resources", type=Path, required=True)
    resume.add_argument("--live", action="store_true")
    scheduler = commands.add_parser("slurm-status", help="report jobs and outputs")
    scheduler.add_argument("--campaign-root", type=Path, required=True)
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
    if args.command == "aggregate-exp022":
        aggregate_exp022(args.campaign_root)
        return
    if args.command == "run-experiment":
        run_experiment(args.campaign_root, args.slug)
        return
    if args.command == "finalize":
        print(
            json.dumps(finalize_campaign(args.campaign_root), indent=2, sort_keys=True)
        )
        return
    if args.command == "build":
        print(
            json.dumps(
                build_publication(args.campaign_root, args.checkout),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command in {"submit", "resume"}:
        function = resume_campaign if args.command == "resume" else submit_campaign
        print(
            json.dumps(
                function(args.campaign_root, args.resources, submit=args.live),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "slurm-status":
        print(json.dumps(slurm_status(args.campaign_root), indent=2, sort_keys=True))
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
