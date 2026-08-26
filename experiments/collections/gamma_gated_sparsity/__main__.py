"""Command-line entrypoint for collection planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .execution import (
    aggregate_exp022,
    build_publication,
    campaign_status,
    compose_campaign,
    finalize_campaign,
    initialize_campaign,
    integrate_repair,
    run_experiment,
    run_experiment_shard,
    run_local,
    validate_campaign,
)
from .plan import build_plan
from .slurm import resume_campaign, slurm_status, submit_campaign, submit_canaries


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
    shard = commands.add_parser(
        "run-experiment-shard", help="run one scheduler-managed inference shard"
    )
    shard.add_argument("--campaign-root", type=Path, required=True)
    shard.add_argument("--slug", required=True)
    shard.add_argument("--index", type=int, required=True)
    shard.add_argument("--count", type=int, required=True)
    repair = commands.add_parser(
        "integrate-repair", help="register one repaired downstream result"
    )
    repair.add_argument("--campaign-root", type=Path, required=True)
    repair.add_argument("--repair-root", type=Path, required=True)
    repair.add_argument("--slug", required=True)
    status = commands.add_parser("status", help="report validated campaign state")
    status.add_argument("--campaign-root", type=Path, required=True)
    status.add_argument("--json", action="store_true")
    validate = commands.add_parser("validate", help="require all planned outputs")
    validate.add_argument("--campaign-root", type=Path, required=True)
    finalize = commands.add_parser(
        "finalize", help="validate outputs and freeze the Pingstore inventory"
    )
    finalize.add_argument("--campaign-root", type=Path, required=True)
    build = commands.add_parser(
        "build", help="promote into a separate checkout and build the publication"
    )
    build.add_argument("--campaign-root", type=Path, required=True)
    build.add_argument("--checkout", type=Path, required=True)
    compose = commands.add_parser(
        "compose", help="compose a complete campaign from base and repair outputs"
    )
    compose.add_argument("--campaign-root", type=Path, required=True)
    compose.add_argument("--campaign-id", required=True)
    compose.add_argument("--base-root", type=Path, required=True)
    compose.add_argument("--overlay-root", type=Path, required=True)
    compose.add_argument("--replace", action="append", required=True)
    submit = commands.add_parser(
        "submit", help="plan or submit the production campaign to Slurm"
    )
    submit.add_argument("--campaign-root", type=Path, required=True)
    submit.add_argument("--resources", type=Path, required=True)
    submit_mode = submit.add_mutually_exclusive_group()
    submit_mode.add_argument("--live", action="store_true")
    submit_mode.add_argument("--test-only", action="store_true")
    canaries = commands.add_parser(
        "canaries", help="plan or submit one production cell per resource tier"
    )
    canaries.add_argument("--campaign-root", type=Path, required=True)
    canaries.add_argument("--resources", type=Path, required=True)
    canary_mode = canaries.add_mutually_exclusive_group()
    canary_mode.add_argument("--live", action="store_true")
    canary_mode.add_argument("--test-only", action="store_true")
    resume = commands.add_parser("resume", help="plan or submit missing Slurm work")
    resume.add_argument("--campaign-root", type=Path, required=True)
    resume.add_argument("--resources", type=Path, required=True)
    resume_mode = resume.add_mutually_exclusive_group()
    resume_mode.add_argument("--live", action="store_true")
    resume_mode.add_argument("--test-only", action="store_true")
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
    if args.command == "run-experiment-shard":
        run_experiment_shard(args.campaign_root, args.slug, args.index, args.count)
        return
    if args.command == "integrate-repair":
        print(
            json.dumps(
                integrate_repair(args.campaign_root, args.repair_root, args.slug),
                indent=2,
                sort_keys=True,
            )
        )
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
    if args.command == "compose":
        print(
            json.dumps(
                compose_campaign(
                    args.campaign_root,
                    args.campaign_id,
                    base_root=args.base_root,
                    overlay_root=args.overlay_root,
                    replacements=args.replace,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command in {"submit", "resume"}:
        function = resume_campaign if args.command == "resume" else submit_campaign
        print(
            json.dumps(
                function(
                    args.campaign_root,
                    args.resources,
                    submit=args.live,
                    test_only=args.test_only,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "canaries":
        print(
            json.dumps(
                submit_canaries(
                    args.campaign_root,
                    args.resources,
                    submit=args.live,
                    test_only=args.test_only,
                ),
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
