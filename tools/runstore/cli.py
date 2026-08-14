"""Command-line interface for the pinglab runstore lifecycle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .archive import archive_run, restore_archive, verify_archive
from .campaigns import (
    activate_campaign,
    catalogue,
    current_view,
    resolve_local_campaign,
)
from .contract import (
    ContractError,
    inventory_payload,
    load_json,
    provenance_gaps,
    validate_inventory,
    validate_run_manifest,
    verify_payload,
    write_json_atomic,
)
from .lifecycle import initialize_run
from .promotion import promote_experiment
from .storage import build_store

DEFAULT_STORE = "r2:pinglab/campaigns"
DEFAULT_LOGICAL_URI = "r2://pinglab/campaigns"


def _default_local_roots() -> list[Path]:
    configured = os.environ.get("PINGLAB_RUNSTORE_LOCAL_ROOTS")
    if configured:
        return [Path(value) for value in configured.split(os.pathsep) if value]
    return [Path("runs/campaigns"), Path("runs/restored")]


def _human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def inspect(
    root: Path,
    *,
    as_json: bool = False,
    write_inventory: bool = False,
    finalize: bool = False,
) -> int:
    root = root.resolve()
    if not root.is_dir():
        raise ContractError(f"run root is not a directory: {root}")

    run_path = root / "run.json"
    run = validate_run_manifest(load_json(run_path)) if run_path.exists() else None
    run_id = run["run_id"] if run is not None else f"legacy:{root.name}"
    actual = inventory_payload(root, run_id=run_id)

    inventory_path = root / "inventory.json"
    inventory_state = "absent"
    if inventory_path.exists():
        if write_inventory or finalize:
            raise ContractError(
                "inventory.json already exists; inspect validates but does not replace it"
            )
        existing = validate_inventory(load_json(inventory_path))
        if existing["run_id"] != run_id:
            raise ContractError("run.json and inventory.json use different run IDs")
        verify_payload(root, existing)
        inventory_state = "valid"
    elif finalize:
        if run is None:
            raise ContractError("--finalize requires a valid run.json")
        if run["status"] not in {"planned", "running"}:
            raise ContractError("--finalize requires a planned or running run")
        if run["archive"] is not None:
            raise ContractError(
                "--finalize refuses a run that already records an archive"
            )
        previous = dict(run)
        completed = dict(run)
        completed["status"] = "complete"
        validate_run_manifest(completed)
        write_json_atomic(run_path, completed)
        try:
            write_json_atomic(inventory_path, actual)
        except Exception:
            write_json_atomic(run_path, previous)
            raise
        run = completed
        inventory_state = "finalized"
    elif write_inventory:
        if run is None:
            raise ContractError("--write-inventory requires a valid run.json")
        write_json_atomic(inventory_path, actual)
        inventory_state = "written"

    result = {
        "root": str(root),
        "run_id": run_id,
        "kind": run.get("kind") if run else "unmanaged-legacy",
        "file_count": actual["file_count"],
        "total_size_bytes": actual["total_size_bytes"],
        "payload_digest": actual["payload_digest"],
        "inventory": inventory_state,
        "provenance_gaps": provenance_gaps(run),
    }
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"root       {result['root']}")
        print(f"run        {result['run_id']} ({result['kind']})")
        print(
            f"payload    {result['file_count']} files · "
            f"{_human_size(result['total_size_bytes'])}"
        )
        print(f"digest     {result['payload_digest']}")
        print(f"inventory  {result['inventory']}")
        gaps = result["provenance_gaps"]
        print("provenance complete" if not gaps else f"provenance {', '.join(gaps)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runstore")
    commands = parser.add_subparsers(dest="command", required=True)
    init_parser = commands.add_parser(
        "init", help="create a unique isolated run or campaign root"
    )
    init_parser.add_argument("root", type=Path)
    init_parser.add_argument("--run-id", required=True)
    init_parser.add_argument("--kind", choices=("adhoc", "campaign"), required=True)
    identity = init_parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--experiment")
    identity.add_argument("--collection")
    init_parser.add_argument("--upstream", action="append", default=[])
    init_parser.add_argument("--provenance-notes", default="")
    init_parser.add_argument(
        "--executor", choices=("legacy", "graph"), default="legacy"
    )
    init_parser.add_argument("--graph-digest")
    init_parser.add_argument("--training-digest")
    init_parser.add_argument(
        "--command",
        dest="execution_command",
        nargs=argparse.REMAINDER,
        required=True,
        help="execution command and arguments; must be the final init option",
    )

    inspect_parser = commands.add_parser(
        "inspect", help="inventory a run root without modifying it"
    )
    inspect_parser.add_argument("root", type=Path)
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")
    inventory_action = inspect_parser.add_mutually_exclusive_group()
    inventory_action.add_argument(
        "--write-inventory",
        action="store_true",
        help="atomically write inventory.json; requires run.json and refuses replacement",
    )
    inventory_action.add_argument(
        "--finalize",
        action="store_true",
        help="mark a planned/running run complete and atomically write its inventory",
    )

    promote_parser = commands.add_parser(
        "promote", help="publish one accepted derived experiment directory"
    )
    promote_parser.add_argument("root", type=Path)
    promote_parser.add_argument("experiment")
    promote_parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/data"),
        help="publication data root; defaults to artifacts/data",
    )

    def add_store_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--store",
            default=os.environ.get("PINGLAB_RUNSTORE_STORE", DEFAULT_STORE),
            help="local directory, file:// URI, or rclone root",
        )
        command_parser.add_argument(
            "--logical-base-uri",
            default=os.environ.get("PINGLAB_RUNSTORE_LOGICAL_URI", DEFAULT_LOGICAL_URI),
            help="durable URI recorded for an rclone-backed store",
        )

    archive_parser = commands.add_parser(
        "archive", help="upload and verify an immutable run archive"
    )
    archive_parser.add_argument("root", type=Path)
    archive_parser.add_argument("--archive-id", required=True)
    add_store_arguments(archive_parser)

    verify_parser = commands.add_parser(
        "verify", help="stream and verify every object in an archive"
    )
    verify_parser.add_argument("archive_id")
    verify_parser.add_argument("--json", action="store_true", dest="as_json")
    add_store_arguments(verify_parser)

    restore_parser = commands.add_parser(
        "restore", help="restore and verify an archive in a new destination"
    )
    restore_parser.add_argument("archive_id")
    restore_parser.add_argument("destination", type=Path)
    restore_parser.add_argument("--json", action="store_true", dest="as_json")
    add_store_arguments(restore_parser)

    campaigns_parser = commands.add_parser(
        "campaigns", help="list local and archived collection campaigns"
    )
    campaigns_parser.add_argument("--local-root", type=Path, action="append")
    campaigns_parser.add_argument("--local-only", action="store_true")
    campaigns_parser.add_argument("--json", action="store_true", dest="as_json")
    campaigns_parser.add_argument(
        "--artifacts-root", type=Path, default=Path("artifacts/data")
    )
    add_store_arguments(campaigns_parser)

    activate_parser = commands.add_parser(
        "activate", help="atomically make one local campaign UI-visible"
    )
    activate_parser.add_argument(
        "campaign", help="local path, campaign ID, or archive ID"
    )
    activate_parser.add_argument("--local-root", type=Path, action="append")
    activate_parser.add_argument(
        "--artifacts-root", type=Path, default=Path("artifacts/data")
    )

    current_parser = commands.add_parser(
        "current", help="show and verify the campaign visible in artifacts"
    )
    current_parser.add_argument(
        "--artifacts-root", type=Path, default=Path("artifacts/data")
    )
    current_parser.add_argument("--no-verify-files", action="store_true")
    current_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _print_result(result: dict, *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return
    for key, value in result.items():
        if key == "total_size_bytes":
            value = f"{value} ({_human_size(value)})"
        print(f"{key:<18} {value}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            result = initialize_run(
                args.root,
                run_id=args.run_id,
                kind=args.kind,
                experiment=args.experiment,
                collection=args.collection,
                command=args.execution_command,
                upstream=args.upstream,
                provenance_notes=args.provenance_notes,
                executor=args.executor,
                graph_digest=args.graph_digest,
                training_digest=args.training_digest,
            )
            _print_result(
                {
                    "run_id": result["run_id"],
                    "kind": result["kind"],
                    "root": str(args.root.resolve()),
                }
            )
            raise SystemExit(0)
        if args.command == "inspect":
            raise SystemExit(
                inspect(
                    args.root,
                    as_json=args.as_json,
                    write_inventory=args.write_inventory,
                    finalize=args.finalize,
                )
            )
        if args.command == "promote":
            _print_result(
                promote_experiment(
                    args.root,
                    args.experiment,
                    artifacts_root=args.artifacts_root,
                )
            )
            raise SystemExit(0)
        if args.command == "archive":
            store = build_store(args.store, logical_base_uri=args.logical_base_uri)
            result = archive_run(args.root, args.archive_id, store)
            _print_result(
                {
                    "archive_id": result["archive"]["archive_id"],
                    "uri": result["archive"]["uri"],
                    "run_id": result["run_id"],
                }
            )
            raise SystemExit(0)
        if args.command == "verify":
            store = build_store(args.store, logical_base_uri=args.logical_base_uri)
            _print_result(verify_archive(store, args.archive_id), as_json=args.as_json)
            raise SystemExit(0)
        if args.command == "restore":
            store = build_store(args.store, logical_base_uri=args.logical_base_uri)
            _print_result(
                restore_archive(store, args.archive_id, args.destination),
                as_json=args.as_json,
            )
            raise SystemExit(0)
        if args.command == "campaigns":
            roots = args.local_root or _default_local_roots()
            store = (
                None
                if args.local_only
                else build_store(args.store, logical_base_uri=args.logical_base_uri)
            )
            active_id = None
            try:
                active_id = current_view(args.artifacts_root, verify_files=False)[
                    "campaign_id"
                ]
            except ContractError:
                pass
            rows = catalogue(roots, store, active_campaign_id=active_id)
            if args.as_json:
                print(json.dumps(rows, indent=2))
            elif not rows:
                print("No campaigns found.")
            else:
                print(
                    f"{'CAMPAIGN':<32} {'LOCATION':<10} {'STATUS':<10} "
                    f"{'PROFILE':<11} {'UI':<7} COMMIT"
                )
                for row in rows:
                    commit = row["git_commit"][:8] if row["git_commit"] else "—"
                    print(
                        f"{row['campaign_id']:<32} "
                        f"{'+'.join(row['locations']):<10} {row['status']:<10} "
                        f"{(row['profile'] or '—'):<11} "
                        f"{('current' if row['active'] else '—'):<7} {commit}"
                    )
            raise SystemExit(0)
        if args.command == "activate":
            roots = args.local_root or _default_local_roots()
            root = resolve_local_campaign(args.campaign, roots)
            _print_result(activate_campaign(root, artifacts_root=args.artifacts_root))
            raise SystemExit(0)
        if args.command == "current":
            result = current_view(
                args.artifacts_root, verify_files=not args.no_verify_files
            )
            _print_result(result, as_json=args.as_json)
            if not result["valid"]:
                raise SystemExit(1)
            raise SystemExit(0)
    except ContractError as exc:
        raise SystemExit(f"runstore: {exc}") from exc
