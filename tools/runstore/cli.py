"""Command-line interface for the pinglab runstore lifecycle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .archive import archive_run, restore_archive, verify_archive
from .contract import (
    ContractError,
    inventory_payload,
    load_json,
    provenance_gaps,
    validate_inventory,
    validate_run_manifest,
    verify_payload,
)
from .storage import build_store

DEFAULT_STORE = "r2:pinglab/campaigns"
DEFAULT_LOGICAL_URI = "r2://pinglab/campaigns"


def _human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def inspect(root: Path, *, as_json: bool = False) -> int:
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
        existing = validate_inventory(load_json(inventory_path))
        if existing["run_id"] != run_id:
            raise ContractError("run.json and inventory.json use different run IDs")
        verify_payload(root, existing)
        inventory_state = "valid"

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
    inspect_parser = commands.add_parser(
        "inspect", help="inventory a run root without modifying it"
    )
    inspect_parser.add_argument("root", type=Path)
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")

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
        if args.command == "inspect":
            raise SystemExit(inspect(args.root, as_json=args.as_json))
        store = build_store(args.store, logical_base_uri=args.logical_base_uri)
        if args.command == "archive":
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
            _print_result(verify_archive(store, args.archive_id), as_json=args.as_json)
            raise SystemExit(0)
        if args.command == "restore":
            _print_result(
                restore_archive(store, args.archive_id, args.destination),
                as_json=args.as_json,
            )
            raise SystemExit(0)
    except ContractError as exc:
        raise SystemExit(f"runstore: {exc}") from exc
