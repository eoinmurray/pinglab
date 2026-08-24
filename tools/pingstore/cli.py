"""Single operator interface for Pinglab scientific data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .archive import archive_dataset, restore_dataset
from .catalogue import Catalogue
from .contracts import PingstoreError, load_json, write_json_atomic
from .inventory import add_remote_catalogue, inventory_local, verify_local_inventory
from .legacy import legacy_catalogue, restore_legacy_archive, verify_legacy_archive
from .materialize import cutover, materialize_shadow
from .migration import build_plan, classify, import_shadow
from .native import capture_campaign_metadata, capture_local_run
from .prune import pruning_plan
from .registry import coverage, registry_path

DEFAULT_ROOT = Path("runs/pingstore")
DEFAULT_STORE = "r2:pinglab/campaigns"
DEFAULT_URI = "r2://pinglab/campaigns"


def _print(value: Any, *, as_json: bool = False) -> None:
    if as_json or not isinstance(value, str):
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        print(value)


def _migration_root(args: argparse.Namespace) -> Path:
    return args.root / "migrations" / args.migration_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pingstore")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status")
    status.add_argument("--json", action="store_true")

    runs = sub.add_parser("runs")
    runs.add_argument("experiment", nargs="?")
    runs.add_argument("--json", action="store_true")

    select = sub.add_parser("select")
    select.add_argument("experiment")
    select.add_argument("run_id")
    select.add_argument("--preview", action="store_true")

    freeze = sub.add_parser("freeze")
    freeze.add_argument("collection")
    freeze.add_argument("--snapshot", required=True)

    archive = sub.add_parser("archive")
    archive.add_argument("dataset_id")
    archive.add_argument("destination", type=Path)

    preview = sub.add_parser("preview")
    preview.add_argument("collection")
    preview.add_argument("--shadow", type=Path, required=True)
    preview.add_argument("--official-only", action="store_true")
    preview.add_argument("--proposal", action="store_true")

    prune = sub.add_parser("prune")
    prune.add_argument("collection")
    prune.add_argument("--plan", action="store_true", required=True)

    verify = sub.add_parser("verify")
    source = verify.add_mutually_exclusive_group(required=True)
    source.add_argument("--local", action="store_true")
    source.add_argument("--r2", action="store_true")
    source.add_argument("--all", action="store_true")
    verify.add_argument("--migration-id", default="legacy-to-pingstore-v1")
    verify.add_argument("--deep", action="store_true")
    verify.add_argument("--archive-id", action="append", default=[])
    verify.add_argument("--store", default=os.environ.get("PINGLAB_RUNSTORE_STORE", DEFAULT_STORE))
    verify.add_argument("--logical-base-uri", default=DEFAULT_URI)

    restore = sub.add_parser("restore")
    restore.add_argument("archive_id")
    restore.add_argument("destination", type=Path)
    restore.add_argument("--native", action="store_true")
    restore.add_argument("--store", default=os.environ.get("PINGLAB_RUNSTORE_STORE", DEFAULT_STORE))
    restore.add_argument("--logical-base-uri", default=DEFAULT_URI)

    capture_local = sub.add_parser("capture-local", help=argparse.SUPPRESS)
    capture_local.add_argument("--repo", type=Path, required=True)
    capture_local.add_argument("--experiment", required=True)
    capture_local.add_argument("--staging", type=Path, required=True)

    capture_campaign = sub.add_parser("capture-campaign", help=argparse.SUPPRESS)
    capture_campaign.add_argument("--campaign-root", type=Path, required=True)

    migrate = sub.add_parser("migrate")
    migrate_sub = migrate.add_subparsers(dest="migration_command", required=True)
    for name in ("inventory", "classify", "plan", "import", "cutover"):
        command = migrate_sub.add_parser(name)
        command.add_argument("--migration-id", default="legacy-to-pingstore-v1")
        if name == "inventory":
            command.add_argument("--repo", type=Path, default=Path.cwd())
            command.add_argument("--r2", action="store_true")
            command.add_argument(
                "--store",
                default=os.environ.get("PINGLAB_RUNSTORE_STORE", DEFAULT_STORE),
            )
            command.add_argument("--logical-base-uri", default=DEFAULT_URI)
        if name == "import":
            destination = command.add_mutually_exclusive_group(required=True)
            destination.add_argument(
                "--shadow",
                action="store_true",
                help="import into an isolated --root for rehearsal",
            )
            destination.add_argument(
                "--local",
                action="store_true",
                help="install working metadata under the selected --root",
            )
        if name == "cutover":
            command.add_argument("--confirm", action="store_true")
    return parser


def _status(catalogue: Catalogue) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    for path in sorted((catalogue.root / "collections").glob("*/collection-dataset.json")):
        dataset = load_json(path)
        datasets.append(
            {
                "collection": dataset["collection"],
                "dataset_id": dataset["dataset_id"],
                "status": dataset["status"],
                "experiments": len(dataset["experiments"]),
                "runs": sum(len(rows) for rows in dataset["runs"].values()),
                "official": len(dataset["official_runs"]),
                "preview_overrides": len(dataset["preview_overrides"]),
            }
        )
    result: dict[str, Any] = {
        "root": str(catalogue.root.resolve()),
        "datasets": datasets,
    }
    repo = Path.cwd().resolve()
    if registry_path(repo).is_file():
        result["coverage"] = coverage(repo)
    return result


def _runs(catalogue: Catalogue, experiment: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = catalogue.root / "experiment-runs"
    pattern = f"*/{experiment}/*/run.json" if experiment else "*/*/*/run.json"
    for path in sorted(root.glob(pattern)):
        run = load_json(path)
        rows.append(
            {
                "run_id": run["run_id"],
                "collection": run["collection"],
                "experiment": run["experiment"],
                "status": run["status"],
                "disposition": run["disposition"],
                "location": run["payload"]["location"],
            }
        )
    return rows


def _load_migration(root: Path) -> tuple[dict, dict, dict]:
    return (
        load_json(root / "inventory.json"),
        load_json(root / "classifications.json"),
        load_json(root / "import-plan.json"),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalogue = Catalogue(args.root)
    try:
        if args.command == "status":
            _print(_status(catalogue), as_json=args.json)
        elif args.command == "runs":
            _print(_runs(catalogue, args.experiment), as_json=args.json)
        elif args.command == "select":
            catalogue.select(args.experiment, args.run_id, preview=args.preview)
            _print({"selected": args.run_id, "preview": args.preview})
        elif args.command == "freeze":
            _print(catalogue.freeze(args.collection, args.snapshot))
        elif args.command == "archive":
            _print(archive_dataset(catalogue, args.dataset_id, args.destination))
        elif args.command == "preview":
            _print(
                materialize_shadow(
                    catalogue,
                    args.collection,
                    args.shadow,
                    use_preview=not args.official_only,
                    use_proposal=args.proposal,
                )
            )
        elif args.command == "prune":
            _print(pruning_plan(catalogue, args.collection))
        elif args.command == "restore":
            if args.native:
                _print(restore_dataset(Path(args.archive_id), args.destination))
            else:
                _print(
                    restore_legacy_archive(
                        args.archive_id,
                        args.destination,
                        store_spec=args.store,
                        logical_uri=args.logical_base_uri,
                    )
                )
        elif args.command == "capture-local":
            _print(
                capture_local_run(
                    args.repo, args.experiment, args.staging, root=args.root
                )
            )
        elif args.command == "capture-campaign":
            campaign_root = args.campaign_root.resolve()
            _print(
                capture_campaign_metadata(
                    campaign_root, load_json(campaign_root / "collection-plan.json")
                )
            )
        elif args.command == "verify":
            result: dict[str, Any] = {}
            if args.local or args.all:
                migration_root = args.root / "migrations" / args.migration_id
                result["local"] = verify_local_inventory(
                    load_json(migration_root / "inventory.json"), deep=args.deep
                )
            if args.r2 or args.all:
                archive_ids = list(args.archive_id)
                if not archive_ids:
                    rows = legacy_catalogue(
                        local_roots=[],
                        store_spec=args.store,
                        logical_uri=args.logical_base_uri,
                    )
                    archive_ids = sorted(
                        {
                            row["store_key"]
                            for row in rows
                            if row.get("store_key") and "/" not in row["store_key"]
                        }
                    )
                result["r2"] = [
                    verify_legacy_archive(
                        archive_id,
                        store_spec=args.store,
                        logical_uri=args.logical_base_uri,
                    )
                    for archive_id in archive_ids
                ]
            _print(result)
        elif args.command == "migrate":
            root = _migration_root(args)
            root.mkdir(parents=True, exist_ok=True)
            if args.migration_command == "inventory":
                value = inventory_local(args.repo)
                if args.r2:
                    value = add_remote_catalogue(
                        value,
                        legacy_catalogue(
                            local_roots=[],
                            store_spec=args.store,
                            logical_uri=args.logical_base_uri,
                        ),
                    )
                write_json_atomic(root / "inventory.json", value)
            elif args.migration_command == "classify":
                value = classify(load_json(root / "inventory.json"))
                write_json_atomic(root / "classifications.json", value)
            elif args.migration_command == "plan":
                value = build_plan(
                    load_json(root / "inventory.json"),
                    load_json(root / "classifications.json"),
                )
                write_json_atomic(root / "import-plan.json", value)
            elif args.migration_command == "import":
                inventory, _classifications, plan = _load_migration(root)
                value = import_shadow(
                    inventory, plan, catalogue=catalogue, migration_root=root
                )
            else:
                if not args.confirm:
                    raise PingstoreError("cutover requires --confirm and explicit approval")
                value = cutover()
            _print(value)
        return 0
    except (PingstoreError, OSError, ValueError) as exc:
        raise SystemExit(f"pingstore: {exc}") from exc
