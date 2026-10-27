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
    status = commands.add_parser("status", help="explain current runsets and storage")
    status.add_argument("--root", type=Path, default=Path("."))
    status.add_argument("--runs-root", type=Path)
    status.add_argument("--source", choices=("auto", "local", "remote"), default="auto")
    status.add_argument("--asset-base", default="https://assets.eoinmurray.info")
    status.add_argument("--json", action="store_true")
    status.add_argument(
        "--inventory",
        action="store_true",
        help="refresh authenticated R2 object inventory",
    )
    status.add_argument(
        "--destination",
        default=os.environ.get("PINGLAB_R2_REMOTE", "r2")
        + ":"
        + os.environ.get("PINGLAB_R2_BUCKET", "pinglab")
        + "/pingstore/runs",
    )
    status.add_argument(
        "--verify",
        action="store_true",
        help="validate local payload bytes before indexing",
    )
    status.add_argument(
        "--index-path", type=Path, help="save a rebuildable versioned JSON index"
    )
    status.add_argument("--catalog-stdin", action="store_true", help=argparse.SUPPRESS)
    status.set_defaults(handler=_status)
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
    scope = prune.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--all",
        action="store_true",
        help="consider all experiments",
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
    prune.add_argument("--storage", choices=("local", "r2", "both"), default="both")
    prune.add_argument("--destination", default=f"{remote}:{bucket}/pingstore/runs")
    prune.add_argument("--asset-base", default="https://assets.eoinmurray.info")
    prune.add_argument("--json", action="store_true")
    prune.add_argument(
        "--malformed-run",
        action="append",
        dest="malformed_runs",
        help="explicit bounded malformed holding to remove (repeatable)",
    )
    action = prune.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--dry-run", action="store_true", help="print the exact immutable prune plan"
    )
    action.add_argument(
        "--confirm", metavar="PLAN_HASH", help="apply an unchanged dry-run plan"
    )
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
    discover.set_defaults(handler=_discover)
    backup.set_defaults(handler=_backup, argument_error=backup.error)
    prune.set_defaults(handler=_prune)
    evict.set_defaults(handler=_evict)
    return parser


def _discover(args) -> int:
    source = args.source or Path(
        os.environ.get("DEMOLAB_PREVIEW_SOURCE") or ".pingstore/runs"
    )
    records = core.discover_runs(source)
    # A single JSON document, only after all candidates passed. No partial output.
    print(json.dumps(records, ensure_ascii=False))
    return 0


def _status(args) -> int:
    catalog, remote_known = None, args.source != "local"
    publications = None
    operation = None
    if args.source != "local":
        if args.catalog_stdin:
            supplied = json.load(sys.stdin)
            if isinstance(supplied, dict) and "catalog" in supplied:
                catalog = supplied["catalog"]
                publications = (supplied.get("control") or {}).get("publications")
                operation = (supplied.get("control") or {}).get("operation")
            else:
                catalog = supplied
            remote_known = catalog is not None
        else:
            try:
                request = core.Request(
                    args.asset_base.rstrip("/") + "/pingstore/catalog.json",
                    headers={"User-Agent": "Pinglab-Pingstore/1.0"},
                )
                with core.urlopen(request, timeout=30) as response:
                    catalog = json.load(response)
                try:
                    control = core.public_json(
                        args.asset_base.rstrip("/") + "/pingstore/control.json"
                    )
                    publications = control["publications"]
                    operation = control.get("operation")
                except OSError:
                    publications = core.empty_publications()
            except (OSError, ValueError) as exc:
                if args.source == "remote":
                    raise core.PingstoreError(
                        f"R2 inventory unavailable: {exc}"
                    ) from exc
                remote_known = False
    inventory_state = None
    if args.inventory:
        if args.source == "local":
            raise core.PingstoreError("--inventory requires auto or remote source mode")
        snapshot = core._maintenance_remote_snapshot(
            args.destination, args.root.resolve()
        )
        catalog = {**snapshot["catalog"], "runs": snapshot["valid"]}
        inventory_state = core.inventory_projection(
            snapshot["objects"], snapshot["valid"]
        )
        publications = snapshot["control"]["publications"]
        operation = snapshot["control"]["operation"]
        remote_known = True
    index = core.reporting_index(
        args.root,
        runs=args.runs_root,
        catalog=catalog,
        source=args.source,
        remote_known=remote_known,
        verify=args.verify,
        publications=publications,
        operation=operation,
        inventory_state=inventory_state,
    )
    if args.index_path:
        core.write_json_atomic(args.index_path, index)
    print(json.dumps(index, sort_keys=True) if args.json else core.render_status(index))
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
    options = dict(
        storage=args.storage,
        destination=args.destination,
        malformed_runs=args.malformed_runs,
    )
    if args.storage == "local":
        options["remote_snapshot"] = core.public_maintenance_snapshot(args.asset_base)
    if args.dry_run:
        plan = core.unified_prune_plan(
            args.root, args.experiments, args.stages, **options
        )
        print(
            json.dumps(plan, sort_keys=True)
            if args.json
            else core.render_unified_prune_plan(plan)
        )
    else:
        result = core.apply_unified_prune(
            args.root, args.confirm, args.experiments, args.stages, **options
        )
        plan = result["plan"]
        print(
            f"Cleanup {result['phase']}: "
            + "; ".join(
                f"{target}: {len(objects)} objects, {sum(row['bytes'] for row in objects.values())} bytes"
                for target, objects in plan["objects"].items()
            )
        )
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"pingstore {args.command}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
