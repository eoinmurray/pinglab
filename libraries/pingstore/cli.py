"""A small CLI over the same Store API used by Python."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .core import Store, discover_runs, load_json
from .remote import DEFAULT_DESTINATION, Remote


def parser():
    result = argparse.ArgumentParser(prog="pingstore")
    result.add_argument("--root", type=Path, default=Path(".pingstore"))
    commands = result.add_subparsers(dest="command", required=True)
    insert = commands.add_parser("insert", help="insert a completed directory")
    insert.add_argument("directory", type=Path)
    insert.add_argument("--record", type=Path, help="run.json metadata to use")
    get = commands.add_parser("get", help="read one run's metadata")
    get.add_argument("id")
    listing = commands.add_parser(
        "list", help="list run metadata without reading payloads"
    )
    listing.add_argument("--experiment")
    listing.add_argument("--stage")
    for name in ("backup", "restore", "delete"):
        command = commands.add_parser(name)
        command.add_argument("ids", nargs="*" if name == "backup" else "+")
        command.add_argument("--destination", default=DEFAULT_DESTINATION)
        if name == "delete":
            command.add_argument("--storage", choices=("local", "r2"), default="local")
            command.add_argument(
                "--yes",
                action="store_true",
                help="confirm explicit targets, including dependant warnings",
            )
            command.add_argument("--dry-run", action="store_true")
    discover = commands.add_parser("discover", help="Demolab discovery")
    discover.add_argument("--source", type=Path)
    migrate = commands.add_parser(
        "migrate", help="copy old records to a separate simple store"
    )
    migrate.add_argument("source", type=Path, help="existing .pingstore directory")
    migrate.add_argument(
        "--apply", action="store_true", help="otherwise show the migration plan"
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.perf_counter()
    store = Store(args.root)
    try:
        if args.command == "insert":
            result = store.insert(
                args.directory, record=load_json(args.record) if args.record else None
            ).record
        elif args.command == "get":
            result = store.get(args.id).record
        elif args.command == "list":
            result = store.list(experiment=args.experiment, stage=args.stage)
        elif args.command == "backup":
            result = store.backup(args.destination, args.ids or None)
        elif args.command == "restore":
            result = store.restore(args.destination, args.ids)
        elif args.command == "discover":
            result = discover_runs(args.source or store.runs)
        elif args.command == "migrate":
            from .migration import migrate

            result = migrate(args.source, store.root, apply=args.apply)
        else:
            remote = Remote(args.destination) if args.storage == "r2" else None
            deps = store.dependants(args.ids)
            if remote:
                index = remote.index()
                deps.update(
                    {
                        i: r["record"]["depends_on"]
                        for i, r in index["runs"].items()
                        if i not in args.ids
                        and set(args.ids).intersection(r["record"]["depends_on"])
                    }
                )
            plan = {"delete": args.ids, "storage": args.storage, "dependants": deps}
            print(json.dumps(plan, indent=2), file=sys.stderr)
            if args.dry_run:
                result = plan
            else:
                if (
                    not args.yes
                    and input(
                        "Delete these runs, including any dependency breakage? [y/N] "
                    ).lower()
                    != "y"
                ):
                    return 1
                result = (
                    remote.delete(store, args.ids, allow_dependants=True)
                    if remote
                    else store.delete(args.ids, allow_dependants=True)
                )
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, EOFError, subprocess.CalledProcessError) as error:
        print(f"pingstore: {error}", file=sys.stderr)
        return 1
    finally:
        print(f"Elapsed: {time.perf_counter() - started:.3f}s", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
