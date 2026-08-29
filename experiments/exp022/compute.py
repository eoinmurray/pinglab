"""Exp022 compute: train/campaign operations and retained diagnostic simulations."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "tools")]

from experiments.exp022.recipe import *  # noqa: F403
from experiments.exp022.recipe import _display_path
from experiments.exp022 import campaign
from helpers import runpod
from helpers.checkpoints import resolve_checkpoint
from helpers.cli import parse_meta
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import reserve_stage, source_run, stage_reservation, stage_run

def cell_dir(name: str) -> Path:
    """Shared per-cell artifact directory."""
    return TRAINING_ROOT / name


def load_cell(name: str) -> Path:
    """Return a trained cell's directory, or fail loudly if this notebook has
    not been run. Analysis notebooks call this instead of training."""
    d = cell_dir(name)
    if not (d / "weights.pth").exists() or not (d / "weights_final.pth").exists():
        raise SystemExit(
            f"missing trained cell '{name}' at {_display_path(d)}; "
            "run exp022 (Training) first to produce the shared cells."
        )
    return d


def _tr06_diagnostic_root() -> Path:
    return Path(os.environ["PINGLAB_ARTIFACTS_ROOT"])


def tr06_diagnostic_done(job_id: str) -> bool:
    """Modal completion hook for one bounded TR-06 readout variant."""
    from experiments.exp022 import tr06_diagnostic

    return (
        job_id in tr06_diagnostic.VARIANTS
        and (_tr06_diagnostic_root() / job_id / "diagnostic_summary.json").exists()
    )


def run_tr06_diagnostic(job_id: str) -> None:
    """Modal execution hook; diagnostic scale is explicit in the job environment."""
    from experiments.exp022 import tr06_diagnostic

    def optional_number(name: str, converter):
        value = os.environ.get(name)
        return None if value is None else converter(value)

    tr06_diagnostic.run_variant(
        job_id,
        root=_tr06_diagnostic_root(),
        max_samples=int(os.environ["EXP022_TR06_DIAGNOSTIC_MAX_SAMPLES"]),
        epochs=int(os.environ["EXP022_TR06_DIAGNOSTIC_EPOCHS"]),
        seed=int(os.environ["EXP022_TR06_DIAGNOSTIC_SEED"]),
        device="auto",
        n_hidden=optional_number("EXP022_TR06_DIAGNOSTIC_N_HIDDEN", int),
        t_ms=optional_number("EXP022_TR06_DIAGNOSTIC_T_MS", float),
        dt_ms=optional_number("EXP022_TR06_DIAGNOSTIC_DT_MS", float),
    )


def runpod_is_done(cell: dict, plumbing: bool) -> bool:
    """A cell is done iff its metrics.json exists AND was trained at the scale
    THIS run expects (max_samples, epochs, dt all matching).

    Existence alone is not enough: a training root may contain cells produced
    under a different run contract, and a bare exists() check would skip those
    and silently ship a mixed-scale, invalid dataset. Comparing the baked config
    makes the marker honest: a stale cell reads as pending and gets retrained.
    """
    p = cell_dir(cell["name"]) / "metrics.json"
    if not p.exists():
        return False
    try:
        cfg = json.loads(p.read_text()).get("config", {})
    except (json.JSONDecodeError, OSError):
        return False
    if plumbing:
        os.environ["PINGLAB_NB022_PLUMBING"] = "1"
    want_ms, want_ep = cell_samples_epochs(cell)
    return (cfg.get("max_samples") == want_ms
            and cfg.get("epochs") == want_ep
            and cfg.get("dt") == cell["dt_ms"])


def _train_one_cell(cell: dict, plumbing: bool) -> None:
    """Train ONE cell by invoking the SNN CLI — flags identical to a local run.

    Writes to cell_dir(name), which sits under TRAINING_ROOT — on a pod that is
    the shared network volume (/shared/training via PINGLAB_TRAINING_ROOT), so
    the artifact is durable the moment it lands. Used by --pod-run and --train-cell.
    """
    _writable_compute_root(TRAINING_ROOT)
    ms, ep = cell_samples_epochs(cell)  # honours PINGLAB_NB022_PLUMBING
    spec = cell
    if plumbing:
        # build_train_args re-applies a canonical cell's own max_samples (60000),
        # which would defeat the tiny plumbing scale. Strip it so the plumbing
        # ms=100 takes — and so runpod_is_done agrees with what was trained.
        spec = {k: v for k, v in cell.items() if k != "max_samples"}
    args = build_train_args(spec, cell_dir(cell["name"]), ms, ep)
    print(
        f"[train-cell] {cell['training_run_id']} / {cell['name']} "
        f"(n={ms}, {ep} ep) → {cell_dir(cell['name'])}"
    )
    subprocess.run([sys.executable, str(SNN_TOOL), *args], cwd=REPO, check=True)
    _stamp_training_run_identity(cell)


def _stamp_training_run_identity(cell: dict) -> None:
    """Attach the public training-run ID to one completed cell's artifacts."""
    directory = cell_dir(cell["name"])
    _writable_compute_root(directory)
    for filename in ("config.json", "metrics.json"):
        path = directory / filename
        if not path.exists():
            raise RuntimeError(f"completed cell is missing {path}")
        payload = json.loads(path.read_text())
        payload["training_run_id"] = cell["training_run_id"]
        payload["training_cell_name"] = cell["name"]
        nested_config = payload.get("config")
        if isinstance(nested_config, dict):
            nested_config["training_run_id"] = cell["training_run_id"]
            nested_config["training_cell_name"] = cell["name"]
        path.write_text(json.dumps(payload, indent=2) + "\n")


def _cell_by_name(name: str) -> dict | None:
    return next((c for c in CANONICAL_CELLS if c["name"] == name), None)


def pod_run() -> None:
    """Pod-side entrypoint (the image runs this compute module with `--pod-run`).

    Trains every cell named in the CELLS env var to the shared volume, skipping
    any already done there (scale-aware marker → free resume across pods), then
    self-terminates. The loop, skip-done and always-self-terminate contract lives
    in runpod.pod_run_loop; here we only say what a cell's done-check and training
    run are.
    """
    plumbing = os.environ.get("PINGLAB_NB022_PLUMBING") == "1"
    print(f"[pod-run] plumbing={plumbing} root={TRAINING_ROOT}")

    def is_done(name: str) -> bool:
        cell = _cell_by_name(name)
        return cell is not None and runpod_is_done(cell, plumbing)

    def run_job(name: str) -> None:
        cell = _cell_by_name(name)
        assert cell is not None  # pod_run_loop only passes registered job ids
        _train_one_cell(cell, plumbing)

    runpod.pod_run_loop(
        job_ids=[c["name"] for c in CANONICAL_CELLS],
        is_done=is_done, run_job=run_job,
    )


def runpod_buckets(cells: list[dict], cells_per_pod: int) -> list[dict]:
    """Assign cells to pods: each canonical cell → its own pod (heavy, isolated);
    every other family packed cells_per_pod at a time. Returns [{name, cells}]."""
    canonical = [c["name"] for c in cells if c["family"] == "canonical"]
    sweep = [c["name"] for c in cells if c["family"] != "canonical"]
    buckets = [{"name": f"canon-{n}", "cells": [n]} for n in canonical]
    for i in range(0, len(sweep), cells_per_pod):
        buckets.append({"name": f"sweep-{i // cells_per_pod:02d}",
                        "cells": sweep[i:i + cells_per_pod]})
    return buckets


def run_via_runpod(argv: list[str]) -> None:
    """`--runpod` dispatch: fire a laptop-independent RunPod fan-out via the shared
    runpod.dispatch path.

    Pods self-run their assigned cells to the shared network volume and
    self-terminate; the laptop only fires them. Retrieve results afterwards with
    `--runpod --collect`, then capture a compute run and explicitly analyse/present it.
    Dry-run by DEFAULT; --live to create pods. Exp022's only bespoke bit is
    runpod_buckets (one pod per canonical cell); everything else is the common
    fan-out in helpers/runpod.py.
    """
    meta, reserved = _dispatch_meta(argv)

    cells = CANONICAL_CELLS
    if meta.only_cells:
        wanted = set(meta.only_cells)
        cells = [c for c in cells if c["name"] in wanted]
        missing = wanted - {c["name"] for c in cells}
        if missing:
            raise SystemExit(f"unknown cell(s): {sorted(missing)}")

    if meta.live and not meta.collect and reserved is None:
        reserved = reserve_stage(REPO / ".pingstore", SLUG, "compute", origin="runpod")
    if meta.collect and reserved is None:
        raise SystemExit("collection requires --run-id from the original dispatch")
    local_root = TRAINING_ROOT
    subdir = runpod.TRAINING_SUBDIR
    extra_env = None
    if reserved:
        temporary = REPO / ".pingstore/runs" / f".{reserved}.tmp"
        reservation = stage_reservation(temporary)
        if (reservation["run_id"] != reserved or reservation["experiment"] != SLUG
                or reservation["stage"] != "compute" or reservation["origin"] != "runpod"):
            raise PingstoreError("RunPod requires its own reserved exp022 compute identity")
        local_root = temporary / "export/cells"
        subdir = f"{reserved}/cells"
        extra_env = {"PINGLAB_TRAINING_ROOT": f"{runpod.VOLUME_MOUNT}/{subdir}",
                     "PINGSTORE_RUN_ID": reserved}
        print(f"reserved compute run: {reserved}")
    runpod.dispatch(
        slug=SLUG, runner=SLUG,
        buckets=runpod_buckets(cells, meta.cells_per_pod),
        gpu=meta.gpu, live=meta.live, plumbing=meta.plumbing, collect=meta.collect,
        collect_subdir=subdir,
        local_collect_dir=str(local_root),
        extra_env=extra_env,
        plumbing_env={"PINGLAB_NB022_PLUMBING": "1"},
    )
    if meta.collect:
        with stage_run(REPO, SLUG, "compute", run_id=reserved, configuration=SCALE,
                       export_root="export/cells", operation="collect-runpod") as run:
            for cell in CANONICAL_CELLS:
                for role in ("best_validation", "final_epoch"):
                    resolve_checkpoint(local_root / cell["name"], role)
            generate_snapshots(local_root, run.export / "snapshots")


def _dispatch_meta(argv: list[str]):
    arguments = list(argv)
    reserved = None
    if "--run-id" in arguments:
        index = arguments.index("--run-id")
        if index + 1 == len(arguments) or arguments[index + 1].startswith("--"):
            raise SystemExit("--run-id requires a reserved identity")
        reserved = arguments[index + 1]
        del arguments[index:index + 2]
    return parse_meta(arguments, allow_dispatch=True), reserved


def _campaign_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--campaign-manifest", type=Path, metavar="ROOT")
    group.add_argument("--campaign-status", type=Path, metavar="MANIFEST")
    group.add_argument("--campaign-list", type=Path, metavar="MANIFEST")
    group.add_argument("--campaign-train-cell", metavar="NAME")
    group.add_argument("--campaign-validate", type=Path, metavar="MANIFEST")
    group.add_argument("--campaign-aggregate", type=Path, metavar="MANIFEST")
    group.add_argument(
        "--campaign-import-compatible", type=Path, metavar="MANIFEST"
    )
    parser.add_argument("--campaign", type=Path, metavar="MANIFEST")
    parser.add_argument("--from-campaign", type=Path, metavar="MANIFEST")
    parser.add_argument("--campaign-id")
    parser.add_argument("--execution-origin", default="campaign",
                        choices=("campaign", "local", "slurm-wilkes"),
                        help="planned producer; campaign permits mixed local/HPC workers")
    parser.add_argument("--tier", default="all")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--retry-only", action="store_true")
    parser.add_argument("--recover-stale", action="store_true")
    parser.add_argument("--plumbing", action="store_true")
    return parser


def _portable_cell_contract(row: dict) -> dict:
    """Return the scientific cell contract with destination paths removed."""
    parameters = copy.deepcopy(row["parameters"])
    parameters.get("arguments", {}).pop("--out-dir", None)
    return {
        "name": row["name"],
        "training_run_id": row["training_run_id"],
        "family": row["family"],
        "resource_tier": row["resource_tier"],
        "parameters": parameters,
    }


def _import_compatible_cells(destination: dict, source_path: Path) -> dict:
    """Copy only source cells with an identical resolved scientific contract."""
    source_path = source_path.resolve()
    source = campaign.load_manifest(source_path)
    source_root = Path(source["campaign_root"])
    if source_path != source_root / "campaign.json":
        raise SystemExit("source manifest must be <campaign-root>/campaign.json")
    source_rows = {row["name"]: row for row in source["cells"]}
    imported: list[str] = []
    incompatible: list[str] = []
    for row in destination["cells"]:
        source_row = source_rows.get(row["name"])
        if (
            source_row is None
            or _portable_cell_contract(source_row) != _portable_cell_contract(row)
        ):
            incompatible.append(row["name"])
            continue
        validation = campaign.validate_cell(source_row)
        if not validation["valid"]:
            raise SystemExit(
                f"compatible source cell is invalid: {row['name']}: "
                + "; ".join(validation["reasons"])
            )
        destination_dir = Path(row["output_directory"])
        _writable_compute_root(destination_dir)
        if destination_dir.exists():
            raise SystemExit(f"import destination already exists: {destination_dir}")
        shutil.copytree(Path(source_row["output_directory"]), destination_dir)
        origin = {
            "campaign_id": source["campaign_id"],
            "campaign_manifest_sha256": source["manifest_sha256"],
            "repository_commit": source["repository"]["commit"],
            "source_directory": source_row["output_directory"],
        }
        for filename in ("config.json", "metrics.json"):
            path = destination_dir / filename
            payload = json.loads(path.read_text())
            payload["imported_cell_provenance"] = origin
            payload["training_run_id"] = row["training_run_id"]
            payload["training_cell_name"] = row["name"]
            nested = payload.get("config")
            if isinstance(nested, dict):
                nested["training_run_id"] = row["training_run_id"]
                nested["training_cell_name"] = row["name"]
            path.write_text(json.dumps(payload, indent=2) + "\n")
        _stamp_campaign_identity(destination_dir, destination, row)
        imported_validation = campaign.validate_cell(row)
        if not imported_validation["valid"]:
            raise RuntimeError(
                f"imported cell failed destination validation: {row['name']}: "
                + "; ".join(imported_validation["reasons"])
            )
        imported.append(row["name"])
    return {
        "source_campaign_id": source["campaign_id"],
        "destination_campaign_id": destination["campaign_id"],
        "imported": imported,
        "pending_incompatible": incompatible,
    }


def _checked_manifest(path: Path, *, allow_generated_dirty: bool = False) -> dict:
    manifest_path = path.resolve()
    manifest = campaign.load_manifest(manifest_path)
    root = Path(manifest["campaign_root"])
    if not root.is_absolute() or root.resolve() != root:
        raise SystemExit("campaign root must be an absolute resolved path")
    if manifest_path != root / "campaign.json":
        raise SystemExit("campaign manifest must be <campaign-root>/campaign.json")
    commit, dirty = campaign.git_identity(REPO)
    if dirty:
        dirty_paths = campaign.git_dirty_paths(REPO)
        allowed_prefixes = (".artifacts/exp022/", ".demolab/pdfs/exp022.pdf")
        if not allow_generated_dirty or any(
            not path.startswith(allowed_prefixes) for path in dirty_paths
        ):
            raise SystemExit("campaign execution requires a clean source worktree")
    if manifest["repository"] != {"commit": commit, "dirty": False}:
        raise SystemExit(
            "campaign manifest does not match the clean checked-out commit: "
            f"manifest={manifest['repository']['commit']} checkout={commit}"
        )
    if manifest.get("environment", {}).get("lockfile") != campaign.lock_identity(REPO):
        raise SystemExit("campaign lockfile identity does not match the checkout")
    tier = manifest.get("selection", {}).get("tier")
    try:
        selected_cells = cells_in_resource_tier(tier)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    manifest_names_list = [row.get("name") for row in manifest.get("cells", [])]
    if len(manifest_names_list) != len(set(manifest_names_list)):
        raise SystemExit("campaign contains duplicate cell names")
    expected_names_list = [cell["name"] for cell in selected_cells]
    if manifest_names_list != expected_names_list:
        raise SystemExit("campaign cell list does not exactly match its declared selection")
    previous_plumbing = os.environ.get("PINGLAB_NB022_PLUMBING")
    runtime_commands = {}
    try:
        if manifest.get("plumbing"):
            os.environ["PINGLAB_NB022_PLUMBING"] = "1"
        else:
            os.environ.pop("PINGLAB_NB022_PLUMBING", None)
        for row in manifest["cells"]:
            spec = _cell_by_name(row["name"])
            assert spec is not None
            samples, epochs = cell_samples_epochs(spec)
            command_spec = ({k: v for k, v in spec.items() if k != "max_samples"}
                            if manifest.get("plumbing") else spec)
            train_args = build_train_args(command_spec, root / "cells" / spec["name"], samples, epochs)
            resolved = campaign.resolved_parameters(
                spec, train_args, samples, epochs,
                scientific_contract=scientific_contract(spec, samples, epochs),
            )
            command = [campaign.python_executable(), str(SNN_TOOL), *train_args]
            output_directory = (root / "cells" / spec["name"]).resolve()
            expected = {
                "name": spec["name"],
                "training_run_id": spec["training_run_id"],
                "family": spec["family"],
                "resource_tier": cell_resource_tier(spec),
                "parameters": resolved,
                "command": command,
                "command_shell": shlex.join(command),
                "output_directory": str(output_directory),
                "required_outputs": list(campaign.REQUIRED_CELL_FILES),
            }
            if row != expected:
                raise SystemExit(f"campaign manifest registry drift for {row['name']}")
            if output_directory.parent != (root / "cells").resolve():
                raise SystemExit(f"campaign output path escapes the cells root: {row['name']}")
            runtime_commands[row["name"]] = command
    finally:
        if previous_plumbing is None:
            os.environ.pop("PINGLAB_NB022_PLUMBING", None)
        else:
            os.environ["PINGLAB_NB022_PLUMBING"] = previous_plumbing
    manifest["_runtime_commands"] = runtime_commands
    return manifest


def _stamp_campaign_identity(directory: Path, manifest: dict, row: dict) -> None:
    _writable_compute_root(directory)
    for filename in ("config.json", "metrics.json"):
        path = directory / filename
        payload = json.loads(path.read_text())
        payload.update({
            "campaign_id": manifest["campaign_id"],
            "campaign_manifest_sha256": manifest["manifest_sha256"],
            "resource_tier": row["resource_tier"],
            "campaign_repository_commit": manifest["repository"]["commit"],
            "campaign_resolved_parameters": row["parameters"],
        })
        nested = payload.get("config")
        if isinstance(nested, dict):
            nested.update({
                "campaign_id": manifest["campaign_id"],
                "campaign_manifest_sha256": manifest["manifest_sha256"],
                "resource_tier": row["resource_tier"],
                "campaign_repository_commit": manifest["repository"]["commit"],
                "campaign_resolved_parameters": row["parameters"],
            })
        campaign.atomic_json(path, payload)


def _gpu_metadata() -> dict:
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return {"available": False}
    if query.returncode != 0:
        return {"available": False}
    return {"available": True, "devices": [line.strip() for line in query.stdout.splitlines()]}


def _campaign_train(manifest_path: Path, name: str, *, recover_stale: bool = False) -> int:
    manifest = _checked_manifest(manifest_path)
    row = campaign.manifest_cell(manifest, name)
    directory = Path(row["output_directory"])
    _writable_compute_root(directory)
    existing = campaign.validate_cell(row)
    if existing["valid"]:
        print(f"[skip-valid] {name} is complete and will not be touched")
        return 0
    record, attempt_lock = campaign.acquire_attempt(
        manifest, row, recover_stale=recover_stale,
    )
    status_path = campaign.status_path(manifest, name)
    exit_code = 1
    attempt_started = time.monotonic()
    try:
        existing = campaign.validate_cell(row)
        if existing["valid"]:
            record.update({
                "ended_at_utc": campaign.utc_now(), "exit_code": 0,
                "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                "state": "complete", "validation": existing,
                "note": "became valid before training ownership was acquired",
            })
            campaign.atomic_json(status_path, record)
            print(f"[skip-valid] {name} became complete and will not be touched")
            return 0
        preserved = campaign.preserve_partial(directory)
        if preserved:
            print(f"[preserve-partial] {directory} -> {preserved}")
        record["gpu"] = _gpu_metadata()
        campaign.atomic_json(status_path, record)
        directory.parent.mkdir(parents=True, exist_ok=True)
        command = manifest["_runtime_commands"][name]
        completed = subprocess.run(command, cwd=REPO)
        exit_code = completed.returncode
        if exit_code == 0:
            spec = _cell_by_name(name)
            assert spec is not None
            old_root = globals()["TRAINING_ROOT"]
            try:
                globals()["TRAINING_ROOT"] = Path(manifest["campaign_root"]) / "cells"
                _stamp_training_run_identity(spec)
            finally:
                globals()["TRAINING_ROOT"] = old_root
            _stamp_campaign_identity(directory, manifest, row)
        validation = campaign.validate_cell(row)
        try:
            metrics_payload = load_metrics(directory)
        except (OSError, ValueError, json.JSONDecodeError):
            metrics_payload = {}
        record.update({
            "ended_at_utc": campaign.utc_now(), "exit_code": exit_code,
            "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
            "state": "complete" if exit_code == 0 and validation["valid"] else "failed",
            "validation": validation,
            "gpu_after": _gpu_metadata(),
            "training_performance": metrics_payload.get("perf"),
            "output_bytes": sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()),
        })
        directory.mkdir(parents=True, exist_ok=True)
        campaign.atomic_json(directory / "attempt.json", record)
        campaign.atomic_json(status_path, record)
        return 0 if record["state"] == "complete" else 1
    except BaseException as exc:
        record.update({
            "ended_at_utc": campaign.utc_now(), "exit_code": exit_code,
            "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
            "state": "failed", "error": f"{type(exc).__name__}: {exc}",
        })
        directory.mkdir(parents=True, exist_ok=True)
        campaign.atomic_json(directory / "attempt.json", record)
        campaign.atomic_json(status_path, record)
        raise
    finally:
        campaign.release_attempt(attempt_lock, record["attempt_id"])


def _handle_campaign_cli(argv: list[str]) -> bool:
    if not any(flag in argv for flag in (
        "--campaign-manifest", "--campaign-status", "--campaign-list",
        "--campaign-train-cell", "--campaign-validate", "--campaign-aggregate",
        "--campaign-import-compatible",
    )):
        return False
    args = _campaign_parser().parse_args(argv[1:])
    if args.campaign_import_compatible:
        if args.from_campaign is None:
            raise SystemExit("--from-campaign is required")
        destination = _checked_manifest(args.campaign_import_compatible)
        print(json.dumps(
            _import_compatible_cells(destination, args.from_campaign),
            indent=2,
            sort_keys=True,
        ))
        return True
    if args.campaign_manifest:
        if not args.campaign_id:
            raise SystemExit("--campaign-id is required")
        try:
            selected = cells_in_resource_tier(args.tier)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.plumbing:
            os.environ["PINGLAB_NB022_PLUMBING"] = "1"
        root = args.campaign_manifest.resolve()
        _writable_compute_root(root)
        manifest = campaign.create_manifest(
            repo=REPO, campaign_root=root, campaign_id=args.campaign_id,
            cells=selected, tier_for=cell_resource_tier,
            samples_epochs=cell_samples_epochs, build_args=build_train_args,
            scientific_contract_for=scientific_contract,
            plumbing=args.plumbing,
            selection_tier=args.tier,
        )
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise SystemExit(
                f"campaign destination already exists and will not be modified: {root}"
            ) from exc
        for child in ("cells", "logs", "status", "submissions"):
            (root / child).mkdir()
        manifest["pingstore_run_id"] = reserve_stage(
            REPO / ".pingstore", SLUG, "compute", origin=args.execution_origin
        )
        campaign.write_manifest(root / "campaign.json", manifest)
        print(root / "campaign.json")
        return True
    manifest_path = (args.campaign or args.campaign_status or args.campaign_list
                     or args.campaign_validate or args.campaign_aggregate)
    if manifest_path is None:
        raise SystemExit("--campaign MANIFEST is required")
    manifest = _checked_manifest(manifest_path)
    if args.campaign_train_cell:
        raise SystemExit(_campaign_train(
            manifest_path, args.campaign_train_cell,
            recover_stale=args.recover_stale,
        ))
    if args.campaign_validate:
        print(f"valid manifest {manifest['campaign_id']} {manifest['manifest_sha256']}")
        return True
    status = campaign.summarize_status(manifest)
    if args.campaign_aggregate:
        if len(manifest["cells"]) != len(CANONICAL_CELLS):
            raise SystemExit("aggregation requires the complete 102-cell registry")
        incomplete = [row["name"] for row in status["cells"] if not row["valid"]]
        if incomplete:
            raise SystemExit(
                f"aggregation refused: {len(incomplete)} cells are not valid"
            )
        capture_campaign(manifest_path, manifest)
        return True
    if args.campaign_list:
        cells = [cell for cell in manifest["cells"] if args.tier == "all" or cell["resource_tier"] == args.tier]
        if args.retry_only:
            retry = set(status["retry_cells"])
            cells = [cell for cell in cells if cell["name"] in retry]
        print("\n".join(cell["name"] for cell in cells))
    elif args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        campaign.print_status(status)
    return True



def _writable_compute_root(directory: Path) -> None:
    resolved = directory.resolve()
    runs = (REPO / ".pingstore/runs").resolve()
    if resolved == runs or resolved == REPO.resolve():
        raise PingstoreError("compute output must be a dedicated working directory")
    if runs in resolved.parents:
        identity = resolved.relative_to(runs).parts[0]
        if not (identity.startswith(".") and identity.endswith(".tmp")):
            raise PingstoreError("compute cannot modify a completed Pingstore run")


def generate_snapshots(bank: Path, output: Path) -> None:
    """Retain fixed digit-0/sample-0 probes; no plotting or discarded recordings."""
    for cell in CANONICAL_CELLS:
        if cell["seed"] != 42:
            continue
        trained = bank / cell["name"]
        checkpoint = resolve_checkpoint(trained, RESULT_CHECKPOINT_ROLE)
        destination = output / cell["name"]
        if destination.exists():
            raise PingstoreError(f"probe output already exists: {destination}")
        args = [
            sys.executable, str(SNN_TOOL), "sim", "--infer",
            "--load-config", str(trained / "config.json"),
            "--load-weights", str(checkpoint["path"]),
            "--digit", "0", "--sample", "0", "--out-dir", str(destination),
        ]
        if cell["family"] == "variable_rate":
            args += ["--input-rate", "5"]
        print(f"[compute probe] {cell['name']}", flush=True)
        subprocess.run(args, cwd=REPO, check=True)
        write_json_atomic(destination / "probe-command.json", {
            "command": args,
            "checkpoint": {key: value for key, value in checkpoint.items() if key != "path"},
        })


def copy_bank(bank: Path, destination: Path) -> list[dict]:
    """Copy scientific evidence without restamping configs or checkpoint roles."""
    from pingstore.contracts import file_sha256

    expected = {cell["name"] for cell in CANONICAL_CELLS}
    actual = {path.name for path in bank.iterdir() if path.is_dir()}
    if actual != expected:
        raise PingstoreError(
            f"bank must contain the 102 registered cells; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    inventory = []
    for name in sorted(expected):
        cell = bank / name
        for required in ("config.json", "metrics.json", "weights.pth", "weights_final.pth"):
            if not (cell / required).is_file():
                raise PingstoreError(f"missing scientific payload: {cell / required}")
        for role in ("best_validation", "final_epoch"):
            resolve_checkpoint(cell, role)
        for path in sorted(cell.rglob("*")):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise PingstoreError(f"unsupported bank entry: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(bank)
            digest = file_sha256(path)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if file_sha256(target) != digest:
                raise PingstoreError(f"import checksum mismatch: {relative}")
            inventory.append({"path": relative.as_posix(), "sha256": digest,
                              "size_bytes": target.stat().st_size})
    return inventory


def import_bank(identity: str, *, run_id: str | None = None) -> str:
    """Copy an explicit v3 compute bank without training; legacy import is separate."""
    source = source_run(REPO / ".pingstore", identity, stage="compute", experiment=SLUG)
    with stage_run(REPO, SLUG, "compute", inputs={"import": source}, run_id=run_id,
                   configuration=source.record["execution"].get("configuration"),
                   export_root="export/cells", operation="import") as run:
        inventory = copy_bank(source.export, run.export / "cells")
        shutil.copy2(source.directory / "run.json", run.evidence / "imported-run.json")
        if (source.directory / "README.md").is_file():
            shutil.copy2(source.directory / "README.md", run.evidence / "imported-README.md")
        write_json_atomic(run.evidence / "import-inventory.json", inventory)
        run.record["historical_evidence"] = {
            "source": source.reference,
            "record": "export/evidence/imported-run.json",
            "note": "Historical cell attempts and inherited/repaired lineage are preserved; "
                    "this execution copied evidence and did not train or simulate.",
        }
        (run.directory / "README.md").write_text(
            "# Exp022 compute — imported model bank\n\n"
            f"Imported byte-preserving scientific evidence from `{identity}`. "
            "The original remains unchanged.\n\n"
            "The 102 cells and both checkpoint roles are under `export/cells/`. "
            "Source execution and full lineage are retained in "
            "`export/evidence/imported-run.json`; this run's local operation is an import, "
            "not historical SLURM execution or retraining.\n\n"
            "Raw raster snapshots were not retained in this historical bank. "
            "Analysis can recover training curves from metrics; presentation must either "
            "explicitly carry historical rasters from the original run or use a new "
            "compute diagnostics run. No simulation is triggered by analyse/present.\n"
        )
    return run.run_id


def capture_campaign(manifest_path: Path, manifest: dict) -> str:
    """Finish campaign computation, including retained probes, without presentation."""
    reserved = manifest.get("pingstore_run_id")
    if reserved is None:
        raise PingstoreError(
            "legacy campaign has no preallocated stage identity; import its completed "
            "Pingstore bank with compute.py --import-source RUN instead"
        )
    bank = Path(manifest["campaign_root"]) / "cells"
    with stage_run(REPO, SLUG, "compute", run_id=reserved, configuration=SCALE,
                   export_root="export/cells", operation="capture-campaign") as run:
        copy_bank(bank, run.export / "cells")
        shutil.copy2(manifest_path, run.evidence / "campaign.json")
        run.record["execution"]["campaign"] = {
            "campaign_id": manifest["campaign_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "repository_commit": manifest["repository"]["commit"],
        }
        generate_snapshots(run.export / "cells", run.export / "snapshots")
        final = campaign.summarize_status(_checked_manifest(manifest_path))
        if any(not row["valid"] for row in final["cells"]):
            raise PingstoreError("campaign changed during compute capture")
    return run.run_id


def main() -> None:
    retired = {"--skip-training", "--plot-only", "--only-missing"} & set(sys.argv[1:])
    if retired:
        raise SystemExit(
            "combined lifecycle flags are retired: use analyse.py --source COMPUTE_RUN "
            "or present.py --source ANALYSIS_RUN; use campaign retries for compute recovery"
        )
    if _handle_campaign_cli(sys.argv):
        return
    if any(flag in sys.argv for flag in ("--runpod", "--reap", "--pod-run", "--train-cell",
                                         "--list-cells")):
        meta, _reserved = _dispatch_meta(sys.argv)
        if meta.list_cells:
            print("\n".join(cell["name"] for cell in cells_in_resource_tier(meta.list_cells)))
            return
        _writable_compute_root(TRAINING_ROOT)
        if meta.train_cell:
            cell = _cell_by_name(meta.train_cell)
            if cell is None:
                raise SystemExit(f"unknown cell: {meta.train_cell}")
            _train_one_cell(cell, meta.plumbing)
        elif meta.reap:
            runpod.reap_all_pods()
        elif meta.pod_run:
            pod_run()
        else:
            run_via_runpod(sys.argv)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="identity already reserved before dispatch")
    parser.add_argument("--import-source", help="copy an explicit v3 compute bank without computation")
    parser.add_argument("--source", help="compute bank for new retained diagnostic probes")
    parser.add_argument("--diagnostics", action="store_true",
                        help="simulate fixed probes only; requires --source")
    args = parser.parse_args()
    if args.import_source:
        if args.source or args.diagnostics:
            parser.error("--import-source cannot be combined with diagnostic execution")
        import_bank(args.import_source, run_id=args.run_id)
        return
    if bool(args.source) != args.diagnostics:
        parser.error("--source and --diagnostics must be used together")
    inputs = {}
    if args.source:
        inputs["bank"] = source_run(REPO / ".pingstore", args.source,
                                   stage="compute", experiment=SLUG)
        if not (inputs["bank"].export / CANONICAL_CELLS[0]["name"]).is_dir():
            parser.error("--source must be a compute run exporting a model bank")
    with stage_run(REPO, SLUG, "compute", inputs=inputs, run_id=args.run_id,
                   configuration=SCALE, export_root="export" if inputs else "export/cells") as run:
        if inputs:
            bank = inputs["bank"].export
        else:
            bank = run.export / "cells"
            previous = globals()["TRAINING_ROOT"]
            try:
                globals()["TRAINING_ROOT"] = bank
                for cell in CANONICAL_CELLS:
                    _train_one_cell(cell, os.environ.get("PINGLAB_NB022_PLUMBING") == "1")
            finally:
                globals()["TRAINING_ROOT"] = previous
        generate_snapshots(bank, run.export / "snapshots")


if __name__ == "__main__":
    main()
