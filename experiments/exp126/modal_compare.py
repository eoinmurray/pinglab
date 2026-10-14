"""Dispatch independent matched implementations concurrently on separate GPUs."""

import shutil
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp126.recipe import condition_role
from experiments.helpers.modal_compute import (
    app,
    assemble_exp126,
    compute,
    compute_pair,
    results,
)
from pingstore.contracts import load_json, validate_operational_run_directory
from pingstore.stages import operation_lock, reserve_stage, stage_reservation


def fetch(
    run_id, reservation, dispatch, files, backend=None, seed=None, expected_inputs=None
):
    root = REPO / ".pingstore"
    temporary = root / "runs" / f".{run_id}.tmp"
    download = temporary / ".scratch" / run_id
    download.mkdir(parents=True)
    remote_path = f"{dispatch}/.pingstore/runs/{run_id}"
    for filename in files:
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe remote result path")
        target = download / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            for block in results.read_file(f"{remote_path}/{filename}"):
                handle.write(block)
    record = validate_operational_run_directory(download)
    if (
        record["run_id"] != run_id
        or record["origin"] != reservation["origin"]
        or record["experiment"] != "exp126"
        or record["stage"] != "compute"
        or load_json(temporary / ".reservation.json") != reservation
    ):
        raise ValueError("downloaded run does not match dispatch")
    if backend is not None:
        if (
            record.get("scope") != "phase"
            or record.get("phase") != backend
            or record["inputs"]
            or record.get("implementation") != backend
            or record["execution"]["configuration"]["seed"] != seed
        ):
            raise ValueError("downloaded phase does not match condition")
    elif (
        record.get("scope") != "full"
        or record["execution"]["operation"] != "assemble"
        or record["inputs"] != expected_inputs
    ):
        raise ValueError("downloaded full run does not match assembly")
    destination = root / "runs" / run_id
    if destination.exists():
        raise FileExistsError(destination)
    download.rename(destination)
    shutil.rmtree(temporary)
    return record


@app.local_entrypoint()
def compare(
    implementation: str = "both",
    seeds: str = "42",
    gpu: str = "A10",
    cpu: int = 4,
    assemble_on_modal: bool = False,
):
    if implementation not in ("both", "snntorch", "snnlab"):
        raise ValueError("implementation must be both, snntorch or snnlab")
    chosen_seeds = [int(value.strip()) for value in seeds.split(",")]
    if (
        not chosen_seeds
        or len(set(chosen_seeds)) != len(chosen_seeds)
        or any(value < 0 for value in chosen_seeds)
    ):
        raise ValueError("seeds must be unique nonnegative integers")
    if cpu < 1 or gpu.lower() == "none":
        raise ValueError("this matched Modal protocol requires a GPU and positive CPUs")
    backends = ("snntorch", "snnlab") if implementation == "both" else (implementation,)
    if assemble_on_modal and implementation != "both":
        raise ValueError("remote assembly requires both phases")
    # One input per container is Modal's default. Two active containers allow
    # paired conditions to overlap without competing for the same GPU.
    function = compute.with_options(gpu=gpu, cpu=cpu, max_containers=2)
    jobs = []
    with operation_lock(REPO / ".pingstore", exclusive=False):
        for seed in chosen_seeds:
            for backend in backends:
                dispatch = uuid.uuid4().hex
                run_id = reserve_stage(
                    REPO / ".pingstore",
                    "exp126",
                    "compute",
                    origin=f"modal-{dispatch}",
                    phase=backend,
                )
                reservation = stage_reservation(
                    REPO / ".pingstore/runs" / f".{run_id}.tmp"
                )
                call = function.spawn(
                    "exp126",
                    reservation,
                    dispatch,
                    False,
                    "cuda",
                    cpu,
                    implementation=backend,
                    seed=seed,
                )
                jobs.append((backend, seed, run_id, reservation, dispatch, call))
                print(
                    f"Spawned {backend} seed={seed}: {run_id}; call={call.object_id}; "
                    f"volume=pinglab-compute-results/{dispatch}/.pingstore/runs/{run_id}",
                    flush=True,
                )
        # All calls are submitted before the first wait. Waiting/download order
        # cannot serialize their GPU computation. Retrieve successful siblings
        # even if one condition fails; never automatically resubmit training.
        errors = []
        completed = []
        for backend, seed, run_id, reservation, dispatch, call in jobs:
            try:
                files = call.get()
                completed.append((backend, seed, run_id, reservation, dispatch, files))
                if not assemble_on_modal:
                    fetch(run_id, reservation, dispatch, files, backend, seed)
                print(f"Completed {backend} seed={seed}: {run_id}", flush=True)
            except Exception as error:
                errors.append(f"{backend} seed={seed} {run_id}: {error}")
                print(errors[-1], file=sys.stderr, flush=True)
        if errors:
            if assemble_on_modal:
                for backend, seed, run_id, reservation, dispatch, files in completed:
                    fetch(run_id, reservation, dispatch, files, backend, seed)
            raise RuntimeError("\n".join(errors))
        if assemble_on_modal:
            dispatch = uuid.uuid4().hex
            full_id = reserve_stage(
                REPO / ".pingstore", "exp126", "compute", origin=f"modal-{dispatch}"
            )
            reservation = stage_reservation(
                REPO / ".pingstore/runs" / f".{full_id}.tmp"
            )
            print(
                f"Assembling on Modal: {full_id}; volume=pinglab-compute-results/{dispatch}/.pingstore/runs/{full_id}",
                flush=True,
            )
            files = assemble_exp126.remote(
                reservation,
                dispatch,
                [{"run_id": row[2], "dispatch": row[4]} for row in completed],
            )
            pins = {}
            for (
                backend,
                seed,
                run_id,
                phase_reservation,
                phase_dispatch,
                phase_files,
            ) in completed:
                record = fetch(
                    run_id,
                    phase_reservation,
                    phase_dispatch,
                    phase_files,
                    backend,
                    seed,
                )
                pins[condition_role(backend, seed)] = {
                    "run_id": run_id,
                    "payload_digest": record["payload_digest"],
                }
            fetch(full_id, reservation, dispatch, files, expected_inputs=pins)
            print(f"Downloaded assembled full compute: {full_id}", flush=True)


@app.local_entrypoint()
def assemble_retained(phase_runs: str, phase_dispatches: str, seed: int = 42):
    """Explicit recovery: assemble existing remote phases without training."""
    from experiments.exp126.recipe import PHASE_EXPORT_FILES

    identities = phase_runs.split(",")
    dispatches = phase_dispatches.split(",")
    if len(identities) != 2 or len(dispatches) != 2 or len(set(identities)) != 2:
        raise ValueError("provide exactly two explicit phase IDs and Volume dispatches")
    completed = []
    with operation_lock(REPO / ".pingstore", exclusive=False):
        for identity, dispatch in zip(identities, dispatches, strict=True):
            reservation = stage_reservation(
                REPO / ".pingstore/runs" / f".{identity}.tmp"
            )
            if reservation["origin"] != f"modal-{dispatch}" or reservation.get(
                "phase"
            ) not in ("snntorch", "snnlab"):
                raise ValueError("phase reservation does not match remote dispatch")
            completed.append((reservation["phase"], identity, dispatch, reservation))
        if {row[0] for row in completed} != {"snntorch", "snnlab"}:
            raise ValueError("retained phases must cover both implementations")
        dispatch = uuid.uuid4().hex
        identity = reserve_stage(
            REPO / ".pingstore", "exp126", "compute", origin=f"modal-{dispatch}"
        )
        reservation = stage_reservation(REPO / ".pingstore/runs" / f".{identity}.tmp")
        print(
            f"Assembling retained phases on Modal: {identity}; dispatch={dispatch}",
            flush=True,
        )
        files = assemble_exp126.remote(
            reservation,
            dispatch,
            [{"run_id": row[1], "dispatch": row[2]} for row in completed],
        )
        pins = {}
        phase_files = [
            "run.json",
            "README.md",
            *[f"export/{name}" for name in PHASE_EXPORT_FILES],
        ]
        for backend, phase_id, phase_dispatch, phase_reservation in completed:
            record = fetch(
                phase_id, phase_reservation, phase_dispatch, phase_files, backend, seed
            )
            pins[condition_role(backend, seed)] = {
                "run_id": phase_id,
                "payload_digest": record["payload_digest"],
            }
        fetch(identity, reservation, dispatch, files, expected_inputs=pins)
        print(f"Downloaded assembled full compute: {identity}", flush=True)


@app.local_entrypoint()
def compare_coallocated(seed: int = 42, gpu: str = "A10", cpu_per_phase: int = 4):
    """Explicit parallel two-GPU compute, remote assembly, ancestry download."""
    if seed < 0 or cpu_per_phase < 1:
        raise ValueError("seed must be nonnegative and CPU count positive")
    with operation_lock(REPO / ".pingstore", exclusive=False):
        phases = []
        for implementation in ("snntorch", "snnlab"):
            dispatch = uuid.uuid4().hex
            identity = reserve_stage(
                REPO / ".pingstore",
                "exp126",
                "compute",
                phase=implementation,
                origin=f"modal-{dispatch}",
            )
            reservation = stage_reservation(
                REPO / ".pingstore/runs" / f".{identity}.tmp"
            )
            phases.append(
                {
                    "implementation": implementation,
                    "seed": seed,
                    "dispatch": dispatch,
                    "reservation": reservation,
                }
            )
            print(f"Reserved paired phase: {identity}; dispatch={dispatch}", flush=True)
        completed = compute_pair.with_options(
            gpu=f"{gpu}:2", cpu=2 * cpu_per_phase
        ).remote(phases, cpu_per_phase)
        dispatch = uuid.uuid4().hex
        identity = reserve_stage(
            REPO / ".pingstore", "exp126", "compute", origin=f"modal-{dispatch}"
        )
        reservation = stage_reservation(REPO / ".pingstore/runs" / f".{identity}.tmp")
        print(
            f"Assembling paired phases on Modal: {identity}; dispatch={dispatch}",
            flush=True,
        )
        files = assemble_exp126.remote(
            reservation,
            dispatch,
            [
                {"run_id": row["reservation"]["run_id"], "dispatch": row["dispatch"]}
                for row in completed
            ],
        )
        pins = {}
        for row in completed:
            phase_id = row["reservation"]["run_id"]
            record = fetch(
                phase_id,
                row["reservation"],
                row["dispatch"],
                row["files"],
                row["implementation"],
                seed,
            )
            pins[condition_role(row["implementation"], seed)] = {
                "run_id": phase_id,
                "payload_digest": record["payload_digest"],
            }
        fetch(identity, reservation, dispatch, files, expected_inputs=pins)
        print(f"Downloaded assembled full compute: {identity}", flush=True)
