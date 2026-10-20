"""Validate and freeze a generic experiment command before reserving paid work.

Scientific settings remain in the experiment's recipe and CLI. This module
checks the common compute contract without importing experiment code or Modal.
"""

import ast
import re
from pathlib import Path

from pingstore.contracts import EXPERIMENT_RE, STAGE_ID_RE
from pingstore.stages import operation_lock

from . import inputs as source_inputs
from . import jobs


def execution_environment(gpu, cpu, smoke=False):
    device = "cpu" if gpu.lower() == "none" else "cuda"
    return {
        "SNNLAB_DEVICE": device,
        "PINGLAB_DEVICE": device,
        "EXP080_DEVICE": device,
        "EXP081_DEVICE": device,
        "PINGLAB_SMOKE": "1" if smoke else "0",
        "PINGLAB_CPU_THREADS": str(cpu),
        "PINGLAB_MODAL_CPU_CORES": str(cpu),
        "OMP_NUM_THREADS": str(cpu),
        "MKL_NUM_THREADS": str(cpu),
        "PINGLAB_DATASET_CACHE": "/modal-output/datasets",
    }


def validate_request(repo, experiment, arguments, environment):
    if not isinstance(experiment, str) or not EXPERIMENT_RE.fullmatch(experiment):
        raise ValueError("experiment must be an expXXX identifier")
    entrypoint = Path(repo) / "experiments" / experiment / "compute.py"
    if not entrypoint.is_file():
        raise ValueError(f"missing experiment compute entrypoint: {entrypoint}")
    tree = ast.parse(entrypoint.read_text())
    flags = {
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }
    if "--run-id" not in flags:
        raise ValueError(
            "compute entrypoint must accept the Pingstore --run-id contract"
        )
    if not isinstance(arguments, list) or any(
        not isinstance(a, str) for a in arguments
    ):
        raise ValueError("experiment arguments must be a list of strings")
    for argument in arguments:
        flag = argument.split("=", 1)[0]
        # Block argparse abbreviations too: the dispatcher owns the reservation.
        if flag.startswith("--") and ("--run-id".startswith(flag) or flag == "--"):
            raise ValueError(
                "the dispatcher owns --run-id; do not pass it to the experiment"
            )
        if flag in {"-h", "--help"}:
            raise ValueError("inspect compute.py --help locally before dispatch")
        if flag.startswith("--") and flag not in flags:
            raise ValueError(f"unknown experiment argument: {flag}")
    if not isinstance(environment, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in environment.items()
    ):
        raise ValueError("environment must map names to strings")
    if environment.get("PINGLAB_SMOKE") == "1" and not any(
        "PINGLAB_SMOKE" in path.read_text()
        for path in (entrypoint, entrypoint.with_name("recipe.py"))
        if path.is_file()
    ):
        raise ValueError("experiment does not declare a PINGLAB_SMOKE profile")
    return entrypoint


def dispatch(
    repo,
    experiment,
    worker,
    volume,
    *,
    app_id,
    arguments,
    inputs,
    environment,
    gpu="A10G",
    cpu=4,
    memory=16384,
    timeout=86400,
    store=None,
):
    validate_dispatch(
        repo, experiment, arguments, inputs, environment, cpu, memory, timeout
    )
    root = Path(store).expanduser().absolute() if store else Path(repo) / ".pingstore"
    options = dict(
        gpu=None if gpu.lower() == "none" else gpu,
        cpu=cpu,
        memory=memory,
        timeout=timeout,
    )
    plan = {
        "entrypoint": f"experiments/{experiment}/compute.py",
        "arguments": arguments,
        "environment": environment,
        "resources": options,
    }
    with operation_lock(root, exclusive=False):
        pins, sources = source_inputs.prepare_inputs(root, inputs)
        source_inputs.upload_inputs(volume, sources)
        return jobs.submit(
            root,
            experiment,
            worker,
            app_id=app_id,
            options=options,
            arguments={
                "arguments": arguments,
                "environment": environment,
                "input_refs": pins,
                "input_runs": [source.reference for source in sources.values()],
            },
            inputs=pins,
            plan=plan,
        )


def validate_dispatch(
    repo, experiment, arguments, inputs, environment, cpu, memory, timeout
):
    """Reject invalid commands before loading the provider or reserving a run."""
    validate_request(repo, experiment, arguments, environment)
    if cpu < 1 or memory < 1 or not 1 <= timeout <= 86400:
        raise ValueError("cpu/memory must be positive; timeout must be 1–86400 seconds")
    if not isinstance(inputs, dict) or any(
        not isinstance(role, str)
        or not role
        or not isinstance(identity, str)
        or not STAGE_ID_RE.fullmatch(identity)
        for role, identity in inputs.items()
    ):
        raise ValueError("inputs must map explicit roles to completed run IDs")
    # Catch omitted transfer pins before spending on a worker that cannot resolve them.
    referenced = {
        token
        for arg in arguments
        for token in re.split(r"[=,]", arg)
        if STAGE_ID_RE.fullmatch(token)
    }
    if referenced - set(inputs.values()):
        raise ValueError(
            "each source run argument must have an explicit --input ROLE=RUN_ID"
        )
