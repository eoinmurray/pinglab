"""Receipt-first Slurm submission shared by experiment-local HPC adapters."""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pingstore.contracts import PingstoreError, write_json_atomic


def submit_pipeline(
    *,
    steps: Sequence[str],
    command_for: Callable[[str, str | None], list[str]],
    receipt: Path,
    live: bool,
    test_only: bool,
    context: Mapping[str, Any] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, str]:
    """Print, validate, or submit an ordered dependency chain exactly once."""
    if live and test_only:
        raise PingstoreError("live submission and test-only are mutually exclusive")
    if receipt.exists():
        raise PingstoreError(
            "submission was already attempted; inspect its receipt and Slurm before recovery"
        )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "pinglab.slurm-submission/v1",
        "status": "submitting",
        "context": dict(context or {}),
        "jobs": {},
        "commands": {},
    }
    if live:
        # Persist intent before the first external mutation.  An ambiguous sbatch
        # response must never become permission for an automatic resubmission.
        write_json_atomic(receipt, payload)
    jobs: dict[str, str] = {}
    dependency = None
    for step in steps:
        command = command_for(step, dependency)
        payload["commands"][step] = command
        print(shlex.join(command), flush=True)
        if live or test_only:
            actual = command if live else [command[0], "--test-only", *command[1:]]
            result = runner(actual, capture_output=True, text=True)
            print(result.stdout + result.stderr, end="", flush=True)
            result.check_returncode()
            if live:
                job_id = result.stdout.strip().split(";", 1)[0]
                if not job_id.isdigit():
                    raise PingstoreError(
                        "ambiguous Slurm response; inspect the receipt and scheduler"
                    )
                jobs[step] = job_id
                payload["jobs"] = jobs
                write_json_atomic(receipt, payload)
                dependency = job_id
        if not live:
            dependency = None
    if live:
        payload["status"] = "submitted"
        write_json_atomic(receipt, payload)
    else:
        print("No jobs submitted.")
    return jobs
