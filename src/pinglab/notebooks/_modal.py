"""Shared helpers for notebook runners to forward --modal-gpu to oscilloscope.

Every notebook runner invokes src/pinglab/oscilloscope.py one or more times via
sh.uv. When the runner is given --modal-gpu <GPU>, each oscilloscope call gets
--modal --modal-gpu <GPU> appended so the job runs on Modal instead of the
local machine.
"""
from __future__ import annotations

VALID_GPUS = ("none", "T4", "L4", "A10G", "A100", "H100")


def parse_modal_gpu(argv: list[str]) -> str | None:
    """Parse --modal-gpu <GPU> out of argv; return None if absent."""
    if "--modal-gpu" not in argv:
        return None
    idx = argv.index("--modal-gpu")
    if idx + 1 >= len(argv):
        raise SystemExit("--modal-gpu requires a value")
    gpu = argv[idx + 1]
    if gpu not in VALID_GPUS:
        raise SystemExit(
            f"--modal-gpu: unknown GPU {gpu!r}, choose from {list(VALID_GPUS)}"
        )
    return gpu


def append_modal_args(args: list[str], modal_gpu: str | None) -> list[str]:
    """Append --modal --modal-gpu <GPU> to args when modal_gpu is set."""
    if modal_gpu:
        return [*args, "--modal", "--modal-gpu", modal_gpu]
    return list(args)
