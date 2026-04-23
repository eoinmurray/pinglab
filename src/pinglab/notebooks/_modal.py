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


class BatchDispatcher:
    """Queue oscilloscope calls for parallel Modal dispatch.

    When modal_gpu is None, each submit() runs synchronously via sh.uv
    (drop-in for the existing local path). When modal_gpu is set, submit()
    queues the cell; drain() then spawns every queued cell inside one
    app.run() context so Modal cold-start + lifecycle is paid once per
    batch instead of once per cell.

    Per-cell GPU override (gpu_override=) supports heterogeneous batches,
    e.g. ping@dt=0.1 needs A10G while the other models at dt=0.1 fit on T4.
    """

    def __init__(self, modal_gpu, repo_path, oscilloscope_path):
        self.modal_gpu = modal_gpu
        self.repo_path = repo_path
        self.oscilloscope_path = oscilloscope_path
        self.pending: list[dict] = []

    def submit(self, osc_args, local_out_dir, gpu_override=None):
        """Run (local) or queue (modal) an oscilloscope invocation.

        osc_args: list starting at the subcommand (e.g. ['train', '--model', ...]).
        local_out_dir: the path the cell writes to (matches --out-dir in osc_args).
        gpu_override: force this cell onto a specific GPU, ignoring self.modal_gpu.
        """
        import sh
        import sys
        cell_gpu = gpu_override or self.modal_gpu
        if cell_gpu is None:
            sh.uv("run", "python", str(self.oscilloscope_path), *osc_args,
                  _cwd=str(self.repo_path), _out=sys.stdout, _err=sys.stderr)
        else:
            self.pending.append({
                "cli_args": list(osc_args),
                "gpu": cell_gpu,
                "local_out_dir": str(local_out_dir),
            })

    def drain(self):
        """Dispatch all queued jobs to Modal in parallel, wait, sync artifacts."""
        if not self.pending:
            return
        from pinglab.modal_app import dispatch_batch_to_modal
        dispatch_batch_to_modal(self.pending)
        self.pending.clear()
