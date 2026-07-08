"""The one way an experiment runner invokes the SNN CLI: `sh.uv`.

Every runner shells out to tools/snn/tool.py to train or infer. `run_cli` wraps
that single call so every runner spawns identically, replacing the mix of
`subprocess.run([sys.executable, ...])` and ad-hoc `uv run` invocations.

On a laptop it runs `uv run python tool.py ...`. On a RunPod pod the deps are
baked into the image, so re-resolving the venv is wasteful and can fail offline;
there the pod sets PINGLAB_NO_SYNC=1 and this runs `uv run --no-sync python ...`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .paths import REPO

SNN_TOOL = REPO / "tools" / "snn" / "tool.py"


def run_cli(args, *, no_sync: bool | None = None, cwd: Path | None = None) -> None:
    """Invoke the SNN CLI with `args` (list starting at the subcommand, e.g.
    ['train', '--epochs', '50']). Raises sh.ErrorReturnCode on non-zero exit.

    no_sync: force `uv run --no-sync`. Defaults to the PINGLAB_NO_SYNC env var
    (set on pods), so laptops sync and pods don't."""
    import sh

    if no_sync is None:
        no_sync = os.environ.get("PINGLAB_NO_SYNC") == "1"
    run_args = ["run"]
    if no_sync:
        run_args.append("--no-sync")
    run_args += ["python", str(SNN_TOOL), *(str(a) for a in args)]
    sh.uv(  # ty: ignore[unresolved-attribute]  # sh builds command attrs at runtime
        *run_args,
        _cwd=str(cwd or REPO),
        _out=sys.stdout,
        _err=sys.stderr,
    )
