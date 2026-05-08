"""Notebook runner for entry 015 — Classification Latency.

There is no separate runner for this entry: the trained weights and
the latency / per-trial analysis are produced as a side effect of
nb016 (sequential MNIST tracking), which trains cuba/coba/ping once
and writes artifacts for both notebooks. Splitting the runner would
just duplicate the training step.

This shim re-invokes nb016.py with the user's CLI flags so
`uv run src/pinglab/notebooks/nb015.py [--tier ...] [--modal-gpu ...]`
works as the entry's reproduction command.

Notebook entry: src/docs/src/pages/notebooks/nb015.mdx
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    nb016 = REPO / "src" / "pinglab" / "notebooks" / "nb016.py"
    if not nb016.exists():
        raise SystemExit(f"missing {nb016}")
    print(f"[nb015] delegating to {nb016.relative_to(REPO)} (shared training)")
    os.execv(sys.executable, [sys.executable, str(nb016), *sys.argv[1:]])


if __name__ == "__main__":
    main()
