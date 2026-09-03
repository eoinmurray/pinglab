"""Cross-process coordination for run writers and destructive maintenance."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

from .contracts import PingstoreError

LOCK_NAME = ".operation.lock"


@contextmanager
def operation_lock(root: Path, *, exclusive: bool):
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        try:
            fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            action = "prune" if exclusive else "reserve or execute a run"
            raise PingstoreError(
                f"cannot {action} while another Pingstore operation is active"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
