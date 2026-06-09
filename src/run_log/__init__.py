"""Oscilloscope run logging: structured intro / progress / summary.

This package re-exports every public name from its sub-modules so
existing callers can keep doing `from run_log import X` or `run_log.X`
without caring about the new layout.

Sub-modules:
    .ansi        — terminal color + width primitives
    .provenance  — git SHA, env hash, run ID, .running marker
    .progress    — intro / per-epoch / summary printing + WarningTracker
    .io          — metrics.jsonl + test_predictions.json sidecars
"""

from .ansi import (  # noqa: F401
    DIVIDER,
    WIDTH,
    _strip_ansi,
    bold,
    c,
    green,
    red,
    yellow,
)
from .io import MetricsJsonl, write_test_predictions  # noqa: F401
from .progress import (  # noqa: F401
    WarningTracker,
    _fmt_kv,
    format_bytes,
    format_eta,
    list_output_files,
    print_epoch,
    print_intro,
    print_progress_header,
    print_summary,
)
from .provenance import (  # noqa: F401
    _env_hash,
    _git_sha,
    provenance,
    run_id,
    write_running_marker,
)
