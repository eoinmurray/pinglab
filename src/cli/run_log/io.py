"""Sidecar writers: metrics.jsonl (append-only) and test_predictions.json."""

from __future__ import annotations

import datetime
import json
from pathlib import Path


class MetricsJsonl:
    """Append-only JSONL writer for per-epoch (or per-step) metrics."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")

    def write(self, **fields):
        fields.setdefault(
            "timestamp", datetime.datetime.now().isoformat(timespec="seconds")
        )
        self._f.write(json.dumps(fields) + "\n")
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def write_test_predictions(path: Path, predictions: list):
    """Save list of {idx, true, pred, correct, logits} records to JSON."""
    with open(path, "w") as f:
        json.dump(predictions, f, indent=2, default=float)
