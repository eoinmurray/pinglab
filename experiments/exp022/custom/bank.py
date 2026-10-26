"""Read the selected exp022 bank and its retained epoch histories.

The compute coordinator installs an explicit working root. Analysis consumes
validated run directories; this module never discovers a latest bank or launches
training. Model definitions and scientific settings remain in recipe.py.
"""

import json
import os
from pathlib import Path

from experiments.exp022.custom.checkpoints import epoch_metrics

REPO = Path(__file__).resolve().parents[3]

TRAINING_ROOT: Path | None = (
    Path(os.environ["PINGLAB_TRAINING_ROOT"])
    if os.environ.get("PINGLAB_TRAINING_ROOT")
    else None
)


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


def training_root() -> Path:
    if TRAINING_ROOT is None:
        raise RuntimeError(
            "exp022 training root is not configured; use staged execution or set "
            "PINGLAB_TRAINING_ROOT explicitly"
        )
    return TRAINING_ROOT


def load_metrics(d: Path) -> dict:
    p = d / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def final_rates(d: Path) -> tuple[float, float]:
    """Last-epoch E / I rate (Hz) from the retained epoch record."""
    rows = epoch_metrics(d)
    if not rows:
        return float("nan"), float("nan")
    row = rows[-1]
    return (
        float(row.get("test_rate_e", row.get("rate_e", float("nan")))),
        float(row.get("test_rate_i", row.get("rate_i", float("nan")))),
    )


def training_curve(d: Path) -> tuple[list[int], list[float]]:
    """Per-epoch (epoch, validation accuracy) from the retained epoch record."""
    eps, accs = [], []
    for r in epoch_metrics(d):
        if "ep" in r and "acc" in r:
            eps.append(int(r["ep"]))
            accs.append(float(r["acc"]))
    return eps, accs
