"""Regression check: every calibrated model still reaches its accuracy threshold.

Skips when calibration artifacts are missing. Uses the new harness location
for verify logic.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pinglab.experiments.mnist_dt_stability.analysis.verify import _fresh_eval
from pinglab.experiments.mnist_dt_stability.config import (
    MODELS,
    SIZES,
    TRAINING_DTS,
    calib_dir,
)


pytestmark = pytest.mark.regression

SIZE = SIZES["standard"]
THRESHOLD = 80.0


def _all_cals():
    return [(dt, m, calib_dir(m, dt, SIZE))
            for dt in TRAINING_DTS for m in MODELS]


def _has_any_calibrations():
    return any(p.exists() for _, _, p in _all_cals())


pytestmark = [pytestmark,
              pytest.mark.skipif(not _has_any_calibrations(),
                                 reason="no calibration artifacts under "
                                        "src/artifacts/mnist-dt-stability/")]


@pytest.mark.parametrize("dt,model,dir_path",
                         _all_cals(),
                         ids=lambda v: str(v))
def test_calibration_meets_threshold(dt, model, dir_path):
    if not dir_path.exists():
        pytest.skip(f"{dir_path} not trained")
    acc = _fresh_eval(dir_path)
    assert acc is not None, f"fresh_eval returned None for {dir_path}"
    assert acc >= THRESHOLD, f"dt={dt} {model}: {acc:.1f}% < {THRESHOLD:.0f}%"
