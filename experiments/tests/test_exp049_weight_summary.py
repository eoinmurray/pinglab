import importlib.util
from pathlib import Path

import numpy as np


def _load_exp049():
    path = Path(__file__).resolve().parents[1] / "exp049.py"
    spec = importlib.util.spec_from_file_location("exp049", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_weight_summaries_keep_e_to_i_and_i_to_e_pruning_separate():
    exp049 = _load_exp049()
    w_ei = exp049.weight_summary(
        np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.0, 0.3])
    )
    w_ie = exp049.weight_summary(
        np.array([0.4, 0.5, 0.6]), np.array([0.4, 0.5, 0.6])
    )

    assert w_ei["trained_zero_fraction"] == 2 / 3
    assert w_ie["trained_zero_fraction"] == 0.0
    assert w_ei["trained_mean"] != w_ie["trained_mean"]
