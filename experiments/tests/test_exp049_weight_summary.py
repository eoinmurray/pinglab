import numpy as np
from experiments import exp049


def test_weight_summaries_keep_e_to_i_and_i_to_e_pruning_separate():
    w_ei = exp049.weight_summary(np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.0, 0.3]))
    w_ie = exp049.weight_summary(np.array([0.4, 0.5, 0.6]), np.array([0.4, 0.5, 0.6]))

    assert w_ei["trained_zero_fraction"] == 2 / 3
    assert w_ie["trained_zero_fraction"] == 0.0
    assert w_ei["trained_mean"] != w_ie["trained_mean"]
