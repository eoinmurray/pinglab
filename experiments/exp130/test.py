"""Scientific protocol and execution-continuity checks.

Run with pytest from the repository root. Small synthetic inputs check the
scientific invariants and native runtime integration without replacing the
report's pinned historical evidence.
"""

import numpy as np
from experiments.exp130 import dataset, recipe
from experiments.exp130.compute import simulate


def test_envelope_bounds_mean_and_burn_in():
    cfg = recipe.condition_configuration(recipe.configuration(), "theta")
    assert cfg["t_ms"] == 1000.0 and cfg["burn_in_ms"] == 0.0
    assert cfg["modulation_onset_ms"] == 0.0
    times = np.arange(round(cfg["t_ms"] / cfg["dt_ms"])) * cfg["dt_ms"]
    e, i = dataset.source_rates(times, cfg)
    np.testing.assert_array_equal(e, i)
    np.testing.assert_array_equal(e[times < cfg["burn_in_ms"]], 2.0)
    visible = e[times >= cfg["burn_in_ms"]]
    assert visible.min() >= 0.2 - 1e-14
    assert visible.max() <= 3.8 + 1e-14
    assert abs(visible.mean() - 2.0) < 1e-14
    for offset, expected in ((0, 2), (50, 3.8), (150, 0.2)):
        value = dataset.source_rates(np.array([cfg["burn_in_ms"] + offset]), cfg)[0][0]
        assert abs(value - expected) < 1e-14
    constant = recipe.condition_configuration(cfg, "constant")
    np.testing.assert_array_equal(
        dataset.source_rates(times, constant)[0], np.full(len(times), 2.0)
    )


def test_poisson_counts_preserve_multiplicity_and_private_channels():
    cfg = recipe.condition_configuration(recipe.configuration(), "constant")
    cfg.update(t_ms=100, n_e=100, n_i=25)
    counts = dataset.afferent_counts(cfg)
    expected = 2.0 * 400 * cfg["dt_ms"] / 1000
    assert counts["private_e"].max() > 1
    assert abs(counts["private_e"].mean() - expected) < 5 * np.sqrt(
        expected / counts["private_e"].size
    )
    assert not np.array_equal(counts["private_e"][:, 0], counts["private_e"][:, 1])
    np.testing.assert_array_equal(
        counts["private_e"], dataset.afferent_counts(cfg)["private_e"]
    )


def test_chunked_execution_preserves_delays_and_state(tmp_path):
    cfg = recipe.condition_configuration(recipe.configuration(), "theta")
    cfg.update(t_ms=50, burn_in_ms=0, modulation_onset_ms=0, n_e=40, n_i=10)
    _, chunked, weights = simulate(cfg, chunk_steps=73, artifacts=tmp_path / "chunked")
    _, whole, other = simulate(cfg, chunk_steps=500, artifacts=tmp_path / "whole")
    for name, value in chunked.items():
        np.testing.assert_array_equal(value, whole[name], err_msg=name)
    for name, value in weights.items():
        np.testing.assert_array_equal(value, other[name], err_msg=name)


def test_physical_weights_and_absent_autapses():
    from experiments.exp130.compute import build_model

    cfg = recipe.configuration()
    cfg.update(n_e=40, n_i=10)
    bundle, model = build_model(cfg)
    parameters = model.parameter_map()
    for projection in bundle.graph["projections"]:
        if projection["connection"] == "recurrent":
            assert not projection["allow_autapses"]
    for pop in ("e", "i"):
        expected = np.eye(cfg[f"n_{pop}"]) * cfg["external_weight_us"]
        np.testing.assert_allclose(
            parameters[f"private_{pop}_to_{pop.upper()}.weight"].detach().numpy(),
            expected,
        )
        for target in ("e", "i"):
            matrix = (
                parameters[f"{pop.upper()}_to_{target.upper()}.weight"].detach().numpy()
            )
            if pop == target:
                np.testing.assert_array_equal(
                    np.diag(matrix), np.zeros(cfg[f"n_{pop}"])
                )
            surviving = matrix[matrix > 0]
            assert len(surviving)
            np.testing.assert_allclose(surviving, cfg[f"recurrent_{pop}_weight_us"])
