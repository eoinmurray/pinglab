"""Focused contract tests for the minimal exp116 design."""

from experiments.exp116 import recipe


def test_minimal_condition_set():
    cfg = recipe.configuration()
    planned = recipe.conditions(cfg)
    assert len(planned) == 14
    assert len({recipe.condition_id(row) for row in planned}) == 14
    assert sum(recipe.is_reference(row, cfg) for row in planned) == 1
    assert sum(recipe.is_primary(row, cfg) for row in planned) == 6


def test_only_reference_receives_the_ramp():
    cfg = recipe.configuration()
    ramped = [row for row in recipe.conditions(cfg) if recipe.is_reference(row, cfg)]
    assert ramped == [cfg["reference"]]


def test_robustness_is_four_endpoint_pairs():
    cfg = recipe.configuration()
    robust = [row for row in recipe.conditions(cfg) if not recipe.is_primary(row, cfg)]
    observed = {(row["sigma_mV"], row["kappa"]): set() for row in robust}
    for row in robust:
        observed[(row["sigma_mV"], row["kappa"])].add(row["tau_GABA_ms"])
    assert len(observed) == 4
    assert all(values == {4.5, 27.0} for values in observed.values())
