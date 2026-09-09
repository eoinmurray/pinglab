"""Collection execution parameters and scientific recipe compatibility."""

import importlib
from pathlib import Path

import pytest
from experiments.exp022 import campaign
from experiments.exp022 import recipe as training
from experiments.exp033 import recipe as theory
from experiments.exp044 import recipe as timestep
from experiments.exp054 import recipe as coupling
from experiments.helpers.operating_point import (
    duration_steps,
    refractory_execution_configuration,
)


@pytest.mark.parametrize(
    "slug,old,new",
    [
        ("exp023", 1, 2),
        ("exp025", 1, 2),
        ("exp037", 2, 3),
        ("exp038", 1, 2),
        ("exp041", 1, 2),
        ("exp042", 4, 5),
        ("exp044", 1, 2),
        ("exp046", 1, 2),
        ("exp047", 1, 2),
        ("exp049", 1, 2),
        ("exp054", 4, 6),
        ("exp082", 1, 2),
    ],
)
def test_new_recipes_declare_collection_model_without_relabelling_old_versions(
    slug, old, new
):
    recipe = importlib.import_module(f"experiments.{slug}.recipe")
    historical = recipe.configuration(version=old)
    current = recipe.configuration()
    assert historical["schema"] == f"{slug}.recipe/v{old}"
    assert "refractory_e_ms" not in historical
    assert "refractory_i_ms" not in historical
    assert current["schema"] == f"{slug}.recipe/v{new}"
    assert current["refractory_e_ms"] == 1.2
    assert current["refractory_i_ms"] == 0.6
    assert current["refractory_policy"] == "exact"


@pytest.mark.parametrize(
    "dt,e_steps,i_steps,trial_steps",
    [
        (0.05, 24, 12, 4000),
        (0.1, 12, 6, 2000),
        (0.2, 6, 3, 1000),
        (0.3, 4, 2, 666),
        (0.6, 2, 1, 333),
    ],
)
def test_selected_conditions_have_declared_physical_timing(
    dt, e_steps, i_steps, trial_steps
):
    dynamics = refractory_execution_configuration(dt)
    assert dynamics["refractory_e_steps"] == e_steps
    assert dynamics["refractory_i_steps"] == i_steps
    assert duration_steps(200.0, dt) == trial_steps
    with pytest.raises(ValueError, match="exactly representable"):
        refractory_execution_configuration(0.25)


def test_training_commands_and_campaign_require_physical_parameters(tmp_path):
    for cell in training.CANONICAL_CELLS:
        samples, epochs = training.cell_samples_epochs(cell)
        args = training.build_train_args(cell, tmp_path / cell["name"], samples, epochs)
        resolved = campaign.resolved_parameters(cell, args, samples, epochs)
        expected = campaign._expected_config({"parameters": resolved})
        assert expected["refractory_e_ms"] == 1.2
        assert expected["refractory_i_ms"] == 0.6
        assert expected["refractory_policy"] == "exact"


def test_loaded_checkpoint_inference_has_explicit_collection_override():
    args = timestep.inference_args(
        Path("old-cell"), Path("weights.pth"), Path("new-output"), samples=100
    )
    assert args[args.index("--refractory-e-ms") + 1] == "1.2"
    assert args[args.index("--refractory-i-ms") + 1] == "0.6"
    assert args[args.index("--refractory-policy") + 1] == "exact"


def test_old_grid_and_theory_remain_readable_after_adoption(monkeypatch):
    assert timestep.configuration(version=1)["dt_sweep_ms"] == [
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
    ]
    assert timestep.configuration()["dt_sweep_ms"] == [0.05, 0.1, 0.2, 0.3, 0.6]
    old = coupling.configuration(version=4)
    monkeypatch.setitem(theory.CELL_E, "tau_ref", 1.2)
    monkeypatch.setitem(theory.CELL_I, "tau_ref", 0.6)
    assert coupling.validate(old) == old
    assert theory.configuration(version=1)["cell_E"]["tau_ref"] == 3.0
    assert theory.configuration(version=1)["cell_I"]["tau_ref"] == 1.5
    assert (
        len(
            training.cells_for_names(c["name"] for c in training.LEGACY_CANONICAL_CELLS)
        )
        == 102
    )
