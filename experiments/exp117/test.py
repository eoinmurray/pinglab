"""Validate native mean-field experiment protocols and export conventions.

Tests exercise the inhibitory-decay sweep and an isolated shortened stage chain.
SNNLab owns the numerical model; no second gain, flow or Jacobian is retained.
"""

import numpy as np
import pytest
from experiments.exp117 import analyse, compute, present, recipe
from pingstore.library import load_json, source_run
from snnlab import meanfield as mf


@pytest.mark.parametrize("tau", recipe.MEAN_FIELD_CHOICES["tau_GABA_sweep_ms"])
def test_native_decay_sweep_and_export_units(tau):
    cfg = recipe.configuration()
    model = mf.substitute(recipe.mean_field_model(cfg), {"I_to_E.tau": tau})
    assert list(model.states) == [
        "E.rate",
        "I.rate",
        "E_to_I.conductance",
        "I_to_E.conductance",
    ]
    assert model.defaults["E_to_I.tau"] == 2
    assert model.defaults["I_to_E.tau"] == tau
    branch = compute.continuation(model, cfg)
    hopf = compute.refine_hopf(model, branch, cfg)
    assert 0 < hopf["I_ext_star_nA"] < 4
    assert abs(hopf["critical_pair_per_ms"][0]) < 1e-9
    assert hopf["critical_pair_per_ms"][1] > 0
    assert hopf["remaining_max_real_per_ms"] < 0
    assert hopf["crossing_slope_per_ms_per_nA"] > 0
    rows = compute.continuation_rows(branch, cfg)
    assert len(rows) == 401
    np.testing.assert_allclose(
        np.array(rows[-1]["equilibrium"][:2]) * 1000,
        [branch[-1][1].state["E.rate"], branch[-1][1].state["I.rate"]],
    )
    assert rows[-1]["equilibrium"][2:] == [
        branch[-1][1].state["E_to_I.conductance"],
        branch[-1][1].state["I_to_E.conductance"],
    ]


def test_native_compute_analyse_present_chain(tmp_path, monkeypatch):
    from pingstore import core

    monkeypatch.setattr(core, "memberships", lambda repo: {"exp117": "experiments"})
    monkeypatch.setattr(
        core,
        "capture_code",
        lambda repo, directory: {
            "git_commit": "0" * 40,
            "dirty": False,
            "code_dirty": False,
        },
    )

    cfg = recipe.configuration()
    # Isolated integration coverage: retain both drive directions and the native
    # crossing search while shortening the expensive nonlinear protocol.
    cfg["mean_field_choices"]["tau_GABA_sweep_ms"] = [6.0]
    cfg["numerical_protocol"]["drive_grid_nA"] = [0, 4, 81]
    cfg["numerical_protocol"]["criticality_ramp"].update(
        points=3, duration_ms=60, measurement_start_ms=40, measurement_samples=41
    )
    monkeypatch.setattr(recipe, "configuration", lambda: cfg)
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    computed = source_run(tmp_path / ".pingstore", compute.compute())
    saved = load_json(computed.file("mean-field-model.json"))
    assert saved["states"]["E.rate"]["unit"] == "Hz"
    assert computed.record["execution"]["environment"]["snnlab"]
    raw = load_json(computed.file("continuation.json"))
    assert raw["configuration"]["schema"] == "exp117.recipe/v4"
    with np.load(computed.file("recording.npz")) as data:
        assert data["states_up"].shape == data["states_down"].shape == (3, 4, 41)
        assert data["measurement_times_ms"][0] == 40
        assert data["measurement_times_ms"][-1] == 60
        assert np.isfinite(data["states_up"]).all()
        assert np.isfinite(data["states_down"]).all()
    analysed = source_run(
        tmp_path / ".pingstore", analyse.analyse(computed.record["run_id"])
    )
    assert all(load_json(analysed.file("results.json"))["result"]["checks"].values())
    presented = source_run(
        tmp_path / ".pingstore", present.present(analysed.record["run_id"])
    )
    assert presented.file("bifurcation_compound.svg").is_file()
    assert presented.file("numbers.json").is_file()


def test_compute_requires_current_recipe_but_readers_retain_recorded_settings():
    old = recipe.configuration()
    old["schema"] = "exp117.recipe/v3"
    old["numerical_protocol"]["equilibrium_root"] = {"method": "recorded"}
    assert recipe.validate(old, historical=True) is old
    with pytest.raises(ValueError, match="inconsistent"):
        recipe.validate(old)
