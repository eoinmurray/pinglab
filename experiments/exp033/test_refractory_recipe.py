"""Scientific recipe isolation across exp033 stages and exp054's old theory."""

import copy
import warnings

import numpy as np
import pytest
from experiments.exp033 import (
    analyse,
    compute,
    evidence,
    inputs,
    numerics,
    present,
    recipe,
)
from experiments.exp033.test import lab as lab
from experiments.exp033.test import synthetic
from experiments.exp054 import compute as coupling_compute
from experiments.exp054 import recipe as coupling_recipe
from pingstore.contracts import PingstoreError, load_json
from scipy.integrate import IntegrationWarning, quad
from scipy.special import erf


def test_stable_gain_at_strong_input_and_exact_historical_arithmetic():
    current, old = recipe.configuration(), recipe.configuration(version=1)
    mu = 0.9753635691850147
    # Independent 70-digit integration of exp(u**2) * erfc(-u).
    expected = 0.03464613847636011648376138733176796796
    with numerics.gain_parameters(current):
        assert numerics.gE(mu, 3) == pytest.approx(expected, abs=1e-14)
        rates = [numerics.gE(x, 3) for x in np.linspace(0.8, 1.2, 41)]
        assert np.all(np.diff(rates) > 0)
        assert np.isfinite(numerics.gE(-5, 3))
    cell = old["cell_E"]
    mu_v = -65 + mu / cell["g_L"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val = quad(
            lambda u: np.exp(min(u * u, 700.0)) * (1 + erf(u)),
            (-65 - mu_v) / 3,
            (-50 - mu_v) / 3,
            limit=200,
        )[0]
        historical = 1 / (cell["tau_ref"] + cell["tau_m"] * np.sqrt(np.pi) * val)
        with numerics.gain_parameters(old):
            assert numerics.gE(mu, 3) == historical


def test_gain_change_is_only_the_refractory_denominator_and_context_restores():
    old, new = recipe.configuration(version=1), recipe.configuration()
    original = numerics.gE(0.8)
    for gain, delta in [(numerics.gE, -1.8), (numerics.gI, -0.9)]:
        with numerics.gain_parameters(old):
            historical = gain(0.5)
            with numerics.gain_parameters(new):
                adopted = gain(0.5)
            assert gain(0.5) == historical
        assert 1 / adopted - 1 / historical == pytest.approx(delta, abs=1e-12)
    with pytest.raises(RuntimeError):
        with numerics.gain_parameters(old):
            raise RuntimeError("test restoration")
    assert numerics.gE(0.8) == original


def test_historical_definitions_and_exp054_execution_ignore_live_gain_defaults(
    monkeypatch,
):
    old = recipe.configuration(version=1)
    cfg = coupling_recipe.configuration(version=4)
    monkeypatch.setitem(recipe.CELL_E, "tau_ref", 99.0)
    monkeypatch.setitem(recipe.CELL_I, "tau_ref", 99.0)
    assert recipe.configuration(version=1) == old
    assert coupling_recipe.validate(cfg) == cfg
    observed = []

    def bounded(_):
        observed.extend([numerics.gE(0.8), numerics.gI(0.8)])
        return "bounded"

    monkeypatch.setattr(coupling_compute, "_mean_field", bounded)
    assert coupling_compute.mean_field(cfg) == "bounded"
    with numerics.gain_parameters(old):
        assert observed == [
            numerics.lif_fi(0.8, old["cell_E"]),
            numerics.lif_fi(0.8, old["cell_I"]),
        ]
    damaged = copy.deepcopy(old)
    damaged["cell_E"]["tau_ref"] = 1.2
    with pytest.raises(PingstoreError, match="recipe"):
        recipe.validate(damaged)


def test_historical_native_stages_keep_recorded_gain_and_new_compute_binds_its_recipe(
    lab, monkeypatch
):
    root, frequency, _, _ = lab
    old = recipe.configuration(version=1)
    with inputs.execution(root, "compute", sources={}, configuration=old) as run:
        evidence.write(run.export, synthetic(version=1))
    analysis = inputs.source(root, analyse.analyse(run.run_id, frequency), "analyse")
    rendered = inputs.source(
        root, present.present(analysis.record["run_id"]), "present"
    )
    for stage in [analysis, rendered]:
        assert inputs.configuration(stage) == old
    assert (
        load_json(rendered.file("numbers.json"))["config"]["cell_E"]["tau_ref"] == 3.0
    )
    monkeypatch.setitem(recipe.CELL_E, "tau_ref", 99.0)
    expected = numerics.lif_fi(0.8, recipe.configuration()["cell_E"])

    def simulate():
        assert numerics.gE(0.8) == expected
        return synthetic()

    monkeypatch.setattr(compute, "simulate", simulate)
    current = inputs.source(root, compute.compute(), "compute")
    assert inputs.configuration(current) == recipe.configuration()


def test_new_coupling_compute_binds_adopted_theory_and_keeps_v5(monkeypatch):
    historical = coupling_recipe.configuration(version=5)
    current = coupling_recipe.configuration()
    assert historical["mean_field"]["cell_E"]["tau_ref"] == 3.0
    assert historical["mean_field"]["cell_I"]["tau_ref"] == 1.5
    assert "gain_integral" not in historical["mean_field"]
    assert current["mean_field"]["cell_E"]["tau_ref"] == 1.2
    assert current["mean_field"]["cell_I"]["tau_ref"] == 0.6
    expected = {}
    for name, version in (("historical", 1), ("current", 2)):
        with numerics.gain_parameters(recipe.configuration(version=version)):
            expected[name] = numerics.gE(0.9753635691850147, 3)
    monkeypatch.setattr(coupling_compute, "_mean_field", lambda _: numerics.gE(0.9753635691850147, 3))
    assert coupling_compute.mean_field(current) == expected["current"]
    assert coupling_compute.mean_field(historical) == expected["historical"]
    assert expected["current"] == pytest.approx(0.034646138476360116, abs=1e-14)
