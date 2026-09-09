"""Separate-source theory refresh, preserving spikes and historical readers."""

import copy

import pytest
from experiments.exp033 import analyse as mf_analyse
from experiments.exp033 import evidence as mf_evidence
from experiments.exp033 import inputs as mf_inputs
from experiments.exp033 import recipe as mf_recipe
from experiments.exp033.test import synthetic
from experiments.exp054 import analyse, compute, evidence, inputs, present, recipe
from experiments.exp054.test import lab as lab
from experiments.exp054.test import resign
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


@pytest.fixture(params=[5, 6])
def sources(lab, monkeypatch, request):
    root, frequency, _ = lab
    monkeypatch.setattr(mf_analyse, "REPO", root)
    original_configuration = recipe.configuration
    with monkeypatch.context() as historical:
        historical.setattr(recipe, "configuration", lambda **kwargs: original_configuration(
            **{**kwargs, "version": kwargs.get("version", request.param)}
        ))
        spikes = compute.compute()
    with mf_inputs.execution(root, "compute", sources={}) as run:
        mf_evidence.write(run.export, synthetic())
    theory = mf_analyse.analyse(run.run_id, frequency)
    monkeypatch.setattr(
        compute, "compute", lambda *a, **k: pytest.fail("simulation launched")
    )
    monkeypatch.setattr(
        mf_analyse, "analyse", lambda *a, **k: pytest.fail("theory remeasured")
    )
    original_read = evidence.read

    def read(path):
        assert path != root / ".pingstore/runs" / spikes / "export", (
            "embedded old theory consumed"
        )
        return original_read(path)

    monkeypatch.setattr(evidence, "read", read)
    return root, spikes, frequency, theory


def test_explicit_theory_refresh_stages_preserve_sources(sources):
    root, spikes, frequency, theory = sources
    spike_run = inputs.source(root, spikes, "compute")
    original = spike_run.reference
    analysis = inputs.source(
        root, analyse.analyse(spikes, frequency, theory_source=theory), "analyse"
    )
    cfg = inputs.configuration(analysis)
    assert cfg["spike_source_recipe"] == evidence.compute_contract(spike_run)
    assert cfg["spike_source_recipe"]["mean_field"]["cell_E"]["tau_ref"] == (
        3.0 if cfg["spike_source_recipe"]["schema"].endswith("v5") else 1.2
    )
    assert cfg["theory_recipe"]["cell_E"]["tau_ref"] == 1.2
    assert set(analysis.record["inputs"]) == {"compute", "frequencies", "theory"}
    data = load_json(analysis.file("results.json"))
    theory_numbers = load_json(
        inputs.source(root, theory, "analyse", experiment="exp033").file("results.json")
    )
    assert data["mean_field"]["hopf"] == theory_numbers["results"]["hopf"]
    assert data["mean_field"]["config"] == theory_numbers["config"]
    assert analysis.record["theory_refresh"]["spiking_probes_reused"] == 51
    rendered = inputs.source(
        root, present.present(analysis.record["run_id"]), "present"
    )
    assert inputs.configuration(rendered) == cfg
    assert {f.name for f in rendered.export.iterdir()} == {
        *recipe.FIGURES,
        "numbers.json",
    }
    assert {
        k: v
        for k, v in load_json(rendered.file("numbers.json")).items()
        if k != "run_id"
    } == data
    assert inputs.source(root, spikes, "compute").reference == original
    assert not (root / ".artifacts").exists()


@pytest.mark.parametrize("damage", ["old_recipe", "frequency_pin", "payload"])
def test_invalid_theory_rejected_before_reservation(sources, damage):
    root, spikes, frequency, theory = sources
    path = root / ".pingstore/runs" / theory
    record = load_json(path / "run.json")
    if damage == "old_recipe":
        record["execution"]["configuration"] = mf_recipe.configuration(version=1)
        write_json_atomic(path / "run.json", record)
    elif damage == "frequency_pin":
        record["inputs"]["frequencies"]["payload_digest"] = "sha256:" + "0" * 64
        write_json_atomic(path / "run.json", record)
    else:
        (path / "export/results.json").write_text("{}")
    before = set((root / ".pingstore/runs").iterdir())
    with pytest.raises(PingstoreError):
        analyse.analyse(spikes, frequency, theory_source=theory)
    assert set((root / ".pingstore/runs").iterdir()) == before


@pytest.mark.parametrize("damage", ["missing_pin", "numbers", "coordinates"])
def test_present_rejects_mixed_theory(sources, damage):
    root, spikes, frequency, theory = sources
    identity = analyse.analyse(spikes, frequency, theory_source=theory)
    path = root / ".pingstore/runs" / identity
    if damage == "missing_pin":
        record = load_json(path / "run.json")
        del record["inputs"]["theory"]
        write_json_atomic(path / "run.json", record)
    elif damage == "numbers":
        data = load_json(path / "export/results.json")
        data["mean_field"]["hopf"]["I_ext_star"] += 0.1
        write_json_atomic(path / "export/results.json", data)
        resign(path)
    else:
        data = evidence.read(path / "export")
        data["mean_field"]["hopf"]["I_ext_star"] += 0.1
        evidence.write(path / "export", data)
        resign(path)
    with pytest.raises(PingstoreError):
        present.analysis_source(root, identity)


def test_refresh_recipe_does_not_relabel_historical_compute():
    old = recipe.configuration(version=4)
    before = copy.deepcopy(old)
    current = recipe.refresh_configuration(old, mf_recipe.configuration())
    assert old == before and recipe.validate(old) == old
    assert recipe.validate_analysis(current) == current
    with pytest.raises(PingstoreError):
        recipe.validate(current)
    with pytest.raises(PingstoreError):
        recipe.refresh_configuration(
            recipe.configuration(version=2), mf_recipe.configuration()
        )
    with pytest.raises(PingstoreError):
        recipe.refresh_configuration(old, mf_recipe.configuration(version=1))
