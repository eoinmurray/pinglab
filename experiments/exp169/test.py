"""Focused checks against retained evidence; no neural simulation is launched."""

from pathlib import Path

import pytest
from experiments.exp169 import recipe
from experiments.exp169.analyse import validate
from pingstore.library import file_sha256, source_run
from snnlab import lang

ROOT = Path(__file__).resolve().parents[2]
RETAINED = "exp169-r003-compute"


def evidence():
    if not (ROOT / ".pingstore/runs" / RETAINED).is_dir():
        pytest.skip("Retained migration evidence is not local")
    return source_run(
        ROOT / ".pingstore", RETAINED, experiment="exp169", stage="compute"
    )


def test_import_is_byte_identical_and_preserves_provenance():
    imported = evidence()
    pin = imported.record["inputs"]["retained"]
    source = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        experiment="exp022",
        stage="compute",
        reference=pin,
        phase="scratch-artificial-inhibition",
    )
    original = {
        p.relative_to(source.outputs): file_sha256(p)
        for p in source.outputs.rglob("*")
        if p.is_file()
    }
    copied = {
        p.relative_to(imported.outputs): file_sha256(p)
        for p in imported.outputs.rglob("*")
        if p.is_file()
    }
    assert original == copied
    assert (
        imported.record["execution"]["configuration"]
        == source.record["execution"]["configuration"]
    )
    assert imported.record["execution"]["no_simulation"] is True


def test_portable_recipe_preserves_native_control_architecture(tmp_path):
    imported = evidence()
    native = lang.load_bundle(
        imported.restore_bundle("network.bundle", tmp_path / "native.bundle")
    )
    assert recipe.author_network(recipe.configuration()).graph == native.graph
    reference = recipe.author_network(recipe.configuration(), include_private=False)
    assert reference.graph["populations"] == native.graph["populations"]
    assert {p["id"] for p in reference.graph["projections"]} == {
        "input_to_E",
        "E_to_I",
        "I_to_E",
        "readout_projection",
    }
    assert {i["id"] for i in reference.graph["inputs"]} == {"drive"}


def test_retained_control_matches_moments_filter_and_frozen_weights():
    results = validate(evidence())
    assert results["checks"]["frozen_checkpoint_parameters_exact"]
    assert results["checks"]["native_baseline_counts_exact"]
    assert results["checks"]["legacy_baseline_mismatched_spike_bins"] == 40
    assert results["matched"]["per_cell_std_relative_error_max"] < 0.01
    assert results["matched"]["pairwise_correlation_mean"] < 0.05
    assert results["conditions"][2]["rate_e_hz"] == pytest.approx(2.275390625)
