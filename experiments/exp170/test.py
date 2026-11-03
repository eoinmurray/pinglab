"""Validate migration and native evidence without executing a simulation."""

from pathlib import Path

import pytest
from experiments.exp170 import recipe
from experiments.exp170.analyse import validate
from pingstore.library import file_sha256, source_run
from snnlab import lang

ROOT = Path(__file__).resolve().parents[2]


def evidence():
    identity = "exp170-r001-compute"
    if not (ROOT / ".pingstore/runs" / identity).exists():
        pytest.skip("Migration evidence is not local")
    return source_run(
        ROOT / ".pingstore", identity, experiment="exp170", stage="compute"
    )


def test_import_is_byte_identical():
    imported = evidence()
    pin = imported.record["inputs"]["retained"]
    original = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        experiment="exp169",
        stage="compute",
        phase="scratch-count-matched-inhibition",
        reference=pin,
    )

    def inventory(source):
        return {
            p.relative_to(source.outputs): file_sha256(p)
            for p in source.outputs.rglob("*")
            if p.is_file()
        }

    assert inventory(imported) == inventory(original)
    assert (
        imported.record["execution"]["configuration"]
        == original.record["execution"]["configuration"]
    )
    assert imported.record["execution"]["no_simulation"] is True


def test_recipe_matches_retained_native_graph(tmp_path):
    imported = evidence()
    bundle = lang.load_bundle(
        imported.restore_bundle("network.bundle", tmp_path / "network.bundle")
    )
    assert recipe.author_network(recipe.configuration()).graph == bundle.graph


def test_exact_source_counts_conductance_matching_and_frozen_weights():
    result = validate(evidence())
    for name in ["ping", "artificial"]:
        assert result[name]["source_rate_mean_hz"] == 120.0
        assert result[name]["source_rate_sd_hz"] == 0.0
    assert result["checks"]["sd_relative_error_max"] < 0.001
    assert result["checks"]["frozen_checkpoint_exact"]
    assert result["artificial"]["pairwise_conductance_correlation"] < 0.05
    assert result["conditions"][2]["rate_e_hz"] == pytest.approx(3.3984375)
