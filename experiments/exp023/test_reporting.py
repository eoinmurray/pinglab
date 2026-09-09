"""Metadata-only correction preserves scientific results and source identities."""

from types import SimpleNamespace

import numpy as np
import pytest
from experiments.exp023 import inputs, present, recipe, reporting
from pingstore import stages
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


@pytest.fixture
def retained(tmp_path, monkeypatch):
    monkeypatch.setattr(present, "REPO", tmp_path)
    monkeypatch.setattr(stages, "memberships", lambda _: {"exp023": "test"})
    monkeypatch.setattr(
        stages, "_capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    cfg = recipe.configuration(version=1)
    with stages.stage_run(tmp_path, "exp023", "compute", configuration=cfg) as run:
        np.savez_compressed(run.export / "recording.npz", spikes=np.zeros((2, 2)))
    compute = inputs.source(tmp_path, run.run_id, "compute")
    measurement = {"schema": "exp023.measurement/v1"}
    numbers = {
        "schema": "exp023.analysis/v1",
        "config": cfg,
        "measurement": measurement,
        "raster": {"ping": {"e_rate_hz": 8.289}},
        "f_gamma_hz": {"ping": 55.971},
        "fi_curves": {"ping": {"e": [3.0, 4.0]}},
    }
    with stages.stage_run(
        tmp_path,
        "exp023",
        "analyse",
        inputs={"compute": compute},
        configuration=measurement,
    ) as run:
        write_json_atomic(run.export / "results.json", numbers)
    analysis = inputs.source(tmp_path, run.run_id, "analyse")
    with stages.stage_run(
        tmp_path,
        "exp023",
        "present",
        inputs={"analysis": analysis, "compute": compute},
        configuration=cfg,
    ) as run:
        write_json_atomic(
            run.export / "numbers.json",
            {**numbers, "run_id": run.run_id, "duration_s": 1, "git_sha": "fixture"},
        )
        (run.export / "trace.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><path d="M 0 0 L 1 1"/></svg>'
        )
    presentation = inputs.source(tmp_path, run.run_id, "present")
    monkeypatch.setattr(reporting, "AUDITED_COMPUTE", compute.reference)
    monkeypatch.setattr(reporting, "AUDITED_BASE", "fixture")
    monkeypatch.setattr(
        present.plots,
        "plot_architecture",
        lambda *a: pytest.fail("metadata correction drew figures"),
    )
    return tmp_path, compute, analysis, presentation


def test_metadata_only_presentation_keeps_all_measurements_and_figure_bytes(retained):
    root, compute, analysis, source = retained
    refs = [r.reference for r in (compute, analysis, source)]
    output = inputs.source(
        root,
        present.present(
            analysis.record["run_id"], metadata_source=source.record["run_id"]
        ),
        "present",
    )
    before = load_json(source.file("numbers.json"))
    after = load_json(output.file("numbers.json"))
    assert after["config"]["schema"] == "exp023.reported-configuration/v1"
    assert after["config"]["biophysics"]["refractory_E_ms"] == 1.2
    assert after["config"]["biophysics"]["refractory_I_ms"] == 0.6
    for key in ("raster", "f_gamma_hz", "fi_curves", "measurement"):
        assert after[key] == before[key]
    assert (
        output.file("trace.svg").read_bytes() == source.file("trace.svg").read_bytes()
    )
    assert output.record["execution"]["operation"] == "metadata-correction"
    assert output.record["metadata_correction"]["measurements_changed"] is False
    assert output.record["inputs"] == {
        "presentation": source.reference,
        "analysis": analysis.reference,
        "compute": compute.reference,
    }
    for r, ref in zip((compute, analysis, source), refs):
        r.check_unchanged()
        assert r.reference == ref
    assert inputs.configuration(compute)["biophysics"]["refractory_E_ms"] == 3.0


def test_correction_refuses_unaudited_source_before_reserving(retained, monkeypatch):
    root, _, analysis, source = retained
    monkeypatch.setattr(
        reporting,
        "AUDITED_COMPUTE",
        {"run_id": "different", "payload_digest": "sha256:" + "0" * 64},
    )
    before = set((root / ".pingstore/runs").iterdir())
    with pytest.raises(PingstoreError, match="audited"):
        present.present(
            analysis.record["run_id"], metadata_source=source.record["run_id"]
        )
    assert set((root / ".pingstore/runs").iterdir()) == before


def test_correction_rejects_changed_recipe_or_producer(retained):
    _, compute, _, _ = retained
    cfg = inputs.configuration(compute)
    cfg["biophysics"]["tau_gaba_ms"] = 9.0
    with pytest.raises(PingstoreError, match="audited"):
        reporting.corrected_configuration(compute, cfg)
    other = SimpleNamespace(
        reference=compute.reference, record={"provenance": {"git_commit": "other"}}
    )
    with pytest.raises(PingstoreError, match="audited"):
        reporting.corrected_configuration(other, recipe.configuration(version=1))
