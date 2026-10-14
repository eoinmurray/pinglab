"""Assembled readers reject incomplete coverage and phase-only inputs."""

from types import SimpleNamespace

import pytest
from experiments.exp126.compute_parts.assembly import assembled_conditions
from experiments.exp126.recipe import assembled_configuration, condition_role


def test_missing_paired_phase_coverage_is_rejected(tmp_path):
    cfg = assembled_configuration([42])
    role = condition_role("snntorch", 42)
    full = SimpleNamespace(
        record={
            "scope": "full",
            "execution": {"operation": "assemble", "configuration": cfg},
            "inputs": {
                role: {
                    "run_id": "exp126-r001-compute-phase-snntorch",
                    "payload_digest": "sha256:" + "0" * 64,
                }
            },
            "assembly": {"output_mapping": {}},
        },
        export=tmp_path,
    )
    with pytest.raises(ValueError, match="coverage or output mapping"):
        assembled_conditions(tmp_path, full)


def test_phase_is_not_an_assembled_full_result(tmp_path):
    phase = SimpleNamespace(
        record={
            "scope": "phase",
            "execution": {"configuration": {"schema": "exp126.matched/v1"}},
        }
    )
    with pytest.raises(ValueError, match="assembled full"):
        assembled_conditions(tmp_path, phase)


def test_assembly_copies_bytes_and_preserves_immutable_phase_inputs(
    tmp_path, monkeypatch
):
    from experiments.exp126.compute_parts import assembly
    from experiments.exp126.recipe import PHASE_EXPORT_FILES, comparison_configuration
    from pingstore import stages

    monkeypatch.setattr(stages, "memberships", lambda _: {"exp126": "demo"})
    monkeypatch.setattr(
        stages, "_capture_code", lambda *_: {"git_commit": "test-source"}
    )
    phases = {}
    original_bytes = {}
    for backend in ("snntorch", "snnlab"):
        with stages.stage_run(
            tmp_path,
            "exp126",
            "compute",
            phase=backend,
            configuration=comparison_configuration(),
        ) as phase:
            for name in PHASE_EXPORT_FILES:
                (phase.export / name).write_bytes(f"{backend}:{name}".encode())
        role = condition_role(backend, 42)
        source = stages.source_run(tmp_path / ".pingstore", phase.run_id, phase=backend)
        phases[role] = source
        original_bytes[role] = {
            name: source.file(name).read_bytes() for name in PHASE_EXPORT_FILES
        }
    monkeypatch.setattr(
        assembly, "select_phases", lambda *_: (phases, assembled_configuration([42]))
    )
    # This case isolates filesystem/lineage semantics from scientific validators.
    monkeypatch.setattr(assembly, "validate_condition", lambda _: None)
    monkeypatch.setattr(
        assembly, "implementation_source_digest", lambda _: "fixture-source"
    )
    identity = assembly.assemble(
        tmp_path, [source.record["run_id"] for source in phases.values()]
    )
    full = stages.source_run(tmp_path / ".pingstore", identity, stage="compute")
    assert full.record["scope"] == "full"
    assert full.record["execution"]["operation"] == "assemble"
    assert full.record["inputs"] == {
        role: source.reference for role, source in phases.items()
    }
    for role, source in phases.items():
        for name, expected in original_bytes[role].items():
            assert (full.unit(role) / name).read_bytes() == expected
            assert source.file(name).read_bytes() == expected
        source.check_unchanged()
