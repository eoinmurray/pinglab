"""Authenticate dependent Modal runs before transfer, execution and collection."""

import shutil

import pytest
from pingstore.library import (
    PingstoreError,
    initialize_layout,
    source_run,
    write_json_atomic,
)

from libraries.modal_dispatcher import inputs as modal_inputs
from libraries.modal_dispatcher import jobs as modal_jobs
from libraries.modal_dispatcher.tests.test_jobs import (
    completed_volume,
    no_call,
    submission,
)


def fixture_source(root, identity="exp998-r001-compute", inputs=None):
    directory = root / "runs" / identity
    initialize_layout(directory, identity.split("-")[0])
    (directory / "export" / "data.json").write_text('{"sample": 1}')
    record = dict(
        schema="pingstore.run/v4",
        run_id=identity,
        experiment=identity.split("-")[0],
        stage="compute",
        scope="full",
        collection="fixture",
        origin="local",
        inputs=inputs or {},
        created_at="2026-10-16T12:00:00+00:00",
        execution={},
        provenance={},
    )
    from pingstore.tests.fixtures import write_fixture_json

    write_fixture_json(directory / "details.json", record)
    return source_run(root, identity)


@pytest.mark.parametrize("include_ancestry", [True, False])
def test_prepare_and_restore_complete_pinned_ancestry(tmp_path, include_ancestry):
    root = tmp_path / "local"
    parent = fixture_source(root)
    child = fixture_source(root, "exp997-r001-compute", {"training": parent.reference})
    inputs, sources = modal_inputs.prepare_inputs(
        root, {"images": child.details["run_id"]}, include_ancestry=include_ancestry
    )
    assert inputs == {"images": child.reference}
    assert list(sources) == (
        [parent.details["run_id"], child.details["run_id"]]
        if include_ancestry
        else [child.details["run_id"]]
    )
    volume = tmp_path / "volume"
    for source in sources.values():
        shutil.copytree(
            source.directory, volume / modal_inputs.input_path(source.reference)
        )
    restored = tmp_path / "worker"
    modal_inputs.restore_inputs(
        restored, volume, [s.reference for s in sources.values()]
    )
    for source in sources.values():
        assert (
            source_run(
                restored, source.details["run_id"], reference=source.reference
            ).reference
            == source.reference
        )


def test_corrupt_transfer_never_becomes_visible(tmp_path):
    source = fixture_source(tmp_path / "local")
    volume = tmp_path / "volume"
    cached = volume / modal_inputs.input_path(source.reference)
    shutil.copytree(source.directory, cached)
    (cached / "run.json").write_text('{"id":"different"}')
    worker = tmp_path / "worker"
    with pytest.raises(PingstoreError):
        modal_inputs.restore_inputs(worker, volume, [source.reference])
    assert not (worker / "runs" / source.details["run_id"]).exists()


@pytest.mark.parametrize("mismatch", [False, True])
def test_collection_requires_exact_submitted_dependencies(tmp_path, mismatch):
    source = fixture_source(tmp_path)
    inputs = {"images": source.reference}
    receipt = submission(tmp_path, inputs)
    volume = completed_volume(tmp_path, receipt, {} if mismatch else inputs)
    if mismatch:
        with pytest.raises(PingstoreError, match="does not match submitted"):
            modal_jobs.check(
                tmp_path,
                receipt["run_id"],
                collect=True,
                volume=volume,
                call_factory=no_call,
            )
        assert not (tmp_path / "runs" / receipt["run_id"]).exists()
    else:
        assert (
            modal_jobs.check(
                tmp_path,
                receipt["run_id"],
                collect=True,
                volume=volume,
                call_factory=no_call,
            )["status"]
            == "collected"
        )


def test_local_input_change_is_rejected_before_collection(tmp_path):
    source = fixture_source(tmp_path)
    inputs = {"images": source.reference}
    receipt = submission(tmp_path, inputs)
    volume = completed_volume(tmp_path, receipt, inputs)
    write_json_atomic(source.directory / "run.json", {"id": "different"})
    with pytest.raises(PingstoreError, match="run requires"):
        modal_jobs.check(
            tmp_path,
            receipt["run_id"],
            collect=True,
            volume=volume,
            call_factory=no_call,
        )
    assert not (tmp_path / "runs" / receipt["run_id"]).exists()
