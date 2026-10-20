"""Authenticate dependent Modal runs before transfer, execution and collection."""

import json
import shutil
from collections import defaultdict

import pytest
from pingstore.contracts import PingstoreError, payload_digest, write_json_atomic
from pingstore.layout import initialize_layout
from pingstore.prune import _hidden_inputs
from pingstore.stages import source_run

from tools.modal_dispatcher import inputs as modal_inputs
from tools.modal_dispatcher import jobs as modal_jobs
from tools.modal_dispatcher.tests.test_jobs import completed_volume, no_call, submission


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
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)
    return source_run(root, identity)


def test_prepare_and_restore_complete_pinned_ancestry(tmp_path):
    root = tmp_path / "local"
    parent = fixture_source(root)
    child = fixture_source(root, "exp997-r001-compute", {"training": parent.reference})
    inputs, sources = modal_inputs.prepare_inputs(
        root, {"images": child.record["run_id"]}
    )
    assert inputs == {"images": child.reference}
    assert list(sources) == [parent.record["run_id"], child.record["run_id"]]
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
                restored, source.record["run_id"], reference=source.reference
            ).reference
            == source.reference
        )


def test_corrupt_transfer_never_becomes_visible(tmp_path):
    source = fixture_source(tmp_path / "local")
    volume = tmp_path / "volume"
    cached = volume / modal_inputs.input_path(source.reference)
    shutil.copytree(source.directory, cached)
    (cached / "export" / "data.json").write_text("corrupt")
    worker = tmp_path / "worker"
    with pytest.raises(PingstoreError):
        modal_inputs.restore_inputs(worker, volume, [source.reference])
    assert not (worker / "runs" / source.record["run_id"]).exists()


@pytest.mark.parametrize("mismatch", [False, True])
def test_collection_requires_exact_submitted_dependencies(tmp_path, mismatch):
    source = fixture_source(tmp_path)
    inputs = {"images": source.reference}
    receipt = submission(tmp_path, inputs)
    hidden = tmp_path / "runs" / f".{receipt['run_id']}.tmp"
    assert json.loads((hidden / "run.json").read_text())["inputs"] == inputs
    reasons = defaultdict(set)
    _hidden_inputs(tmp_path / "runs", {source.record["run_id"]: source.record}, reasons)
    assert any(
        reason.startswith("incomplete-input:")
        for reason in reasons[source.record["run_id"]]
    )
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
    (source.export / "data.json").write_text('{"sample": 2}')
    record = dict(source.record, payload_digest=payload_digest(source.directory))
    write_json_atomic(source.directory / "run.json", record)
    with pytest.raises(PingstoreError, match="checksum changed"):
        modal_jobs.check(
            tmp_path,
            receipt["run_id"],
            collect=True,
            volume=volume,
            call_factory=no_call,
        )
    assert not (tmp_path / "runs" / receipt["run_id"]).exists()
