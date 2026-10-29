"""Current-runset policy and the rebuildable reporting index."""

import json

import pytest
from pingstore import cli, core

from .test_prune import fixture_store, make_run, reference


def record(path):
    return json.loads((path / "run.json").read_text())


def test_recency_ties_local_preference_and_source_modes(tmp_path):
    paths = fixture_store(tmp_path)
    local = {
        path.name: record(path)
        for path in paths.values()
        if core.STAGE_ID_RE.fullmatch(path.name) and (path / "run.json").is_file()
    }
    newest = make_run(
        tmp_path,
        "exp001-r999-present",
        stage="present",
        created_at="2026-02-01T00:00:00Z",
    )
    remote = {**local, newest.name: record(newest)}
    assert (
        core.resolve_run_index(local, remote)["selections"]["exp001"]
        == paths["latest"].name
    )
    assert core.resolve_run_index({}, remote)["selections"]["exp001"] == newest.name
    assert (
        core.resolve_run_index(local, remote, source="remote")["selections"]["exp001"]
        == newest.name
    )
    assert (
        newest.name not in core.resolve_run_index(local, remote, source="local")["runs"]
    )
    tied = dict(record(newest), run_id="exp001-r998-present")
    assert (
        core.resolve_run_index({}, {newest.name: record(newest), tied["run_id"]: tied})[
            "selections"
        ]["exp001"]
        == newest.name
    )


def test_exact_cross_experiment_graph_and_visible_reasons(tmp_path):
    parent = make_run(
        tmp_path,
        "exp141-r001-compute",
        stage="compute",
        created_at="2026-01-01T00:00:00Z",
    )
    present = make_run(
        tmp_path,
        "exp142-r002-present",
        stage="present",
        created_at="2026-01-02T00:00:00Z",
        inputs={"shared": reference(parent)},
    )
    records = {path.name: record(path) for path in (parent, present)}
    result = core.resolve_run_index(
        records,
        {},
        source="local",
        reports={"exp142": {"inputs": ["exp142"]}},
        protections={parent.name: "control"},
        unfinished={parent.name: ".pending"},
    )
    assert result["runsets"]["exp142"] == sorted(records)
    row = result["runs"][parent.name]
    assert row["status"] == "Current"
    assert row["storage"] == "Local"
    assert row["reasons"] == [
        "current:exp142",
        "protection:control",
        "report:exp142",
        "unfinished:.pending",
        "unresolved-publication",
    ]
    assert result["report_selections"]["exp142"] == {"exp142": present.name}


@pytest.mark.parametrize("kind", ["missing", "changed", "cycle"])
def test_invalid_current_ancestry_is_unresolved_without_fallback(tmp_path, kind):
    paths = fixture_store(tmp_path)
    old = record(paths["old_present"])
    newest = record(paths["latest"])
    newest["inputs"] = {"source": reference(paths["old_compute"])}
    parent = record(paths["old_compute"])
    if kind == "missing":
        parent = None
    elif kind == "changed":
        newest["inputs"]["source"]["payload_digest"] = "sha256:" + "0" * 64
    else:
        parent["inputs"] = {
            "cycle": {
                "run_id": newest["run_id"],
                "payload_digest": newest["payload_digest"],
            }
        }
    local = {old["run_id"]: old, newest["run_id"]: newest}
    if parent:
        local[parent["run_id"]] = parent
    result = core.resolve_run_index(local, {})
    assert "exp001" not in result["selections"]
    assert "exp001" in result["selection_errors"]
    assert result["runs"][newest["run_id"]]["status"] == "Unresolved"


def test_incomplete_is_not_current_and_unused_broken_history_does_not_block(tmp_path):
    paths = fixture_store(tmp_path)
    records = {
        path.name: record(path)
        for path in paths.values()
        if core.STAGE_ID_RE.fullmatch(path.name) and (path / "run.json").is_file()
    }
    records[paths["old_present"].name]["inputs"]["compute"]["run_id"] = (
        "exp999-r999-compute"
    )
    incomplete = dict(
        record(paths["latest"]), run_id="exp001-r999-present", execution={}
    )
    records[incomplete["run_id"]] = incomplete
    index = core.resolve_run_index(records, {})
    assert index["selections"]["exp001"] == paths["latest"].name
    assert incomplete["run_id"] not in index["runs"]
    assert index["runs"][paths["old_present"].name]["status"] == "Unresolved"
    assert not index["selection_errors"]


def test_broken_local_copy_is_not_replaced_by_remote(tmp_path):
    paths = fixture_store(tmp_path)
    remote = {paths["latest"].name: record(paths["latest"])}
    index = core.resolve_run_index(
        {}, remote, local_errors={paths["latest"].name: "checksum mismatch"}
    )
    assert not index["selections"]
    assert "checksum mismatch" in index["selection_errors"]["exp001"]


def test_local_byte_validation_cached_and_invalidated(tmp_path, monkeypatch):
    paths = fixture_store(tmp_path)
    index = core.reporting_index(tmp_path, source="local", verify=True)
    validate = core.validate_operational_run_directory
    monkeypatch.setattr(
        core,
        "validate_operational_run_directory",
        lambda path: pytest.fail(f"rehashed {path}"),
    )
    assert core.reporting_index(tmp_path, source="local") == index
    monkeypatch.setattr(core, "validate_operational_run_directory", validate)
    (paths["latest"] / "export/result.json").write_text("{}")
    revised = core.reporting_index(tmp_path, source="local")
    assert revised["generation"] != index["generation"]
    assert "exp001" in revised["selection_errors"]
    assert revised["runs"][paths["latest"].name]["verification"]["local"] == "invalid"


def test_unchanged_layout_reuses_validation_but_new_entries_are_checked(
    tmp_path, monkeypatch
):
    paths = fixture_store(tmp_path)
    before = core.reporting_index(tmp_path, source="local")
    validate = core.validate_layout
    checked = []

    def track(directory):
        checked.append(directory.name)
        return validate(directory)

    monkeypatch.setattr(core, "validate_layout", track)
    assert core.reporting_index(tmp_path, source="local") == before
    assert checked == []
    (paths["latest"] / "export/unexpected").mkdir()
    revised = core.reporting_index(tmp_path, source="local")
    assert checked == [paths["latest"].name]
    assert "exp001" in revised["selection_errors"]


@pytest.mark.parametrize("kind", ["payload", "marker", "run", "manifest"])
def test_cached_layout_cannot_hide_symlinks_or_invalid_records(tmp_path, kind):
    paths = fixture_store(tmp_path)
    core.reporting_index(tmp_path, source="local")
    latest = paths["latest"]
    outside = tmp_path / "outside"
    if kind == "run":
        latest.rename(outside)
        latest.symlink_to(outside, target_is_directory=True)
    elif kind == "manifest":
        (latest / "run.json").write_text('{"schema":"broken"}')
    else:
        outside.write_text("not scientific evidence")
        target = latest / ("export/result.json" if kind == "payload" else ".protected")
        target.unlink(missing_ok=True)
        target.symlink_to(outside)
    index = core.reporting_index(tmp_path, source="local")
    assert "exp001" in index["selection_errors"]
    assert index["runs"][latest.name]["verification"]["local"] == "invalid"


def test_report_edits_protections_and_new_runs_change_index_generation(tmp_path):
    paths = fixture_store(tmp_path)
    before = core.reporting_index(tmp_path, source="local")
    (paths["old_present"] / ".protected").write_text("control")
    protected = core.reporting_index(tmp_path, source="local")
    assert protected["generation"] != before["generation"]
    assert protected["runs"][paths["old_present"].name]["status"] == "Required"
    make_run(
        tmp_path,
        "exp001-r999-present",
        stage="present",
        created_at="2026-02-01T00:00:00Z",
    )
    after = core.reporting_index(tmp_path, source="local")
    assert after["selections"]["exp001"] == "exp001-r999-present"
    assert after["generation"] != protected["generation"]


def test_status_json_and_human_share_index_and_do_not_claim_unused(tmp_path, capsys):
    fixture_store(tmp_path)
    args = ["status", "--root", str(tmp_path), "--source", "local"]
    assert cli.main([*args, "--json"]) == 0
    index = json.loads(capsys.readouterr().out)
    assert cli.main(args) == 0
    human = capsys.readouterr().out
    assert index["generation"] in human
    assert "Unresolved" in human
    assert "Unused" not in human


def test_remote_outage_is_explicit_and_index_is_deterministic(tmp_path):
    paths = fixture_store(tmp_path)
    local = {
        path.name: record(path)
        for path in paths.values()
        if core.STAGE_ID_RE.fullmatch(path.name) and (path / "run.json").is_file()
    }
    first = core.resolve_run_index(local, {}, remote_known=False)
    assert first == core.resolve_run_index(
        dict(reversed(list(local.items()))), {}, remote_known=False
    )
    assert all(
        "R2 inventory unavailable" in row["unresolved"]
        for row in first["runs"].values()
    )
    assert all(row["verification"]["r2"] == "unknown" for row in first["runs"].values())


def test_json_number_transport_does_not_change_generation(tmp_path):
    paths = fixture_store(tmp_path)
    value = record(paths["latest"])
    value["execution"]["configuration"] = {"duration": 1.0}
    first = core.resolve_run_index({}, {value["run_id"]: value}, source="remote")
    value["execution"]["configuration"]["duration"] = 1
    second = core.resolve_run_index({}, {value["run_id"]: value}, source="remote")
    assert first["generation"] == second["generation"]


def test_invalid_local_copy_still_counts_as_both_stores(tmp_path):
    paths = fixture_store(tmp_path)
    value = record(paths["latest"])
    index = core.resolve_run_index(
        {},
        {value["run_id"]: value},
        local_errors={value["run_id"]: "checksum mismatch"},
    )
    row = index["runs"][value["run_id"]]
    assert row["storage"] == "Both"
    assert row["verification"] == {"local": "invalid", "r2": "catalog-verified"}


def test_page_indexing_never_hashes_scientific_arrays(tmp_path, monkeypatch):
    paths = fixture_store(tmp_path)
    monkeypatch.setattr(
        core,
        "payload_digest",
        lambda path: pytest.fail("page indexing read scientific payload bytes"),
    )
    index = core.reporting_index(tmp_path, source="local")
    assert (
        index["runs"][paths["latest"].name]["verification"]["local"]
        == "metadata-validated"
    )
    assert core.reporting_index(tmp_path, source="local") == index


def test_failed_verification_stays_unresolved_until_repaired(tmp_path):
    paths = fixture_store(tmp_path)
    core.reporting_index(tmp_path, source="local")
    payload = paths["latest"] / "export/result.json"
    original = payload.read_bytes()
    payload.write_text("{}")
    invalid = core.reporting_index(tmp_path, source="local", verify=True)
    assert invalid["runs"][paths["latest"].name]["verification"]["local"] == "invalid"
    assert (
        "exp001" in core.reporting_index(tmp_path, source="local")["selection_errors"]
    )
    payload.write_bytes(original)
    repaired = core.reporting_index(tmp_path, source="local", verify=True)
    assert (
        repaired["runs"][paths["latest"].name]["verification"]["local"]
        == "byte-verified"
    )
    assert not repaired["selection_errors"]


def test_inventory_uncertainty_and_remote_operations_never_mark_history_unused(
    tmp_path,
):
    paths = fixture_store(tmp_path)
    records = {
        p.name: record(p)
        for p in paths.values()
        if core.STAGE_ID_RE.fullmatch(p.name) and (p / "run.json").is_file()
    }
    known = core.empty_publications() | {"known": True}
    inventory = dict(known=True, present=sorted(records), unresolved={}, protected=[])
    clear = core.resolve_run_index(
        {}, records, publications=known, inventory_state=inventory
    )
    assert clear["runs"][paths["old_present"].name]["status"] == "Unused"
    operation = dict(kind="prune", plan_hash="sha256:" + "a" * 64, run_ids=[])
    busy = core.resolve_run_index(
        {}, records, publications=known, inventory_state=inventory, operation=operation
    )
    assert busy["runs"][paths["old_present"].name]["status"] == "Required"
    assert any(
        "unfinished-operation:prune:" in r
        for r in busy["runs"][paths["old_present"].name]["reasons"]
    )
    uncertain = core.resolve_run_index(
        {}, records, publications=known, inventory_state=dict(inventory, known=False)
    )
    assert uncertain["runs"][paths["old_present"].name]["status"] == "Unresolved"
    assert len({clear["generation"], busy["generation"], uncertain["generation"]}) == 3
