from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.collections.gamma_gated_sparsity import execution
from experiments.collections.gamma_gated_sparsity.graph import (
    EXPERIMENTS,
    Experiment,
    ordered_experiments,
)
from experiments.collections.gamma_gated_sparsity.plan import REPO, build_plan


def test_graph_orders_dependencies_and_replaces_exp048_with_exp082() -> None:
    ordered = ordered_experiments()
    positions = {experiment.slug: index for index, experiment in enumerate(ordered)}
    assert "exp048" not in positions
    assert positions["exp022"] < positions["exp082"]
    assert positions["exp041"] < positions["exp033"]
    exp082 = next(experiment for experiment in EXPERIMENTS if experiment.slug == "exp082")
    assert exp082.training_run == "TR-06"


def test_graph_rejects_unknown_dependencies_and_cycles() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ordered_experiments((Experiment("a", ("missing",)),))
    with pytest.raises(ValueError, match="cycle"):
        ordered_experiments((Experiment("a", ("b",)), Experiment("b", ("a",))))


def test_plan_requires_external_absolute_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        build_plan(Path("runs/campaign"), "smoke")
    with pytest.raises(ValueError, match="external"):
        build_plan(REPO / "temp" / "campaign", "smoke")
    payload = build_plan(tmp_path / "campaign", "smoke")
    assert payload["campaign_root"] == str((tmp_path / "campaign").resolve())


def test_plan_paths_are_isolated_and_all_runners_are_integrated(tmp_path: Path) -> None:
    payload = build_plan(tmp_path / "campaign", "smoke")
    rows = [row for stage in payload["stages"] for row in stage["experiments"]]
    paths = [path for row in rows for path in row["paths"].values()]
    assert all(str(tmp_path.resolve()) in path for path in paths)
    assert payload["executable"]
    assert all(row["integrated"] or row["slug"] == "exp022" for row in rows)
    assert payload["excluded"] == ["exp048"]
    assert payload["blocking_issues"] == [69, 47]
    assert all(row["command"] for row in rows)
    assert all(row["required_outputs"] for row in rows)


def test_init_composes_runstore_and_exp022_manifests(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "campaign"
    source = {
        "git_commit": "a" * 40,
        "git_clean": True,
        "lockfile": {"path": "uv.lock", "sha256": "b" * 64},
    }
    monkeypatch.setattr(execution, "_require_clean_source", lambda: None)

    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if "tools.runstore" in command:
            (root / "exp022").mkdir(parents=True)
            (root / "downstream").mkdir()
            (root / "derived" / "artifacts").mkdir(parents=True)
            (root / "logs").mkdir()
            execution.write_json_atomic(root / "run.json", {"source": source})
        elif "experiments.exp022" in command:
            exp022 = root / "exp022"
            exp022.mkdir()
            (exp022 / "campaign.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    plan = execution.initialize_campaign(root, "smoke-test", smoke=True)
    assert plan["profile"] == "smoke"
    assert (root / "run.json").is_file()
    assert (root / "exp022" / "campaign.json").is_file()
    assert (root / execution.PLAN_NAME).is_file()
    assert any("--plumbing" in call for call in calls)


def test_local_resume_runs_in_dependency_order(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "campaign"
    plan = build_plan(root, "resume-test")
    plan["profile"] = "smoke"
    plan["source"] = {"git_clean": True}
    plan["exp022_manifest"] = str(root / "exp022" / "campaign.json")
    root.mkdir()
    (root / execution.STATUS_DIR).mkdir()
    execution.write_json_atomic(root / execution.PLAN_NAME, plan)
    monkeypatch.setattr(execution, "source_provenance", lambda: plan["source"])

    seen = []

    def complete(row):
        seen.append(row["slug"])
        for output in row["required_outputs"]:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")

    monkeypatch.setattr(
        execution, "_run_exp022", lambda _plan, row: complete(row)
    )
    def resume_downstream(_plan, row):
        if not execution._outputs_valid(row):
            complete(row)

    monkeypatch.setattr(
        execution, "_run_downstream", resume_downstream
    )
    execution.run_local(root)
    assert seen == [row["slug"] for row in execution.rows_in_order(plan)]
    execution.run_local(root)
    assert seen == [row["slug"] for row in execution.rows_in_order(plan)]
    assert not [
        row for row in execution.validate_campaign(root)["experiments"]
        if not row["outputs_valid"]
    ]


def test_finalize_delegates_to_runstore_after_validation(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    execution.write_json_atomic(root / "run.json", {
        "run_id": "smoke", "status": "running",
    })
    monkeypatch.setattr(execution, "validate_campaign", lambda _root: {})

    def fake_run(command, **_kwargs):
        assert command[-2:] == [str(root), "--finalize"]
        execution.write_json_atomic(root / "run.json", {
            "run_id": "smoke", "status": "complete",
        })
        execution.write_json_atomic(root / "inventory.json", {
            "file_count": 3,
            "total_size_bytes": 42,
            "payload_digest": "a" * 64,
        })
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    assert execution.finalize_campaign(root) == {
        "campaign_id": "smoke",
        "status": "complete",
        "file_count": 3,
        "total_size_bytes": 42,
        "payload_digest": "a" * 64,
    }


def test_publication_build_runs_promotion_from_separate_checkout(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "campaign"
    checkout = tmp_path / "publication"
    root.mkdir()
    checkout.mkdir()
    (root / "inventory.json").write_text("{}")
    execution.write_json_atomic(root / "run.json", {
        "run_id": "smoke", "status": "complete",
    })
    plan = build_plan(root, "smoke")
    plan["profile"] = "smoke"
    plan["source"] = {"git_commit": "a" * 40, "git_clean": True, "lockfile": None}
    calls = []

    monkeypatch.setattr(execution, "load_plan", lambda _root: plan)
    monkeypatch.setattr(execution, "validate_campaign", lambda _root: {})
    monkeypatch.setattr(execution, "_checkout_source", lambda _path: plan["source"])
    monkeypatch.setattr(execution.shutil, "which", lambda _name: "/usr/bin/uv")

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="built 1 entry", stderr="")

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    result = execution.build_publication(root, checkout)
    assert result["promoted"] == [
        row["slug"] for row in execution.rows_in_order(plan)
    ]
    assert all(call[1]["cwd"] == checkout for call in calls)
    assert all(str(checkout) in call[0] for call in calls)


def test_publication_build_rejects_stubbed_entries(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "campaign"
    checkout = tmp_path / "publication"
    root.mkdir()
    checkout.mkdir()
    (root / "inventory.json").write_text("{}")
    execution.write_json_atomic(root / "run.json", {
        "run_id": "smoke", "status": "complete",
    })
    plan = build_plan(root, "smoke")
    plan["source"] = {"git_commit": "a" * 40, "git_clean": True, "lockfile": None}
    monkeypatch.setattr(execution, "load_plan", lambda _root: plan)
    monkeypatch.setattr(execution, "validate_campaign", lambda _root: {})
    monkeypatch.setattr(execution, "_checkout_source", lambda _path: plan["source"])
    monkeypatch.setattr(execution.shutil, "which", lambda _name: "/usr/bin/uv")

    def fake_run(command, **_kwargs):
        output = "built 37 entries, 1 stubbed: exp022" if "demolab" in command else ""
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    with pytest.raises(execution.CollectionError, match="stubbed"):
        execution.build_publication(root, checkout)
