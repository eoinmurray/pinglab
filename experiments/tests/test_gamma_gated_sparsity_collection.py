from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.collections.gamma_gated_sparsity import execution, slurm
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
    assert positions["exp041"] < positions["exp054"]
    assert {"exp023", "exp047", "exp080", "exp081"} <= positions.keys()
    exp082 = next(
        experiment for experiment in EXPERIMENTS if experiment.slug == "exp082"
    )
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
    assert payload["blocking_issues"] == []
    assert payload["acceptance_issues"] == [47]
    assert all(row["command"] for row in rows)
    assert all(row["required_outputs"] for row in rows)


def test_runner_environment_exposes_shared_derived_root(tmp_path: Path) -> None:
    payload = build_plan(tmp_path / "campaign", "smoke")
    payload["profile"] = "smoke"
    payload["exp022_manifest"] = str(tmp_path / "campaign/exp022/campaign.json")
    row = next(row for row in execution.rows_in_order(payload) if row["slug"] == "exp054")
    environment = execution._runner_environment(payload, row)
    assert environment["PINGLAB_COLLECTION_DERIVED_ROOT"] == str(
        (tmp_path / "campaign/derived/artifacts/data").resolve()
    )
    assert environment["PINGLAB_SMOKE"] == "1"


def test_init_composes_runstore_and_exp022_manifests(
    tmp_path: Path,
    monkeypatch,
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

    monkeypatch.setattr(execution, "_run_exp022", lambda _plan, row: complete(row))

    def resume_downstream(_plan, row):
        if not execution._outputs_valid(row):
            complete(row)

    monkeypatch.setattr(execution, "_run_downstream", resume_downstream)
    execution.run_local(root)
    assert seen == [row["slug"] for row in execution.rows_in_order(plan)]
    execution.run_local(root)
    assert seen == [row["slug"] for row in execution.rows_in_order(plan)]
    assert not [
        row
        for row in execution.validate_campaign(root)["experiments"]
        if not row["outputs_valid"]
    ]


def test_finalize_delegates_to_runstore_after_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    execution.write_json_atomic(
        root / "run.json",
        {
            "run_id": "smoke",
            "status": "running",
        },
    )
    monkeypatch.setattr(execution, "validate_campaign", lambda _root: {})

    def fake_run(command, **_kwargs):
        assert command[-2:] == [str(root), "--finalize"]
        execution.write_json_atomic(
            root / "run.json",
            {
                "run_id": "smoke",
                "status": "complete",
            },
        )
        execution.write_json_atomic(
            root / "inventory.json",
            {
                "file_count": 3,
                "total_size_bytes": 42,
                "payload_digest": "a" * 64,
            },
        )
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
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "campaign"
    checkout = tmp_path / "publication"
    root.mkdir()
    checkout.mkdir()
    (root / "inventory.json").write_text("{}")
    execution.write_json_atomic(
        root / "run.json",
        {
            "run_id": "smoke",
            "status": "complete",
        },
    )
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
    assert result["promoted"] == [row["slug"] for row in execution.rows_in_order(plan)]
    assert all(call[1]["cwd"] == checkout for call in calls)
    assert all(str(checkout) in call[0] for call in calls)


def test_publication_build_rejects_stubbed_entries(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "campaign"
    checkout = tmp_path / "publication"
    root.mkdir()
    checkout.mkdir()
    (root / "inventory.json").write_text("{}")
    execution.write_json_atomic(
        root / "run.json",
        {
            "run_id": "smoke",
            "status": "complete",
        },
    )
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


def _slurm_resources(tmp_path: Path) -> dict:
    return {
        "account": "SL2-test",
        "partition": "ampere",
        "mnist_cache": str(tmp_path / "mnist"),
        "uv": "/usr/bin/uv",
        "exp022": {
            tier: {
                "time": "01:00:00",
                "cpus": 4,
                "memory_gb": 16,
                "gpus": 1,
                "concurrency": 2,
            }
            for tier in slurm.TIERS
        },
        "jobs": {
            kind: {"time": "00:30:00", "cpus": 2, "memory_gb": 8, "gpus": 0}
            for kind in ("aggregate", "downstream", "finalize")
        },
    }


def test_slurm_resources_require_every_measured_tier(tmp_path: Path) -> None:
    resources = _slurm_resources(tmp_path)
    resources["exp022"].pop("variable_rate")
    path = tmp_path / "resources.json"
    execution.write_json_atomic(path, resources)
    with pytest.raises(execution.CollectionError, match="variable_rate"):
        slurm.load_resources(path)


def test_slurm_dry_run_preserves_collection_dependencies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    plan = build_plan(root, "production-test")
    plan["profile"] = "production"
    plan["source"] = {
        "git_commit": "a" * 40,
        "git_clean": True,
        "lockfile": {"path": "uv.lock", "sha256": "b" * 64},
    }
    plan["exp022_manifest"] = str(root / "exp022" / "campaign.json")
    Path(plan["exp022_manifest"]).parent.mkdir()
    execution.write_json_atomic(
        Path(plan["exp022_manifest"]), {"manifest_sha256": "c" * 64}
    )
    resources_path = tmp_path / "resources.json"
    execution.write_json_atomic(resources_path, _slurm_resources(tmp_path))
    monkeypatch.setattr(slurm, "load_plan", lambda _root: plan)
    monkeypatch.setattr(slurm, "_exp022_cells", lambda *_args: ["cell-a"])

    payload = slurm.submit_campaign(root, resources_path)
    assert payload["source"] == plan["source"]
    assert payload["exp022_manifest_sha256"] == "c" * 64
    assert len(payload["expected_outputs"]) == len(execution.rows_in_order(plan))
    jobs = {job["name"]: job for job in payload["jobs"]}
    standard = jobs["exp022-standard"]
    assert "--cpus-per-task=4" in standard["command"]
    assert "--mem=16G" in standard["command"]
    assert "--gres=gpu:1" in standard["command"]
    aggregate = jobs["ggs-exp022-aggregate"]
    assert any(
        argument.startswith("--dependency=afterok:<standard-job-id>")
        for argument in aggregate["command"]
    )
    exp033 = jobs["ggs-exp033"]
    assert any("<ggs-exp041-job-id>" in argument for argument in exp033["command"])
    final = jobs["ggs-finalize"]
    dependency = next(
        argument for argument in final["command"] if argument.startswith("--dependency")
    )
    assert "<ggs-exp054-job-id>" in dependency
    assert "<ggs-exp042-job-id>" in dependency
    assert "<ggs-exp046-job-id>" in dependency
    final_outputs = [
        argument for argument in final["command"] if argument.startswith("--output=")
    ]
    assert str(root / "logs") not in final_outputs[0]
    assert ".scheduler-logs" in final_outputs[0]
    assert not (root / "submissions").exists()


def test_slurm_accepts_smoke_profile(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    plan = build_plan(root, "slurm-smoke")
    plan["profile"] = "smoke"
    plan["source"] = {"git_clean": True}
    plan["exp022_manifest"] = str(root / "exp022" / "campaign.json")
    Path(plan["exp022_manifest"]).parent.mkdir()
    execution.write_json_atomic(
        Path(plan["exp022_manifest"]), {"manifest_sha256": "c" * 64}
    )
    resources_path = tmp_path / "resources.json"
    execution.write_json_atomic(resources_path, _slurm_resources(tmp_path))
    monkeypatch.setattr(slurm, "load_plan", lambda _root: plan)
    monkeypatch.setattr(slurm, "_exp022_cells", lambda *_args: ["cell-a"])

    payload = slurm.submit_campaign(root, resources_path)
    assert payload["mode"] == "dry-run"
    assert len(payload["jobs"]) == 23


def test_slurm_test_only_calls_sbatch_without_submitting(monkeypatch) -> None:
    seen = []

    def fake_run(command, **_kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=0, stdout="admission accepted", stderr="")

    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    result = slurm._run(
        ["sbatch", "--parsable", "job.sbatch"],
        submit=False,
        test_only=True,
        dry_id="<dry>",
    )
    assert result == "<test-only>"
    assert seen == [["sbatch", "--test-only", "--parsable", "job.sbatch"]]


def test_production_canaries_select_one_missing_cell_per_tier(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    plan = build_plan(root, "production-canaries")
    plan["profile"] = "production"
    plan["source"] = {"git_commit": "a" * 40, "git_clean": True}
    plan["exp022_manifest"] = str(root / "exp022" / "campaign.json")
    Path(plan["exp022_manifest"]).parent.mkdir()
    execution.write_json_atomic(
        Path(plan["exp022_manifest"]), {"manifest_sha256": "c" * 64}
    )
    resources_path = tmp_path / "resources.json"
    execution.write_json_atomic(resources_path, _slurm_resources(tmp_path))
    monkeypatch.setattr(slurm, "load_plan", lambda _root: plan)
    monkeypatch.setattr(
        slurm, "_exp022_cells", lambda _manifest, tier, _uv: [f"{tier}-a", f"{tier}-b"]
    )

    payload = slurm.submit_canaries(root, resources_path)
    assert payload["purpose"] == "production-resource-canaries"
    assert [job["name"] for job in payload["jobs"]] == [
        f"exp022-canary-{tier}" for tier in slurm.TIERS
    ]
    assert [job["cells"] for job in payload["jobs"]] == [
        [f"{tier}-a"] for tier in slurm.TIERS
    ]
    assert all("--array=0-0%2" in job["command"] for job in payload["jobs"])
    assert not (root / "submissions").exists()
