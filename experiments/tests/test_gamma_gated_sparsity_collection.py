from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments import (
    exp022,
    exp024,
    exp025,
    exp037,
    exp038,
    exp041,
    exp042,
    exp044,
    exp046,
    exp049,
    exp080,
    exp082,
)
from experiments.collections.gamma_gated_sparsity import execution, slurm, workloads
from experiments.collections.gamma_gated_sparsity.graph import (
    EXPERIMENTS,
    Experiment,
    ordered_experiments,
)
from experiments.collections.gamma_gated_sparsity.plan import REPO, build_plan


def test_collection_production_training_horizon_is_50_epochs() -> None:
    assert exp022.EPOCHS_STANDARD == 50
    assert exp080.EPOCHS_STANDARD == 50


def test_downstream_cell_banks_resolve_through_exp022_registry() -> None:
    registered = {
        run_id: {cell["name"] for cell in exp022.training_run_cells(run_id)}
        for run_id in ("TR-02", "TR-03", "TR-04", "TR-05", "TR-06", "TR-07")
    }

    for module, seeds in (
        (exp025, exp025.SEEDS),
        (exp037, exp037.SEEDS_BASELINE),
        (exp038, exp038.SEEDS_BASELINE),
    ):
        assert {
            module.cell_dir(model, target, seed).name
            for model in module.MODELS
            for target in module.RATE_TARGET_GRID_HZ
            for seed in seeds
        } == registered["TR-02"]
    assert {
        exp024.cell_dir(model, seed).name
        for model in exp024.MODELS
        for seed in exp024.SEEDS
    } <= registered["TR-02"]

    assert {
        exp041.cell_dir(tau, seed).name
        for tau in exp041.TAU_GABA_SWEEP
        for seed in exp041.SEEDS
    } == registered["TR-03"]
    assert {
        exp046.exp041_cell_dir(tau, seed).name
        for tau in exp046.TAU_GABA_SWEEP_MS
        for seed in exp046.SEEDS
    } == registered["TR-03"]
    assert {
        exp044.cell_dir(dt, seed).name
        for dt in exp044.DT_SWEEP_MS
        for seed in exp044.SEEDS
    } == registered["TR-04"]
    assert {
        exp049.cell_dir(condition, seed).name
        for condition in exp049.COND_ORDER
        for seed in exp049.SEEDS
    } == registered["TR-05"]
    assert {exp082.training_dir(seed).name for seed in exp082.SEEDS} == registered[
        "TR-06"
    ]
    assert {
        exp025.low_w_in_cell_dir(w_in, seed).name
        for w_in in exp025.LOW_W_IN_VALUES
        for seed in exp025.LOW_W_IN_SEEDS
    } <= registered["TR-07"]

    exp042_sources = exp042.checkpoint_source_dirs()
    assert {path.name for path in exp042_sources["exp022_tr02"]} <= registered["TR-02"]
    assert {path.name for path in exp042_sources["exp041_tr03"]} == registered["TR-03"]


def test_graph_orders_dependencies_and_replaces_exp048_with_exp082() -> None:
    ordered = ordered_experiments()
    positions = {experiment.slug: index for index, experiment in enumerate(ordered)}
    assert "exp048" not in positions
    assert positions["exp022"] < positions["exp082"]
    assert positions["exp041"] < positions["exp033"]
    assert positions["exp041"] < positions["exp054"]
    assert positions["exp022"] < positions["exp042"]
    assert positions["exp041"] < positions["exp042"]
    assert {"exp023", "exp047", "exp080", "exp081"} <= positions.keys()
    exp082 = next(
        experiment for experiment in EXPERIMENTS if experiment.slug == "exp082"
    )
    assert exp082.training_run == "TR-06"
    exp042_node = next(
        experiment for experiment in EXPERIMENTS if experiment.slug == "exp042"
    )
    assert exp042_node.dependencies == ("exp022", "exp041")
    assert exp042_node.training_run == "TR-02"


def test_exp042_declares_checkpoint_sources_by_owner_and_training_run() -> None:
    sources = exp042.checkpoint_source_dirs()
    assert set(sources) == {"exp022_tr02", "exp041_tr03"}
    assert [path.name for path in sources["exp022_tr02"]] == [
        "ping__off__seed42",
        "ping__off__seed43",
        "ping__off__seed44",
    ]
    assert len(sources["exp041_tr03"]) == 18
    assert all(path.name.startswith("ping__tg") for path in sources["exp041_tr03"])


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
    assert payload["acceptance_issues"] == []
    assert all(row["command"] for row in rows)
    assert all(row["required_outputs"] for row in rows)


def test_runner_environment_exposes_shared_derived_root(tmp_path: Path) -> None:
    payload = build_plan(tmp_path / "campaign", "smoke")
    payload["profile"] = "smoke"
    payload["exp022_manifest"] = str(tmp_path / "campaign/exp022/campaign.json")
    row = next(
        row for row in execution.rows_in_order(payload) if row["slug"] == "exp054"
    )
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
    plan["source"] = {
        "git_commit": "a" * 40,
        "git_clean": True,
        "lockfile": {"path": "uv.lock", "sha256": "b" * 64},
    }
    plan["exp022_manifest"] = str(root / "exp022" / "campaign.json")
    root.mkdir()
    Path(plan["exp022_manifest"]).parent.mkdir()
    execution.write_json_atomic(
        Path(plan["exp022_manifest"]), {"manifest_sha256": "c" * 64}
    )
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
        execution._stamp_collection_provenance(plan, row)

    monkeypatch.setattr(execution, "_run_exp022", lambda _plan, row: complete(row))

    def resume_downstream(_plan, row):
        if not execution._outputs_valid_for_plan(plan, row):
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


def test_collection_provenance_rejects_cross_campaign_output(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    plan = build_plan(root, "campaign-a")
    plan["source"] = {
        "git_commit": "a" * 40,
        "git_clean": True,
        "lockfile": {"path": "uv.lock", "sha256": "b" * 64},
    }
    plan["exp022_manifest"] = str(root / "exp022" / "campaign.json")
    Path(plan["exp022_manifest"]).parent.mkdir(parents=True)
    execution.write_json_atomic(
        Path(plan["exp022_manifest"]), {"manifest_sha256": "c" * 64}
    )
    row = execution.rows_in_order(plan)[0]
    output = Path(row["required_outputs"][0])
    output.parent.mkdir(parents=True)
    execution.write_json_atomic(
        output, {"collection_provenance": {"campaign_id": "campaign-b"}}
    )

    assert not execution._outputs_valid_for_plan(plan, row)
    with pytest.raises(execution.CollectionError, match="different campaign"):
        execution._stamp_collection_provenance(plan, row)


def test_compose_campaign_replaces_selected_outputs_and_records_sources(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "base"
    overlay = tmp_path / "overlay"
    destination = tmp_path / "composite"
    base.mkdir()
    overlay.mkdir()
    base_plan = build_plan(base, "base-run")
    execution.write_json_atomic(base / execution.PLAN_NAME, base_plan)
    execution.write_json_atomic(
        base / "run.json",
        {
            "run_id": "base-run",
            "status": "complete",
            "source": {"git_commit": "a" * 40},
        },
    )
    execution.write_json_atomic(
        base / "inventory.json",
        {"run_id": "base-run", "payload_digest": "b" * 64},
    )
    execution.write_json_atomic(
        overlay / "run.json",
        {
            "run_id": "repair-run",
            "status": "planned",
            "source": {"git_commit": "c" * 40},
        },
    )
    slugs = [row["slug"] for row in execution.rows_in_order(base_plan)]
    for source_root, run_id, selected in (
        (base, "base-run", slugs),
        (overlay, "repair-run", ["exp022", "exp025"]),
    ):
        for slug in selected:
            derived = source_root / "derived/artifacts/data" / slug
            derived.mkdir(parents=True)
            execution.write_json_atomic(
                derived / "numbers.json",
                {
                    "marker": run_id,
                    "collection_provenance": {
                        "campaign_id": run_id,
                        "source_git_commit": ("a" if run_id == "base-run" else "c")
                        * 40,
                    },
                },
            )
            (derived / "figure.svg").write_text("<svg/>\n")

    monkeypatch.setattr(execution, "_require_clean_source", lambda: None)
    monkeypatch.setattr(
        execution,
        "_inspect_runstore",
        lambda root: (
            {"inventory": "valid"} if root == base else {"inventory": "absent"}
        ),
    )

    def fake_run(command, **_kwargs):
        assert "tools.runstore" in command and "init" in command
        (destination / "exp022").mkdir(parents=True)
        (destination / "downstream").mkdir()
        (destination / "derived/artifacts").mkdir(parents=True)
        (destination / "logs").mkdir()
        execution.write_json_atomic(
            destination / "run.json",
            {
                "run_id": "composite-run",
                "source": {"git_commit": "d" * 40},
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    result = execution.compose_campaign(
        destination,
        "composite-run",
        base_root=base,
        overlay_root=overlay,
        replacements=["exp022", "exp025"],
    )

    assert result["experiments"] == len(slugs)
    assert (
        execution.load_json(destination / "derived/artifacts/data/exp022/numbers.json")[
            "marker"
        ]
        == "repair-run"
    )
    assert (
        execution.load_json(destination / "derived/artifacts/data/exp037/numbers.json")[
            "marker"
        ]
        == "base-run"
    )
    composition = execution.load_json(destination / "composition.json")
    assert composition["experiments"]["exp025"]["run_id"] == "repair-run"
    assert composition["experiments"]["exp037"]["run_id"] == "base-run"


def test_integrate_repair_preserves_base_and_records_repaired_source(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "campaign"
    repair_root = tmp_path / "repair"
    root.mkdir()
    (root / execution.STATUS_DIR).mkdir()
    plan = build_plan(root, "campaign-a")
    base_source = {
        "git_commit": "a" * 40,
        "git_clean": True,
        "lockfile": {"path": "uv.lock", "sha256": "b" * 64},
    }
    integration_source = {
        "git_commit": "e" * 40,
        "git_clean": True,
        "lockfile": {"path": "uv.lock", "sha256": "b" * 64},
    }
    expected_repair_source = {
        "git_commit": "d" * 40,
        "git_clean": True,
        "lockfile": {
            "path": "uv.lock",
            "sha256": execution.hashlib.sha256(b"repair lock").hexdigest(),
        },
    }
    plan["source"] = base_source
    plan["profile"] = "production"
    plan["exp022_manifest"] = str(root / "exp022/campaign.json")
    Path(plan["exp022_manifest"]).parent.mkdir()
    execution.write_json_atomic(
        Path(plan["exp022_manifest"]), {"manifest_sha256": "c" * 64}
    )
    execution.write_json_atomic(root / execution.PLAN_NAME, plan)
    execution.write_json_atomic(
        root / "run.json",
        {
            "run_id": "campaign-a",
            "status": "running",
            "upstream": [],
            "provenance_notes": "publication campaign",
        },
    )

    row = next(row for row in execution.rows_in_order(plan) if row["slug"] == "exp082")
    source_dir = repair_root / "derived/artifacts/data/exp082"
    source_dir.mkdir(parents=True)
    for required in row["required_outputs"]:
        name = Path(required).name
        (source_dir / name).write_text("{}\n")
    execution.write_json_atomic(
        repair_root / "repair-run.json",
        {
            "experiment": "exp082",
            "source_git_commit": expected_repair_source["git_commit"],
            "base_campaign_root": str(root),
            "base_campaign_source_git_commit": base_source["git_commit"],
            "exp022_manifest_file_sha256": execution._sha256(
                Path(plan["exp022_manifest"])
            ),
        },
    )
    monkeypatch.setattr(execution, "source_provenance", lambda: integration_source)

    def fake_git(command, **_kwargs):
        if "merge-base" in command:
            return SimpleNamespace(returncode=0)
        assert command[1:3] == [
            "show",
            f"{expected_repair_source['git_commit']}:uv.lock",
        ]
        return SimpleNamespace(returncode=0, stdout=b"repair lock")

    monkeypatch.setattr(execution.subprocess, "run", fake_git)

    result = execution.integrate_repair(root, repair_root, "exp082")

    updated = execution.load_json(root / execution.PLAN_NAME)
    numbers = execution.load_json(Path(row["required_outputs"][0]))
    run = execution.load_json(root / "run.json")
    assert result["source_git_commit"] == expected_repair_source["git_commit"]
    assert result["integration_source_git_commit"] == integration_source["git_commit"]
    assert updated["source"] == base_source
    assert updated["repairs"]["exp082"]["source"] == expected_repair_source
    assert updated["repairs"]["exp082"]["integration_source"] == integration_source
    assert (
        numbers["collection_provenance"]["source_git_commit"]
        == expected_repair_source["git_commit"]
    )
    assert (
        numbers["collection_provenance"]["repair"]["base_source_git_commit"]
        == base_source["git_commit"]
    )
    assert run["upstream"] == [
        f"exp082-repair:{expected_repair_source['git_commit']}:{repair_root.resolve()}"
    ]
    assert (
        execution.load_json(root / execution.STATUS_DIR / "exp082.json")["state"]
        == "complete"
    )


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
    repair_source = {
        "git_commit": "d" * 40,
        "git_clean": True,
        "lockfile": None,
    }
    plan["repairs"] = {"exp082": {"source": repair_source}}
    monkeypatch.setattr(execution, "_checkout_source", lambda _path: repair_source)
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
            for kind in ("aggregate", "downstream", "heavy_downstream", "finalize")
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
    assert len(payload["expected_outputs"]) == sum(
        len(row["required_outputs"]) for row in execution.rows_in_order(plan)
    )
    jobs = {job["name"]: job for job in payload["jobs"]}
    standard = jobs["exp022-standard"]
    assert "--cpus-per-task=4" in standard["command"]
    assert "--mem=16G" in standard["command"]
    assert "--gres=gpu:1" in standard["command"]
    assert standard["command"][-1].endswith(
        "experiments/exp022_support/train-array.sbatch"
    )
    aggregate = jobs["ggs-exp022-aggregate"]
    assert f"--export=ALL,PINGLAB_ROOT={slurm.REPO}" in aggregate["command"]
    assert any(
        argument.startswith("--dependency=afterok:<standard-job-id>")
        for argument in aggregate["command"]
    )
    exp033 = jobs["ggs-exp033"]
    assert any("<ggs-exp041-job-id>" in argument for argument in exp033["command"])
    exp042_job = jobs["ggs-exp042"]
    exp042_dependency = next(
        argument
        for argument in exp042_job["command"]
        if argument.startswith("--dependency")
    )
    exp042_shards = jobs["ggs-exp042-inference"]
    assert "--array=0-7%8" in exp042_shards["command"]
    assert exp042_shards["shard_count"] == 8
    shard_dependency = next(
        argument
        for argument in exp042_shards["command"]
        if argument.startswith("--dependency")
    )
    assert "<ggs-exp022-aggregate-job-id>" in shard_dependency
    assert "<ggs-exp041-job-id>" in shard_dependency
    assert "<ggs-exp042-inference-job-id>" in exp042_dependency
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


def test_collection_job_uses_explicit_repository_root() -> None:
    wrapper = (
        slurm.REPO
        / "experiments"
        / "collections"
        / "gamma_gated_sparsity"
        / "collection-job.sbatch"
    ).read_text()
    assert "${PINGLAB_ROOT:?" in wrapper
    assert 'cd "$PINGLAB_ROOT"' in wrapper
    assert 'dirname "$0"' not in wrapper
    assert "run-experiment-shard" in wrapper
    assert "SLURM_ARRAY_TASK_ID" in wrapper


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
    assert len(payload["jobs"]) == 26


def test_workload_shards_are_disjoint_complete_and_stable(monkeypatch) -> None:
    jobs = [f"job-{index}" for index in range(19)]
    runner = SimpleNamespace(infer_jobs=lambda: jobs)
    monkeypatch.setattr(workloads, "_runner", lambda _slug: runner)
    monkeypatch.setitem(workloads.SHARD_COUNTS, "exp-test", 4)

    shards = [workloads.jobs_for_shard("exp-test", index, 4) for index in range(4)]
    assert [job for shard in shards for job in shard] != jobs
    assert sorted(job for shard in shards for job in shard) == sorted(jobs)
    assert all(
        set(left).isdisjoint(right)
        for i, left in enumerate(shards)
        for right in shards[i + 1 :]
    )
    assert shards == [
        workloads.jobs_for_shard("exp-test", index, 4) for index in range(4)
    ]


def test_exp037_shard_completion_uses_checkpoint_cache_tag(
    tmp_path: Path, monkeypatch
) -> None:
    train_dir = tmp_path / "training" / "coba__off__seed42"
    checkpoint = {"path": train_dir / "weights.pth", "sha256": "a" * 64}
    monkeypatch.setattr(exp037, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(exp037, "baseline_dir", lambda _model, _seed: train_dir)
    monkeypatch.setattr(exp037, "resolve_checkpoint", lambda *_args: checkpoint)
    monkeypatch.setattr(exp037, "cache_tag", lambda _checkpoint: "best__aaaa")

    output = (
        exp037._perturb_out_dir(train_dir, "drop", 0.0) / "best__aaaa" / "metrics.json"
    )
    output.parent.mkdir(parents=True)
    output.write_text("{}\n")

    assert exp037.job_is_done("sweep__coba__seed42__drop__0")


def test_exp042_shard_completion_uses_checkpoint_cache_tag(
    tmp_path: Path, monkeypatch
) -> None:
    train_dir = tmp_path / "training" / "ping__off__seed42"
    checkpoint = {"path": train_dir / "weights_final.pth", "sha256": "b" * 64}
    monkeypatch.setattr(exp042, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(exp042, "resolve_checkpoint", lambda *_args: checkpoint)
    monkeypatch.setattr(exp042, "cache_tag", lambda _checkpoint: "final__bbbb")
    spec = {
        "train_dir": train_dir,
        "condition": "baseline",
        "seed_offset": 42,
    }

    expected = (
        tmp_path
        / "artifacts"
        / "baseline"
        / train_dir.name
        / "final__bbbb"
        / "metrics.json"
    )
    assert exp042._job_metrics_path(spec) == expected


def test_plan_records_reviewed_heavy_workload_contracts(tmp_path: Path) -> None:
    plan = build_plan(tmp_path / "campaign", "production")
    rows = {row["slug"]: row for row in execution.rows_in_order(plan)}
    assert rows["exp082"]["execution"] == {
        "mode": "sharded-inference",
        "shards": 6,
        "partition": "ordered-round-robin",
        "workload_contract": {
            "condition_jobs": 132,
            "simulator_launches_max": 1_058,
            "classified_presentations": 26_400,
        },
    }
    assert rows["exp025"]["execution"] == {"mode": "monolithic"}


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
