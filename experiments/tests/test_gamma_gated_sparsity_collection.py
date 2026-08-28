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
from experiments.exp023 import collection as exp023_collection
from experiments.exp024 import collection as exp024_collection
from experiments.exp025 import collection as exp025_collection
from experiments.exp037 import collection as exp037_collection
from experiments.exp038 import collection as exp038_collection
from experiments.exp041 import collection as exp041_collection
from experiments.exp042 import collection as exp042_collection
from experiments.exp044 import collection as exp044_collection
from experiments.exp046 import collection as exp046_collection
from experiments.exp049 import collection as exp049_collection
from experiments.exp081 import collection as exp081_collection


def test_collection_production_training_horizon_is_50_epochs() -> None:
    assert exp022.EPOCHS_STANDARD == 50
    assert exp080.EPOCHS_STANDARD == 50


def test_downstream_cell_banks_resolve_through_exp022_registry() -> None:
    registered = {
        run_id: {cell["name"] for cell in exp022.training_run_cells(run_id)}
        for run_id in ("TR-02", "TR-03", "TR-04", "TR-05", "TR-06", "TR-07")
    }

    for module, seeds in (
        (exp037, exp037.SEEDS_BASELINE),
    ):
        assert {
            module.cell_name(model, target, seed)
            for model in module.MODELS
            for target in module.RATE_TARGET_GRID_HZ
            for seed in seeds
        } == registered["TR-02"]
    assert {
        exp038.cell_name(m, t, s)
        for m in exp038.MODELS
        for t in exp038.RATE_TARGET_GRID_HZ
        for s in exp038.SEEDS_BASELINE
    } == registered["TR-02"]
    assert {
        exp025.cell_name(m, t, s)
        for m in exp025.MODELS
        for t in exp025.RATE_TARGET_GRID_HZ
        for s in exp025.SEEDS
    } == registered["TR-02"]
    assert {
        exp024.cell_name(model, seed)
        for model in exp024.MODELS
        for seed in exp024.SEEDS
    } <= registered["TR-02"]

    assert {
        exp041.cell_name(tau, seed)
        for tau in exp041.TAU_GABA_SWEEP
        for seed in exp041.SEEDS
    } == registered["TR-03"]
    assert {
        exp046.cell_name(tau, seed)
        for tau in exp046.TAU_GABA_SWEEP_MS
        for seed in exp046.SEEDS
    } == registered["TR-03"]
    assert {
        exp044.cell_name(dt, seed)
        for dt in exp044.DT_SWEEP_MS
        for seed in exp044.SEEDS
    } == registered["TR-04"]
    assert {
        exp049.cell_name(condition, seed)
        for condition in exp049.COND_ORDER
        for seed in exp049.SEEDS
    } == registered["TR-05"]
    assert {exp082.training_dir(seed).name for seed in exp082.SEEDS} == registered[
        "TR-06"
    ]
    assert {
        exp025.low_w_in_cell_name(w_in, seed)
        for w_in in exp025.LOW_W_IN_VALUES
        for seed in exp025.LOW_W_IN_SEEDS
    } <= registered["TR-07"]

    assert {exp042.cell_name(seed) for seed in exp042.SEEDS} <= registered["TR-02"]


def test_graph_orders_dependencies_and_replaces_exp048_with_exp082() -> None:
    ordered = ordered_experiments()
    positions = {experiment.slug: index for index, experiment in enumerate(ordered)}
    assert "exp048" not in positions
    assert positions["exp022"] < positions["exp082"]
    assert positions["exp041"] < positions["exp033"]
    assert positions["exp041"] < positions["exp054"]
    assert positions["exp022"] < positions["exp042"]
    assert {"exp023", "exp047", "exp080", "exp081"} <= positions.keys()
    exp082 = next(
        experiment for experiment in EXPERIMENTS if experiment.slug == "exp082"
    )
    assert exp082.training_run == "TR-06"
    exp042_node = next(
        experiment for experiment in EXPERIMENTS if experiment.slug == "exp042"
    )
    assert exp042_node.dependencies == ("exp022",)
    assert exp042_node.training_run == "TR-02"


def test_exp042_declares_checkpoint_sources_by_owner_and_training_run() -> None:
    assert [exp042.cell_name(seed) for seed in exp042.SEEDS] == [
        "ping__off__seed42",
        "ping__off__seed43",
        "ping__off__seed44",
    ]


def test_exp042_catalog_contains_only_figure_generating_jobs() -> None:
    jobs = [job["id"] for job in exp042.jobs(exp042.configuration())]
    assert len(jobs) == 66
    assert all("xtau" not in job and "alpha_mix" not in job for job in jobs)
    assert workloads.workload_contract("exp042", smoke=False) == {
        "condition_jobs": 66,
        "simulator_launches_max": 66,
    }
    assert workloads.workload_contract("exp042", smoke=True) == {
        "condition_jobs": 39,
        "simulator_launches_max": 39,
    }


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
    assert all(row["command"] or row["execution"]["mode"] in {"exp023-staged", "exp024-staged", "exp025-staged", "exp037-staged", "exp038-staged", "exp041-staged", "exp042-staged", "exp044-staged", "exp046-staged", "exp049-staged", "exp081-staged"} for row in rows)
    audit = next(row for row in rows if row["slug"] == "exp024")
    assert audit["execution"]["stages"] == ["analyse", "present"]
    assert not any(".artifacts" in path for path in audit["required_outputs"])
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
        (tmp_path / "campaign/derived/.artifacts").resolve()
    )
    assert environment["PINGLAB_SMOKE"] == "1"


def test_init_composes_pingstore_and_exp022_manifests(
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

    def fake_initialize(_root, **_kwargs):
        (root / "exp022").mkdir(parents=True)
        (root / "downstream").mkdir()
        (root / "derived/.artifacts").mkdir(parents=True)
        (root / "logs").mkdir()
        execution.write_json_atomic(root / "run.json", {"source": source})

    def fake_run(command, **_kwargs):
        calls.append(command)
        if "experiments.exp022.compute" in command:
            exp022 = root / "exp022"
            exp022.mkdir(exist_ok=True)
            (exp022 / "campaign.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(execution, "initialize_run", fake_initialize)
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

    # Scientific work is mocked here; real stage-reference validation has its
    # own fixture-run coverage in test_exp024_stages.py and test_exp081.py.
    for adapter in (exp023_collection, exp024_collection, exp025_collection, exp037_collection, exp038_collection, exp041_collection, exp042_collection, exp044_collection, exp046_collection, exp049_collection, exp081_collection):
        monkeypatch.setattr(adapter, "completed", lambda repo, plan, row:
                            execution.load_json(Path(row["required_outputs"][0])))

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
            derived = source_root / "derived/.artifacts" / slug
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
        "_inspect_campaign",
        lambda root: (
            {"inventory": "valid"} if root == base else {"inventory": "absent"}
        ),
    )

    def fake_initialize(_root, **_kwargs):
        (destination / "exp022").mkdir(parents=True)
        (destination / "downstream").mkdir()
        (destination / "derived").mkdir(parents=True)
        (destination / "logs").mkdir()
        execution.write_json_atomic(
            destination / "run.json",
            {
                "run_id": "composite-run",
                "source": {"git_commit": "d" * 40},
            },
        )

    monkeypatch.setattr(execution, "initialize_run", fake_initialize)
    result = execution.compose_campaign(
        destination,
        "composite-run",
        base_root=base,
        overlay_root=overlay,
        replacements=["exp022", "exp025"],
    )

    assert result["experiments"] == len(slugs)
    assert (
        execution.load_json(destination / "derived/.artifacts/exp022/numbers.json")[
            "marker"
        ]
        == "repair-run"
    )
    assert (
        execution.load_json(destination / "derived/.artifacts/exp047/numbers.json")[
            "marker"
        ]
        == "base-run"
    )
    composition = execution.load_json(destination / "composition.json")
    assert composition["experiments"]["exp025"]["run_id"] == "repair-run"
    assert composition["experiments"]["exp047"]["run_id"] == "base-run"

    composite_plan = execution.load_json(destination / execution.PLAN_NAME)
    rows = {row["slug"]: row for row in execution.rows_in_order(composite_plan)}
    assert not execution._outputs_valid_for_plan(composite_plan, rows["exp025"])
    assert execution._outputs_valid_for_plan(composite_plan, rows["exp047"])

    (destination / "derived/.artifacts/exp047/figure.svg").write_text(
        "<svg><text>tampered</text></svg>\n"
    )
    assert not execution._outputs_valid_for_plan(composite_plan, rows["exp047"])


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
    source_dir = repair_root / "derived/.artifacts/exp082"
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


def test_finalize_captures_campaign_and_writes_pingstore_inventory(
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

    calls = []
    staged = {"slug": "exp024", "execution": {"mode": "exp024-staged"}}
    staged_root = {"slug": "exp081", "execution": {"mode": "exp081-staged"}}
    staged_audit = {"slug": "exp044", "execution": {"mode": "exp044-staged"}}
    legacy = {"slug": "exp023", "execution": {"mode": "monolithic"}}
    monkeypatch.setattr(execution, "load_plan", lambda _root: {
        "campaign_id": "smoke", "stages": [{"experiments": [staged, staged_root, staged_audit, legacy]}],
    })
    monkeypatch.setattr(
        execution,
        "capture_campaign_metadata",
        lambda captured_root, plan: calls.append((captured_root, plan)),
    )
    assert execution.finalize_campaign(root) == {
        "campaign_id": "smoke",
        "status": "complete",
        "file_count": 0,
        "total_size_bytes": 0,
        "payload_digest": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    }
    assert calls == [(root, {"campaign_id": "smoke", "stages": [{"experiments": [legacy]}]})]


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
    promotions = []
    from pingstore import materialize
    for adapter in (exp023_collection, exp024_collection, exp025_collection, exp037_collection, exp038_collection, exp041_collection, exp042_collection, exp044_collection, exp046_collection, exp049_collection, exp081_collection):
        monkeypatch.setattr(adapter, "completed", lambda repo, plan, row:
                            SimpleNamespace(record={"run_id": row["slug"] + "-r003-present-local"}))
    monkeypatch.setattr(materialize, "materialize_run", lambda store, identity, target:
                        promotions.append((store, identity.split("-", 1)[0], target)))
    monkeypatch.setattr(
        execution,
        "promote_experiment",
        lambda root, slug, *, artifacts_root: promotions.append(
            (root, slug, artifacts_root)
        ),
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="built 1 entry", stderr="")

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    result = execution.build_publication(root, checkout)
    assert result["promoted"] == [row["slug"] for row in execution.rows_in_order(plan)]
    assert len(promotions) == len(result["promoted"])
    assert calls == [
        (
            [
                "/usr/bin/uv",
                "run",
                "--frozen",
                "--project",
                str(checkout),
                "demolab",
                "build",
            ],
            {"cwd": checkout, "check": True, "capture_output": True, "text": True},
        )
    ]


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
    monkeypatch.setattr(execution, "promote_experiment", lambda *_args, **_kwargs: None)
    from pingstore import materialize
    for adapter in (exp023_collection, exp024_collection, exp025_collection, exp037_collection, exp038_collection, exp041_collection, exp042_collection, exp044_collection, exp046_collection, exp049_collection, exp081_collection):
        monkeypatch.setattr(adapter, "completed", lambda repo, plan, row:
                            SimpleNamespace(record={"run_id": row["slug"] + "-r003-present-local"}))
    monkeypatch.setattr(materialize, "materialize_run", lambda *args: None)

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
        "experiments/exp022/slurm/train-array.sbatch"
    )
    assert Path(standard["command"][-1]).is_file()
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
    assert "<ggs-exp041-job-id>" not in shard_dependency
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


def test_exp037_legacy_shard_execution_requires_explicit_v3_bank():
    from experiments.collections.gamma_gated_sparsity.workloads import execute_shard
    with pytest.raises(ValueError, match="explicit v3 bank"):
        execute_shard("exp037", 0, 6, smoke=False)


def test_exp042_legacy_shard_execution_requires_explicit_v3_bank():
    with pytest.raises(ValueError, match="explicit v3 bank"):
        workloads.execute_shard("exp042", 0, 8, smoke=False)


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
    assert rows["exp025"]["execution"] == {"mode": "exp025-staged", "stages": ["compute", "analyse", "present"]}


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
