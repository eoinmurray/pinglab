"""Exp047 contract regressions using synthetic metrics, never simulation."""

import copy
import hashlib
import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.exp047 import (
    analyse,
    collection,
    compute,
    evidence,
    historical,
    import_gold2,
    inputs,
    measurements,
    present,
    recipe,
)
from pingstore import stages
from pingstore.contracts import (
    PingstoreError,
    load_json,
    payload_digest,
    write_json_atomic,
)
from pingstore.discovery import discover_runs


def fixture_documents(cfg, item):
    config = {
        "mode": "sim",
        "model": "ping",
        "input": "synthetic-spikes",
        "n_hidden": [cfg["n_e"]],
        "n_in": cfg["n_in"],
        "n_inh": item["n_i"],
        "ei_strength": cfg["g_ei_total"],
        "ei_ratio": item["g_ie_total"] / cfg["g_ei_total"],
        "w_in": [cfg["w_in_mean"]],
        "w_in_initial_zero_fraction": cfg["w_in_initial_zero_fraction"],
        "recurrent_initial_zero_fraction": 0.0,
        "spike_rate": cfg["input_rate_hz"],
        "n_batch": cfg["n_batch"],
        "t_ms": cfg["t_ms"],
        "dt": cfg["dt_ms"],
        "seed": item["seed"],
        "dales_law": True,
        "private_w_in": False,
        "scale_w_in": 1.0,
        "scale_w_ei": 1.0,
        "scale_w_ie": 1.0,
    }
    e, i = float(item["seed"] - 30), float(item["seed"])
    metrics = {
        "mode": "probe",
        "model": "ping",
        "config": {
            "dt": cfg["dt_ms"],
            "t_ms": cfg["t_ms"],
            "n_in": cfg["n_in"],
            "n_hidden": cfg["n_e"],
            "n_inh": item["n_i"],
            "ei_strength": cfg["g_ei_total"],
            "ei_ratio": item["g_ie_total"] / cfg["g_ei_total"],
            "input_rate_hz": cfg["input_rate_hz"],
            "n_batch": cfg["n_batch"],
            "load_weights": None,
        },
        "rate_e_hz": e,
        "rate_i_hz": i,
        "rates_hz": {"hid": e, "inh": i},
    }
    return config, metrics


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for module in (compute, analyse, present, import_gold2):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(stages, "memberships", lambda _: {"exp047": "demo"})
    monkeypatch.setattr(
        stages, "_capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    monkeypatch.setenv("PINGLAB_SMOKE", "1")
    calls = []

    def simulate(args, **kwargs):
        calls.append(args)

        def arg(flag):
            return args[args.index(flag) + 1]

        cfg = recipe.configuration(smoke=arg("--t-ms") == "200.0")
        item = recipe.job(
            int(arg("--n-inh")), float(arg("--ei-ratio")), int(arg("--seed"))
        )
        config, metrics = fixture_documents(cfg, item)
        output = Path(arg("--out-dir"))
        assert args == recipe.simulation_args(cfg, item, output)
        write_json_atomic(output / "config.json", config)
        write_json_atomic(output / "metrics.json", metrics)
        (output / "run.sh").write_text("# synthetic fixture; not executed\n")

    monkeypatch.setattr(compute, "run_cli", simulate)
    return tmp_path, calls


def resign(directory):
    record = load_json(directory / "run.json")
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)


def test_independent_stages_preserve_shared_rows_and_never_publish(repo, monkeypatch):
    root, calls = repo
    compute_id = compute.compute()
    assert len(calls) == 28
    source = inputs.source(root, compute_id, "compute")
    original = source.reference
    assert source.record["inputs"] == {}
    assert source.record["execution"]["environment"] == {"PINGLAB_SMOKE": "1"}
    assert not list(source.export.rglob("run.sh"))
    assert (
        len(list((source.directory / "provenance/simulations").glob("*/run.sh"))) == 28
    )
    monkeypatch.setenv("PINGLAB_SMOKE", "0")
    monkeypatch.setattr(
        compute, "run_cli", lambda *a, **k: pytest.fail("downstream simulated")
    )
    analysis_id = analyse.analyse(compute_id)
    analysis = inputs.source(root, analysis_id, "analyse")
    result = load_json(analysis.export / "results.json")
    assert result["recipe"]["profile"] == "smoke"
    assert result["summary"]["fixed_total"]["1"]["16"] == {
        "r_e_hz_mean": 10.5,
        "r_e_hz_sd": pytest.approx(2**-0.5),
        "r_i_hz_mean": 40.5,
        "r_i_hz_sd": pytest.approx(2**-0.5),
    }
    for g, j in zip(
        ["1", "2", "4"], ["0.00390625", "0.0078125", "0.015625"], strict=True
    ):
        assert (
            result["raw"]["fixed_total"][g]["256"]
            == result["raw"]["fixed_synapse"][j]["256"]
        )
    assert (
        result["raw"]["fixed_total"]["1"]["64"]
        == result["raw"]["fixed_synapse"]["0.015625"]["64"]
    )
    assert discover_runs(root / ".pingstore/runs") == []
    monkeypatch.setattr(
        measurements, "analyse_rows", lambda *a: pytest.fail("presentation aggregated")
    )
    monkeypatch.setattr(
        evidence, "rows", lambda *a: pytest.fail("presentation remeasured")
    )
    present_id = present.present(analysis_id)
    output = inputs.source(root, present_id, "present")
    assert output.record["inputs"] == {"analysis": analysis.reference}
    numbers = load_json(output.presentation / "numbers.json")
    assert all(numbers[k] == v for k, v in result.items())
    assert all((output.presentation / name).is_file() for name in recipe.FIGURES)
    svg = (output.presentation / "pool_size_controls.svg").read_text()
    assert "Fixed mean synapse" in svg
    assert "Fixed realised synapse" not in svg
    assert all(p.is_file() for p in output.presentation.iterdir())
    assert not (root / ".artifacts").exists()
    assert inputs.source(root, compute_id, "compute").reference == original
    assert [r["id"] for r in discover_runs(root / ".pingstore/runs")] == [present_id]


@pytest.mark.parametrize(
    "smoke,count,seeds", [(False, 42, [40, 41, 42]), (True, 28, [40, 41])]
)
def test_unique_grid_keeps_original_recipe(smoke, count, seeds):
    cfg = recipe.configuration(smoke=smoke)
    jobs = recipe.jobs(cfg)
    assert len(jobs) == count
    assert len({j["id"] for j in jobs}) == count
    assert {j["seed"] for j in jobs} == set(seeds)
    assert len(list(recipe.conditions(cfg))) == 18
    assert {j["g_ie_total"] for j in jobs if j["n_i"] == 16} == {
        0.0625,
        0.125,
        0.25,
        1.0,
        2.0,
        4.0,
    }


@pytest.mark.parametrize("mode", ["simulation", "analysis", "presentation"])
def test_stage_failures_remain_hidden(repo, monkeypatch, mode):
    root, _ = repo

    def fail(*a, **k):
        raise RuntimeError("fixture failure")

    if mode == "simulation":
        monkeypatch.setattr(compute, "run_cli", fail)
        command = compute.compute
    else:
        compute_id = compute.compute()
        if mode == "analysis":
            monkeypatch.setattr(measurements, "analyse_rows", fail)
            command = partial(analyse.analyse, compute_id)
        else:
            analysis_id = analyse.analyse(compute_id)
            monkeypatch.setattr(present.plots, "plot_controls", fail)
            command = partial(present.present, analysis_id)
    before = (
        {p.name for p in (root / ".pingstore/runs").glob("exp047-*")}
        if (root / ".pingstore/runs").exists()
        else set()
    )
    with pytest.raises(RuntimeError, match="fixture failure"):
        command()
    assert {p.name for p in (root / ".pingstore/runs").glob("exp047-*")} == before
    hidden = list((root / ".pingstore/runs").glob(".exp047-*.tmp"))
    assert len(hidden) == 1
    assert (hidden[0] / "provenance/writer.lock").exists()
    with pytest.raises(PingstoreError, match="interrupted"):
        command(run_id=hidden[0].name[1:-4])


@pytest.mark.parametrize(
    "corruption", ["payload", "manifest", "nested_metadata", "symlink", "root", "v2"]
)
def test_sources_and_ancestry_reject_corruption(repo, corruption):
    root, _ = repo
    compute_id = compute.compute()
    analysis_id = analyse.analyse(compute_id)
    source = inputs.source(root, compute_id, "compute")
    if corruption == "payload":
        next(source.export.glob("probe/*/metrics.json")).write_text("{}")
    elif corruption == "nested_metadata":
        next(
            (source.directory / "provenance/simulations").glob("*/config.json")
        ).write_text("{}")
    elif corruption == "symlink":
        (source.export / "link").symlink_to(source.export / "evidence.json")
    elif corruption == "root":
        (source.directory / "unexpected.txt").write_text("invalid root")
    else:
        record = load_json(source.directory / "run.json")
        if corruption == "v2":
            record["schema"] = "pingstore.run/v2"
        else:
            record["execution"]["note"] = "manifest changed"
        write_json_atomic(source.directory / "run.json", record)
    with pytest.raises(PingstoreError):
        present.present(analysis_id)


@pytest.mark.parametrize(
    "field,value",
    [("n_inh", 99), ("seed", 99), ("w_in", [9.0]), ("private_w_in", True)],
)
def test_resigned_wrong_scientific_config_is_rejected(repo, field, value):
    root, _ = repo
    identity = compute.compute()
    source = inputs.source(root, identity, "compute")
    path = next((source.directory / "provenance/simulations").glob("*/config.json"))
    doc = load_json(path)
    doc[field] = value
    write_json_atomic(path, doc)
    resign(source.directory)
    with pytest.raises(PingstoreError, match="configuration differs"):
        analyse.analyse(identity)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, "1", True])
def test_nonfinite_negative_and_non_numeric_rates_fail(value):
    with pytest.raises(PingstoreError, match="finite and nonnegative"):
        evidence.finite_rate(value)


def test_missing_metric_cannot_be_replaced_with_implicit_recomputation(repo):
    root, calls = repo
    identity = compute.compute()
    source = inputs.source(root, identity, "compute")
    next(source.export.glob("probe/*/metrics.json")).unlink()
    resign(source.directory)
    with pytest.raises(PingstoreError, match="metric grid"):
        analyse.analyse(identity)
    assert len(calls) == 28


def test_wrong_stage_and_experiment_are_rejected(repo):
    root, _ = repo
    identity = compute.compute()
    with pytest.raises(PingstoreError, match="not an exp047 analyse"):
        present.present(identity)
    source = inputs.source(root, identity, "compute")
    record = load_json(source.directory / "run.json")
    record["experiment"] = "exp048"
    write_json_atomic(source.directory / "run.json", record)
    with pytest.raises(PingstoreError):
        analyse.analyse(identity)


@pytest.mark.parametrize("stage", ["analyse", "present"])
def test_ancestor_changes_during_stage_prevent_completion(repo, monkeypatch, stage):
    root, _ = repo
    compute_id = compute.compute()
    source = inputs.source(root, compute_id, "compute")
    if stage == "analyse":
        original = measurements.analyse_rows

        def changed(*args):
            result = original(*args)
            (source.directory / "README.md").write_text("changed during analysis")
            return result

        monkeypatch.setattr(measurements, "analyse_rows", changed)
        command, identity = analyse.analyse, compute_id
    else:
        analysis_id = analyse.analyse(compute_id)

        def changed(*args):
            record = load_json(source.directory / "run.json")
            record["execution"]["note"] = "changed ancestor during plotting"
            write_json_atomic(source.directory / "run.json", record)

        monkeypatch.setattr(present.plots, "plot_controls", changed)
        command, identity = present.present, analysis_id
    with pytest.raises(PingstoreError):
        command(identity)
    assert not list((root / ".pingstore/runs").glob(f"exp047-*-{stage}"))


def test_analysis_grid_validation_does_not_accept_invented_or_missing_rows(repo):
    root, _ = repo
    identity = analyse.analyse(compute.compute())
    source = inputs.source(root, identity, "analyse")
    path = source.export / "results.json"
    result = load_json(path)
    result["raw"]["fixed_synapse"]["0.015625"]["64"][0]["r_e_hz"] += 1
    write_json_atomic(path, result)
    resign(source.directory)
    with pytest.raises(PingstoreError, match="shared simulation rows disagree"):
        present.present(identity)


def row_for(root):
    from experiments.collections.gamma_gated_sparsity.plan import build_plan

    plan = build_plan(root / "campaign", "fixture", smoke=True)
    return next(
        row
        for stage in plan["stages"]
        for row in stage["experiments"]
        if row["slug"] == "exp047"
    )


def test_collection_plan_registers_exp047_adapter_and_rejects_legacy(repo):
    from experiments.collections.gamma_gated_sparsity import execution

    root, _ = repo
    row = row_for(root)
    assert row["command"] == []
    assert row["execution"] == {
        "mode": "exp047-staged",
        "stages": list(collection.STAGES),
    }
    assert row["required_outputs"] == [
        str(root / "campaign/downstream/exp047/stage-refs.json")
    ]
    assert row["dependencies"] == ()
    assert execution._stage_adapter("exp047") is collection
    assert not execution._outputs_valid_for_plan(
        {}, {**row, "execution": {"mode": "monolithic"}}
    )


def test_slurm_reserves_exp047_stages_before_mock_submission(repo, monkeypatch):
    from experiments.collections.gamma_gated_sparsity import slurm
    from experiments.collections.gamma_gated_sparsity.plan import build_plan

    root, _ = repo
    campaign = root / "campaign"
    plan = build_plan(campaign, "fixture", smoke=True)
    plan.update(
        profile="smoke",
        source={"git_commit": "fixture"},
        exp022_manifest=str(root / "bank.json"),
    )
    write_json_atomic(root / "bank.json", {"manifest_sha256": "fixture"})
    resources = root / "resources.json"
    resources.write_text("{}")
    monkeypatch.setattr(slurm, "REPO", root)
    monkeypatch.setattr(slurm, "load_plan", lambda _: plan)
    monkeypatch.setattr(slurm, "load_resources", lambda _: {})
    monkeypatch.setattr(
        slurm, "_outputs_valid_for_plan", lambda _, row: row["slug"] != "exp047"
    )
    monkeypatch.setattr(
        slurm, "_run", lambda *a, **k: pytest.fail("real scheduler call")
    )
    calls = []

    def submit(*args, **kwargs):
        calls.append(kwargs["name"])
        if kwargs["name"] == "ggs-exp047":
            reserved = load_json(campaign / "downstream/exp047/stage-reservations.json")
            assert set(reserved) == set(collection.STAGES)
            for stage, identity in reserved.items():
                record = stages.stage_reservation(
                    root / ".pingstore/runs" / f".{identity}.tmp"
                )
                assert record["stage"] == stage
                assert record["origin"] == "slurm-wilkes"
            assert kwargs["dependencies"] == []
        return {"name": kwargs["name"], "job_id": "mock-" + kwargs["name"]}

    monkeypatch.setattr(slurm, "_submit_job", submit)
    slurm.submit_campaign(campaign, resources, submit=True)
    assert calls == ["ggs-exp047", "ggs-finalize"]


def test_collection_explicit_dispatch_reuse_and_profile_guard(repo, monkeypatch):
    root, _ = repo
    row, plan = row_for(root), {"profile": "smoke"}
    calls = []

    def dispatch(command, **kwargs):
        calls.append(command)
        stage = command[2].rsplit(".", 1)[-1]
        identity = command[command.index("--run-id") + 1]
        if stage == "compute":
            assert "--source" not in command
            compute.compute(run_id=identity)
        else:
            source_id = command[command.index("--source") + 1]
            getattr({"analyse": analyse, "present": present}[stage], stage)(
                source_id, run_id=identity
            )
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(collection.subprocess, "run", dispatch)
    refs = collection.execute(root, plan, row)
    assert len(calls) == 3
    assert collection.execute(root, plan, row) == refs
    assert len(calls) == 3
    with pytest.raises(PingstoreError, match="profile differs"):
        collection.execute(root, {"profile": "production"}, row)
    altered = copy.deepcopy(refs)
    del altered["compute"]
    write_json_atomic(Path(row["required_outputs"][0]), altered)
    with pytest.raises(PingstoreError, match="incomplete stage lineage"):
        collection.references(root, row)


def test_reservations_reject_legacy_and_orphaned_completion(repo):
    root, _ = repo
    row = row_for(root)
    reserved = collection.reserve(root, row, origin="slurm-wilkes")
    assert collection.reserve(root, row) == reserved
    for stage, identity in reserved.items():
        assert identity.endswith("-" + stage)
        record = stages.stage_reservation(root / ".pingstore/runs" / f".{identity}.tmp")
        assert record["origin"] == "slurm-wilkes"
    compute.compute(run_id=reserved["compute"])
    with pytest.raises(PingstoreError, match="lacks campaign reference"):
        collection.reserve(root, row)
    with pytest.raises(PingstoreError, match="legacy exp047"):
        collection.require_staged({"execution": {"mode": "monolithic"}})


def test_retired_runner_fails_before_any_output(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "experiments.exp047", "--plot-only"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "independent stages" in result.stderr


def test_package_import_is_inert():
    root = Path(__file__).resolve().parents[2]
    code = "import sys,json; import experiments.exp047; print(json.dumps(sorted(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = json.loads(result.stdout)
    assert not any(
        name in loaded
        for name in ("torch", "matplotlib.pyplot", "experiments.exp047.compute")
    )


@pytest.fixture
def archive(tmp_path):
    root, live = tmp_path / "archive", tmp_path / "remote"
    root.mkdir()
    live.mkdir()
    cfg, rows = recipe.configuration(), {}
    for item in recipe.jobs(cfg):
        directory = root / historical.STATE / item["id"]
        directory.mkdir(parents=True)
        config, metric = fixture_documents(cfg, item)
        write_json_atomic(directory / "config.json", config)
        write_json_atomic(directory / "metrics.json", metric)
        for name in ("run.sh", "run.jsonl", "output.log"):
            (directory / name).write_text("synthetic historical fixture\n")
        rows[item["id"]] = evidence.metric(metric, cfg, item)
    source = {
        "git_commit": historical.COMMIT,
        "git_clean": True,
        "lockfile": {"sha256": "a" * 64},
    }
    records = {
        "lineage.json": {
            "sources": {"base": {"run_id": historical.CAMPAIGN}},
            "selection": {"base_experiment_state": ["exp047"]},
        },
        f"{historical.BASE}/run.json": {
            "run_id": historical.CAMPAIGN,
            "source": source,
        },
        f"{historical.BASE}/collection-plan.json": {
            "source": source,
            "stages": [
                {
                    "experiments": [
                        {
                            "slug": "exp047",
                            "dependencies": [],
                            "training_run": None,
                            "execution": {"mode": "monolithic"},
                        }
                    ]
                }
            ],
        },
        f"{historical.BASE}/collection-status/exp047.json": {
            "state": "complete",
            "experiment": "exp047",
        },
        f"{historical.DERIVED}/numbers.json": {
            **measurements.analyse_rows(rows, cfg),
            "run_id": "r001",
            "collection_provenance": {
                "campaign_id": historical.CAMPAIGN,
                "source_git_commit": historical.COMMIT,
                "experiment": "exp047",
                "dependencies": [],
                "training_run": None,
                "lockfile_sha256": "a" * 64,
            },
        },
    }
    for name, value in records.items():
        write_json_atomic(root / name, value)
    for name, value in {
        f"{historical.BASE}/logs/collection/ggs-exp047_{historical.JOB}.out": f"job={historical.JOB} host=fixture action=run-experiment experiment=exp047\n[published] fixture\n",
        f"{historical.BASE}/logs/collection/ggs-exp047_{historical.JOB}.err": "",
        f"{historical.DERIVED}/run.sh": "# fixture",
        f"{historical.DERIVED}/pool_size_controls.svg": "<svg/>",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    write_json_atomic(
        root / "run.json",
        {
            "run_id": "gold-2",
            "contract_version": "runstore/v1",
            "archive": {"uri": historical.URI},
        },
    )
    refresh_inventory(root, live)
    return root, live


def refresh_inventory(root, live):
    rows = [
        {
            "path": str(p.relative_to(root)),
            "size_bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in root.rglob("*")
        if p.is_file()
        and str(p.relative_to(root)) not in ("run.json", "inventory.json")
    ]
    write_json_atomic(
        root / "inventory.json",
        {
            "run_id": "gold-2",
            "contract_version": "runstore/v1",
            "files": rows,
            "file_count": len(rows),
            "total_size_bytes": sum(row["size_bytes"] for row in rows),
        },
    )
    for name in ("run.json", "inventory.json"):
        (live / name).write_bytes((root / name).read_bytes())


def test_historical_plan_selects_all_metrics_and_retains_original_inventory(archive):
    import gzip

    root, live = archive
    plan = historical.make_plan(root, live)
    assert plan["source_file_count"] == 220
    metrics = [row for row in plan["files"] if row["target"].startswith("export/")]
    assert len(metrics) == 42
    assert plan["upstream_inputs"] == {}
    assert plan["producer"]["source_git_commit"] == historical.COMMIT
    original_inventory = (root / "inventory.json").read_bytes()
    inventory_row = next(
        row for row in plan["files"] if row["path"] == "inventory.json"
    )
    compressed = gzip.compress(original_inventory, mtime=0)
    assert inventory_row["retained_bytes"] == len(compressed)
    assert gzip.decompress(compressed) == original_inventory
    assert plan["excluded_experiment_files"][0]["path"].endswith(".svg")
    assert not (root / ".pingstore").exists()


@pytest.mark.parametrize(
    "corruption",
    ["live", "checksum", "config", "summary", "producer", "job", "symlink"],
)
def test_historical_plan_rejects_inconsistent_evidence(archive, corruption):
    root, live = archive
    if corruption == "live":
        (live / "run.json").write_text("{}")
    elif corruption == "checksum":
        next((root / historical.STATE).glob("*/metrics.json")).write_text("{}")
    elif corruption == "symlink":
        path = root / historical.DERIVED / "run.sh"
        path.unlink()
        path.symlink_to(live / "run.json")
    else:
        if corruption == "config":
            path = next((root / historical.STATE).glob("*/config.json"))
            doc = load_json(path)
            doc["seed"] = 99
        elif corruption == "summary":
            path = root / historical.DERIVED / "numbers.json"
            doc = load_json(path)
            doc["summary"]["fixed_total"]["1"]["16"]["r_e_hz_mean"] += 1
        elif corruption == "producer":
            path = root / historical.BASE / "run.json"
            doc = load_json(path)
            doc["source"]["git_commit"] = "b" * 40
        else:
            path = (
                root
                / historical.BASE
                / "logs/collection"
                / f"ggs-exp047_{historical.JOB}.out"
            )
            path.write_text(path.read_text().replace(historical.JOB, "99999999"))
            refresh_inventory(root, live)
            with pytest.raises(PingstoreError):
                historical.make_plan(root, live)
            return
        write_json_atomic(path, doc)
        refresh_inventory(root, live)
    with pytest.raises(PingstoreError):
        historical.make_plan(root, live)


def test_import_preserves_approved_bytes_producer_and_independent_stages(repo, archive):
    import gzip

    root, calls = repo
    source, live = archive
    plan = historical.make_plan(source, live)
    identity = import_gold2.import_subset(source, plan, live)
    run = inputs.source(root, identity, "compute")
    assert calls == []
    assert run.record["execution"]["operation"] == "historical-import"
    assert run.record["origin"] == "local"
    assert run.record["inputs"] == {}
    assert run.record["historical_import"]["producer"] == plan["producer"]
    for row in plan["files"]:
        data = (run.directory / row["target"]).read_bytes()
        assert len(data) == row["retained_bytes"]
        if row["encoding"] == "gzip":
            data = gzip.decompress(data)
        assert data == (source / row["path"]).read_bytes()
    assert historical.make_plan(source, live) == plan
    analysis_id = analyse.analyse(identity)
    present_id = present.present(analysis_id)
    numbers = load_json(
        inputs.source(root, present_id, "present").presentation / "numbers.json"
    )
    original = load_json(source / historical.DERIVED / "numbers.json")
    assert all(
        numbers[k] == original[k] for k in ("config", "definition", "raw", "summary")
    )
    assert not (root / ".artifacts").exists()


def test_changed_plan_is_rejected_before_allocation(repo, archive):
    root, _ = repo
    source, live = archive
    plan = historical.make_plan(source, live)
    plan["files"][0]["target"] = "../../escape"
    with pytest.raises(PingstoreError, match="plan differs"):
        import_gold2.import_subset(source, plan, live)
    assert not (root / ".pingstore").exists()


def test_source_mutation_during_import_never_exposes_completed_run(
    repo, archive, monkeypatch
):
    root, _ = repo
    source, live = archive
    plan = historical.make_plan(source, live)
    original = import_gold2.copy_selected

    def changed(*args):
        result = original(*args)
        (source / args[2]["path"]).write_bytes(b"mutated after copying")
        return result

    monkeypatch.setattr(import_gold2, "copy_selected", changed)
    with pytest.raises(PingstoreError):
        import_gold2.import_subset(source, plan, live)
    assert not list((root / ".pingstore/runs").glob("exp047-*"))
    assert len(list((root / ".pingstore/runs").glob(".exp047-*.tmp"))) == 1


def test_article_renders_explicit_presentation_with_equations_and_seed_list(repo):
    import re
    import shutil

    from demolab_cli import _paths

    root, _ = repo
    cid = compute.compute()
    aid = analyse.analyse(cid)
    pid = present.present(aid)
    output = inputs.source(root, pid, "present")
    source_root = Path(__file__).resolve().parents[2]
    (root / "writings").mkdir()
    for name in ("exp047.typ", "run-inputs.typ"):
        shutil.copy2(source_root / "writings" / name, root / "writings" / name)
    (root / ".demolab").mkdir()
    shutil.copy2(_paths.TYP / "lib.typ", root / ".demolab/lib.typ")
    write_json_atomic(
        root / "preview.json",
        {"exp047": {"exp047": "/" + str(output.export.relative_to(root))}},
    )
    document = root / "document.typ"
    document.write_text(
        '#set page(paper: "a4", margin: 18mm)\n'
        '#set text(size: 10pt)\n#import "writings/exp047.typ": body\n#body\n'
    )
    command = [
        _paths.find_typst(source_root),
        "compile",
        "--root",
        str(root),
        "--input",
        "demolab-preview-file=/preview.json",
    ]
    for extra, target in (
        (["--format", "png", "--ppi", "80"], "article-{p}.png"),
        (["--features", "html", "--format", "html"], "article.html"),
    ):
        result = subprocess.run(
            command + extra + [str(document), str(root / target)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
    assert list(root.glob("article-*.png"))
    html = (root / "article.html").read_text()
    images = re.findall(r"<img\b[^>]*>", html)
    assert len(images) == 1
    assert 'alt="' in images[0] and 'src="data:image/svg+xml' in images[0]
    assert html.count('<math display="block">') == 2
    assert html.index(">Results<") < html.index(">Methods<")
    assert "seeds 40, 41" in html
    assert "15.625" in html  # Exact recipe, not divergent half-way rounding.
    (output.export / "numbers.json").write_text("broken JSON")
    result = subprocess.run(
        command
        + [
            "--features",
            "html",
            "--format",
            "html",
            str(document),
            str(root / "broken.html"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    write_json_atomic(root / "preview.json", {})
    result = subprocess.run(
        command
        + [
            "--features",
            "html",
            "--format",
            "html",
            str(document),
            str(root / "pending.html"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "required run is unavailable" in (root / "pending.html").read_text()
