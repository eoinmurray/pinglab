"""Synthetic staged probes; no production simulations or historical imports."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from experiments.exp037 import (
    analyse,
    collection,
    compute,
    inputs,
    measurements,
    present,
    recipe,
)
from experiments.exp044.test import _common_config
from pingstore import stages
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    payload_digest,
    write_json_atomic,
)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(
        stages, "memberships", lambda _: {"exp022": "demo", "exp037": "demo"}
    )
    monkeypatch.setattr(
        stages, "_capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    monkeypatch.setenv("PINGLAB_SMOKE", "1")
    with stages.stage_run(
        tmp_path, "exp022", "compute", export_root="export/cells"
    ) as run:
        for cell in recipe.bank_cells():
            directory = run.export / "cells" / cell["cell_name"]
            directory.mkdir(parents=True)
            cfg = {
                **_common_config(),
                "dt": 0.1,
                "seed": cell["seed"],
                "n_hidden": 200,
                "n_inh": 64,
                "hidden_sizes": [200],
                "ei_strength": float(cell["model"] == "ping"),
                "ei_ratio": 2.0,
                "v_grad_dampen": 1000.0 if cell["model"] == "ping" else 1.0,
                "fr_reg_upper_strength": 0.0
                if cell["rate_target_hz"] is None
                else 0.041,
                "fr_reg_upper_target_hz": cell["rate_target_hz"] or 0.0,
            }
            write_json_atomic(directory / "config.json", cfg)
            cps = {}
            for role, filename, epoch in (
                ("best_validation", "weights.pth", 43),
                ("final_epoch", "weights_final.pth", 50),
            ):
                target = directory / filename
                target.write_bytes((cell["cell_name"] + role).encode())
                cps[role] = {
                    "filename": filename,
                    "epoch": epoch,
                    "sha256": file_sha256(target),
                }
            write_json_atomic(
                directory / "metrics.json",
                {
                    "config": cfg,
                    "training_cell_name": cell["cell_name"],
                    "best_epoch": 43,
                    "best_acc": 91.0,
                    "checkpoints": cps,
                    "epochs": [
                        {
                            "ep": ep,
                            "acc": 80 + ep / 10,
                            "rate_e": ep / 2,
                            "test_rate_e": ep / 3,
                            "test_rate_i": ep / 4,
                        }
                        for ep in range(1, 51)
                    ],
                },
            )
    calls = []

    def simulate(args, **kwargs):
        from config import save_selected_npz

        fields = []
        if "--output-fields" in args:
            for arg in args[args.index("--output-fields") + 1 :]:
                if arg.startswith("--"):
                    break
                fields.append(arg)

        def save(path, **arrays):
            save_selected_npz(path, arrays, fields or None)

        assert kwargs == {"no_sync": True}
        calls.append(args)

        def value(key):
            return args[args.index(key) + 1]

        assert Path(value("--load-weights")).name == "weights.pth"
        out = Path(value("--out-dir"))
        out.mkdir(parents=True)
        train = load_json(Path(value("--load-config")))
        uniform = "--n-batch" in args
        strength = (
            float(value("--ei-strength"))
            if "--ei-strength" in args
            else train["ei_strength"]
        )
        rate = float(value("--input-rate")) if "--input-rate" in args else 25.0
        cfg = {
            **train,
            "perturb_mode": value("--perturb-mode"),
            "perturb_level": [float(value("--perturb-level"))],
            "load_config": value("--load-config"),
            "load_weights": value("--load-weights"),
            "input": "synthetic-spikes" if uniform else "dataset",
            "infer": not uniform,
            "n_hidden": [train["n_hidden"]],
            "tau_gaba": train["tau_gaba_ms"],
            "ei_strength": strength,
            "spike_rate": rate,
            "scale_w_in": 1.0,
            "scale_w_ei": 1.0,
            "scale_w_ie": 1.0,
            "scale_projection": [],
            "intervention": [],
            "max_samples": int(value("--max-samples"))
            if "--max-samples" in args
            else None,
            "skip_load": ["W_ei.", "W_ie."] if "--skip-load" in args else [],
        }
        if "--sample-index" in args:
            cfg["sample_index"] = int(value("--sample-index"))
            e, i = np.zeros((2000, 200), dtype=bool), np.zeros((2000, 64), dtype=bool)
            e[::20, ::2] = True
            i[::30] = True
            save(
                out / "snapshot.npz",
                dt=0.1,
                n_e=200,
                n_i=64,
                label=7,
                spk_e=e,
                spk_i=i,
                unused_voltage=np.zeros((2000, 200)),
            )
        elif uniform:
            cfg["n_batch"] = int(value("--n-batch"))
            write_json_atomic(
                out / "metrics.json",
                {
                    "config": {
                        **train,
                        "ei_strength": strength,
                        "input_rate_hz": rate,
                        "n_batch": cfg["n_batch"],
                    },
                    "rate_e_hz": rate / 2,
                    "rate_i_hz": rate,
                },
            )
        else:
            n = cfg["max_samples"]
            acc = (
                90 - int(cfg["perturb_level"][0])
                if cfg["perturb_mode"] == "add"
                else 90
            )
            write_json_atomic(
                out / "metrics.json",
                {
                    "config": {
                        **train,
                        "ei_strength": strength,
                        "load_weights": value("--load-weights"),
                        "evaluation_partition": "official_mnist_test",
                        "evaluation_samples": n,
                    },
                    "best_acc": acc,
                    "n_correct": n * acc // 100,
                    "n_total": n,
                    "rates_hz": {"hid": 20.0, "inh": 10.0},
                },
            )
        write_json_atomic(out / "config.json", cfg)
        (out / "run.sh").write_text("synthetic simulator command\n")

    monkeypatch.setattr(compute, "run_cli", simulate)
    return tmp_path, run.run_id, calls


def resign(directory):
    path = directory / "run.json"
    record = load_json(path)
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(path, record)


@pytest.mark.parametrize("mutation", ["sample_count", "snapshot", "config", "missing"])
def test_analyse_rejects_corrupt_even_resigned_payload(lab, mutation):
    root, bank, _ = lab
    cid = compute.compute(bank)
    c = inputs.source(root, cid, "compute")
    cfg = recipe.configuration(smoke=True)
    jobs = recipe.jobs(cfg)
    if mutation == "snapshot":
        p = (
            c.export
            / next(j["path"] for j in jobs if j["kind"] == "raster")
            / "snapshot.npz"
        )
        with np.load(p) as raw:
            data = {k: raw[k] for k in raw.files}
        data["spk_e"] = np.full_like(data["spk_e"], 2, dtype=np.int8)
        np.savez_compressed(p, **data)
    elif mutation == "config":
        p = c.directory / "provenance/simulations" / jobs[0]["path"] / "config.json"
        d = load_json(p)
        d["spike_rate"] = 111
        write_json_atomic(p, d)
    else:
        job = next(j for j in jobs if j["kind"] == "sweep")
        p = c.export / job["path"] / "metrics.json"
        if mutation == "missing":
            p.unlink()
        else:
            d = load_json(p)
            d["n_total"] -= 1
            write_json_atomic(p, d)
    resign(c.directory)
    with pytest.raises((PingstoreError, OSError, ValueError)):
        analyse.analyse(cid)
    assert not list((root / ".pingstore/runs").glob("*-analyse"))


def test_inputs_reject_changed_upstream_manifest(lab):
    root, bank, _ = lab
    cid = compute.compute(bank)
    p = root / ".pingstore/runs" / bank / "run.json"
    d = load_json(p)
    d["execution"]["extra"] = "changed"
    write_json_atomic(p, d)
    with pytest.raises(PingstoreError):
        analyse.analyse(cid)


def test_failed_simulation_never_completes(lab, monkeypatch):
    root, bank, _ = lab

    def fail(*a, **k):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(compute, "run_cli", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        compute.compute(bank)
    assert not list((root / ".pingstore/runs").glob("exp037-*-compute"))
    assert list((root / ".pingstore/runs").glob(".exp037-*-compute.tmp"))


def test_collection_reserves_dispatches_and_resumes(lab, monkeypatch):
    root, bank, _ = lab
    manifest = root / "bank.json"
    write_json_atomic(manifest, {"pingstore_run_id": bank})
    row = {
        "slug": "exp037",
        "execution": {"mode": "exp037-staged"},
        "paths": {"state": str(root / "campaign/state")},
        "required_outputs": [str(root / "campaign/state/stage-refs.json")],
    }
    plan = {"profile": "smoke", "exp022_manifest": str(manifest)}
    reservations = collection.reserve(root, row)
    assert all(value.endswith("-" + stage) for stage, value in reservations.items())
    calls = []

    def dispatch(command, **kwargs):
        calls.append(command)
        stage = command[2].rsplit(".", 1)[1]
        method = {
            "compute": compute.compute,
            "analyse": analyse.analyse,
            "present": present.present,
        }[stage]
        method(
            command[command.index("--source") + 1],
            run_id=command[command.index("--run-id") + 1],
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(collection.subprocess, "run", dispatch)
    refs = collection.execute(root, plan, row)
    assert len(calls) == 3
    assert set(refs) == {"bank", "compute", "analyse", "present"}
    assert (
        collection.completed(root, plan, row).record["run_id"]
        == reservations["present"]
    )
    collection.execute(root, plan, row)
    assert len(calls) == 3
    with pytest.raises(PingstoreError):
        collection.require_staged({"execution": {"mode": "monolithic"}})


def test_retired_entrypoints_and_import_side_effects(tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = "from experiments import exp037; assert exp037.CHECKPOINT_ROLE == 'best_validation'"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())
    result = subprocess.run(
        [sys.executable, "-m", "experiments.exp037"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "explicit" in result.stderr


def test_v2_and_wrong_stage_inputs_are_rejected(lab):
    root, bank_id, _ = lab
    cid = compute.compute(bank_id)
    with pytest.raises(PingstoreError, match="not a"):
        present.present(cid)
    path = root / ".pingstore/runs" / cid / "run.json"
    record = load_json(path)
    record["schema"] = "pingstore.run/v2"
    write_json_atomic(path, record)
    with pytest.raises(PingstoreError):
        analyse.analyse(cid)


def test_source_change_during_compute_prevents_completion(lab, monkeypatch):
    root, bank_id, _ = lab
    original = compute.run_cli

    def simulate(args, **kwargs):
        original(args, **kwargs)
        path = root / ".pingstore/runs" / bank_id / "README.md"
        path.write_text("changed while running\n")

    monkeypatch.setattr(compute, "run_cli", simulate)
    with pytest.raises(PingstoreError):
        compute.compute(bank_id)
    assert not list((root / ".pingstore/runs").glob("exp037-*-compute"))


def test_present_rejects_resigned_incomplete_analysis(lab):
    root, bank_id, _ = lab
    cid = compute.compute(bank_id)
    aid = analyse.analyse(cid)
    source = inputs.source(root, aid, "analyse")
    path = source.export / "results.json"
    result = load_json(path)
    result["perturbation"].pop()
    write_json_atomic(path, result)
    resign(source.directory)
    with pytest.raises(PingstoreError, match="incomplete"):
        present.present(aid)
    assert not list((root / ".pingstore/runs").glob("*-present"))


def test_article_renders_only_selected_presentation(lab):
    import re
    import shutil

    from demolab_cli import _paths

    root, bank_id, _ = lab
    cid = compute.compute(bank_id)
    aid = analyse.analyse(cid)
    pid = present.present(aid)
    output = inputs.source(root, pid, "present")
    source_root = Path(__file__).resolve().parents[2]
    shutil.copytree(source_root / "writings", root / "writings")
    (root / ".demolab").mkdir()
    shutil.copy2(_paths.TYP / "lib.typ", root / ".demolab/lib.typ")
    write_json_atomic(
        root / "preview.json",
        {"exp037": {"exp037": "/" + str(output.export.relative_to(root))}},
    )
    document = root / "document.typ"
    document.write_text(
        '#set page(paper: "a4", margin: 18mm)\n#set text(size: 10pt)\n#import "writings/exp037.typ": body\n#body\n'
    )
    command = [
        _paths.find_typst(source_root),
        "compile",
        "--root",
        str(root),
        "--input",
        "demolab-preview-file=/preview.json",
        "--format",
        "png",
        "--ppi",
        "80",
        str(document),
        str(root / "article-{p}.png"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert list(root.glob("article-*.png"))
    html_command = [
        _paths.find_typst(source_root),
        "compile",
        "--features",
        "html",
        "--format",
        "html",
        "--root",
        str(root),
        "--input",
        "demolab-preview-file=/preview.json",
        str(document),
        str(root / "article.html"),
    ]
    result = subprocess.run(html_command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    html = (root / "article.html").read_text()
    images = re.findall(r"<img\b[^>]*>", html)
    assert len(images) == 5
    assert all('alt="' in tag and 'src="' in tag for tag in images)
    assert len(re.findall(r"<figcaption\b", html)) == 5
    # Older v3 presentations lack the optional image-label projection.
    numbers = load_json(output.export / "numbers.json")
    numbers.pop("illustrative_labels")
    write_json_atomic(output.export / "numbers.json", numbers)
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    (output.export / "numbers.json").write_text("broken JSON")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0


def test_independent_stages_preserve_measurements_and_never_publish(lab, monkeypatch):
    root, bank, calls = lab
    cid = compute.compute(bank)
    assert len(calls) == 54
    source = inputs.source(root, cid, "compute")
    for path in source.export.rglob("snapshot.npz"):
        with np.load(path) as data:
            assert set(data.files) == set(recipe.SNAPSHOT_ARRAYS)
    monkeypatch.setattr(
        compute, "run_cli", lambda *a, **k: pytest.fail("downstream simulation")
    )
    aid = analyse.analyse(cid)
    analysis = inputs.source(root, aid, "analyse")
    result = load_json(analysis.export / "results.json")
    assert len(result["perturbation"]) == 42
    assert len(result["baseline_results"]) == 36
    assert {r["rate_e"] for r in result["baseline_results"]} == {25.0}
    assert result["plot_data"]["baseline_e_rate_hz"] == {"coba": 25.0, "ping": 25.0}
    assert {r["role"] for r in result["checkpoint_provenance"]} == {"best_validation"}
    for name in (
        "raster",
        "plot_data",
        "baseline_rows",
        "summarize_accuracy",
        "summarize_perturbation_rows",
    ):
        monkeypatch.setattr(
            measurements, name, lambda *a, **k: pytest.fail("presentation measurement")
        )
    pid = present.present(aid)
    output = inputs.source(root, pid, "present")
    assert {f.name for f in output.export.iterdir()} == {
        "numbers.json",
        "_manifest.json",
        *recipe.FIGURES,
    }
    assert output.record["inputs"] == {"analysis": analysis.reference}
    assert not (root / ".artifacts").exists()
    assert load_json(output.export / "numbers.json")["illustrative_labels"] == [7] * 12


def test_recipe_preserves_production_and_smoke_grids():
    for smoke, total, sweeps in ((False, 204, 192), (True, 54, 42)):
        jobs = recipe.jobs(recipe.configuration(smoke=smoke))
        assert len(jobs) == total
        assert len({j["id"] for j in jobs}) == total
        assert len({j["path"] for j in jobs}) == total
        assert sum(j["kind"] == "sweep" for j in jobs) == sweeps
        assert {j["seed"] for j in jobs if j["kind"] == "sweep"} == {42, 43, 44}
        assert {j["sample_index"] for j in jobs if j["kind"] == "raster"} == {0}
    assert recipe.SHARDS == 6


def test_raster_selection_preserves_dtype_sum_and_rng(tmp_path):
    e = (np.arange(20 * 256).reshape(20, 256) % 3 == 0).astype(np.float32)
    i = (np.arange(20 * 128).reshape(20, 128) % 5 == 0).astype(np.float32)
    np.savez(
        tmp_path / "snapshot.npz", spk_e=e[:, None, :], spk_i=i[:, None, :], label=7
    )
    job = {"model": "ping", "seed": 42, "mode": "drop", "level": 0.5}
    result = measurements.raster(tmp_path, {"dt": 0.1, "t_ms": 2.0}, job)
    rng = np.random.default_rng(0)
    ei = np.sort(rng.choice(256, 200, replace=False))
    ii = np.sort(rng.choice(128, 64, replace=False))
    et, en = np.where(e[:, ei].astype(bool))
    it, inn = np.where(i[:, ii].astype(bool))
    for key, expected in (
        ("e_t", et * 0.1),
        ("e_n", en),
        ("i_t", it * 0.1),
        ("i_n", inn + 206),
    ):
        np.testing.assert_array_equal(result[key], expected)
    assert result["e_rate_hz"] == float(e.sum() / (256 * 0.002))


def test_shards_collect_without_reexecuting_and_resume_verified_work(lab, monkeypatch):
    root, bank, calls = lab
    monkeypatch.setattr(
        compute, "_capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    rid = stages.reserve_stage(root / ".pingstore", "exp037", "compute")
    for index in range(6):
        compute.shard(bank, run_id=rid, index=index)
    assert len(calls) == 54
    compute.shard(bank, run_id=rid, index=0)
    assert len(calls) == 54
    compute.compute(bank, run_id=rid, collect=True)
    assert len(calls) == 54
    source = inputs.source(root, rid, "compute")
    assert (
        len(list((source.directory / "provenance/shards").glob("*/completed.json")))
        == 6
    )
    with pytest.raises(PingstoreError):
        compute.shard(bank, run_id=rid, index=0)


@pytest.mark.parametrize("fault", ["payload", "attachment", "bank", "profile"])
def test_shard_resume_rejects_changed_evidence(lab, monkeypatch, fault):
    root, bank, _ = lab
    monkeypatch.setattr(
        compute, "_capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    rid = stages.reserve_stage(root / ".pingstore", "exp037", "compute")
    compute.shard(bank, run_id=rid, index=0)
    directory = root / ".pingstore/runs" / f".{rid}.tmp"
    if fault == "profile":
        monkeypatch.setenv("PINGLAB_SMOKE", "0")
    elif fault == "bank":
        (root / ".pingstore/runs" / bank / "README.md").write_text("changed")
    else:
        path = next(
            (
                directory
                / ("export" if fault == "payload" else "provenance/simulations")
            ).rglob("*.json")
        )
        path.write_text("{}")
    with pytest.raises(PingstoreError):
        compute.shard(bank, run_id=rid, index=0)
    assert not (root / ".pingstore/runs" / rid).exists()


def test_collect_rejects_missing_shards_and_busy_reservation(lab, monkeypatch):
    root, bank, _ = lab
    rid = stages.reserve_stage(root / ".pingstore", "exp037", "compute")
    with pytest.raises((OSError, PingstoreError)):
        compute.compute(bank, run_id=rid, collect=True)
    directory = root / ".pingstore/runs" / f".{rid}.tmp"
    with compute._compute_lock(directory, exclusive=False):
        with pytest.raises(PingstoreError, match="busy"):
            compute.compute(bank, run_id=rid)


def test_collection_keeps_six_staged_compute_shards(tmp_path):
    from experiments.collections.gamma_gated_sparsity.plan import build_plan
    from experiments.collections.gamma_gated_sparsity.workloads import jobs_for_shard

    plan = build_plan(tmp_path / "campaign", "fixture")
    row = next(
        row
        for stage in plan["stages"]
        for row in stage["experiments"]
        if row["slug"] == "exp037"
    )
    assert row["execution"]["mode"] == "exp037-staged"
    assert row["execution"]["shards"] == 6
    assert row["execution"]["stages"] == ["compute", "analyse", "present"]
    assert row["command"] == []
    shards = [jobs_for_shard("exp037", index, 6) for index in range(6)]
    assert sum(map(len, shards)) == 204
    assert len(set().union(*map(set, shards))) == 204


def test_collection_dispatches_shards_with_bank_and_reservation(lab, monkeypatch):
    root, bank, _ = lab
    manifest = root / "bank.json"
    write_json_atomic(manifest, {"pingstore_run_id": bank})
    row = {
        "slug": "exp037",
        "execution": {"mode": "exp037-staged"},
        "paths": {"state": str(root / "campaign/state")},
        "required_outputs": [str(root / "campaign/state/stage-refs.json")],
    }
    plan = {"profile": "smoke", "exp022_manifest": str(manifest)}
    reservations = collection.reserve(root, row)
    calls = []
    monkeypatch.setattr(
        collection.subprocess, "run", lambda command, **kw: calls.append((command, kw))
    )
    result = collection.execute_shard(root, plan, row, 2, 6)
    command, kwargs = calls[0]
    assert command[2] == "experiments.exp037.compute"
    assert command[command.index("--source") + 1] == bank
    assert command[command.index("--run-id") + 1] == reservations["compute"]
    assert command[command.index("--shard-index") + 1] == "2"
    assert kwargs["env"]["PINGLAB_SMOKE"] == "1"
    assert result["compute_run_id"] == reservations["compute"]
    with pytest.raises(PingstoreError):
        collection.execute_shard(root, plan, row, 0, 5)


def test_reviewed_figures_keep_coordinates_show_full_range_and_omit_run_ids(
    tmp_path, monkeypatch
):
    from experiments.exp037 import plots

    captured = []
    monkeypatch.setattr(plots, "save_figure", lambda fig, *a, **k: captured.append(fig))
    data = {"use_pct": True, "panels": {}}
    for mode in ("drop", "add"):
        data["panels"][mode] = {
            model: {
                "x": [0, 80, 100] if mode == "drop" else [0, 100, 201.426],
                "mean": [90, 89, 10.6],
                "lo": [89, 88, 10],
                "hi": [91, 90, 11.2],
            }
            for model in recipe.MODELS
        }
    plots.plot_perturbation_curves(data, tmp_path / "curves", "exp037-r999-present")
    figure = captured[0]
    assert not figure.texts
    for axis, mode in zip(figure.axes, ("drop", "add")):
        for line, model in zip(axis.lines[:2], recipe.MODELS):
            np.testing.assert_array_equal(
                line.get_xdata(), data["panels"][mode][model]["x"]
            )
            np.testing.assert_array_equal(
                line.get_ydata(), data["panels"][mode][model]["mean"]
            )
    assert figure.axes[1].get_xlim()[1] > 201.426
    assert "reference E rate" in figure.axes[1].get_xlabel()
    assert "Poisson" not in figure.axes[1].get_title(loc="left")
    assert "probability" in figure.axes[0].get_xlabel()


def test_reviewed_article_structure_and_scientific_caveats():
    text = (Path(__file__).resolve().parents[2] / "writings/exp037.typ").read_text()
    assert 'date: "2026-05-30"' in text
    assert 'updated_at: "2026-08-28"' in text
    assert (
        text.index("== Abstract") < text.index("== Results") < text.index("== Methods")
    )
    assert "== Discussion" not in text
    assert "minimum-validation-loss epoch" in text
    assert "not test-set baseline rates" in text
    assert "does not match relative perturbation doses" in text
    assert "capped at one" in text
    assert "digit 0" not in text
    assert 'fit: "contain"' in text
    assert "#reference-list" in text and "#cite(1)" in text
