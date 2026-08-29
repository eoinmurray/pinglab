"""Synthetic staged probes; no production simulations or historical imports."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from experiments.exp038 import (
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
        stages, "memberships", lambda _: {"exp022": "demo", "exp038": "demo"}
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

        assert value("--device") == "auto"
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
            write_json_atomic(
                out / "metrics.json",
                {
                    "config": {
                        **train,
                        "ei_strength": strength,
                        "evaluation_partition": "official_mnist_test",
                        "evaluation_samples": n,
                    },
                    "best_acc": 90.0,
                    "n_correct": n * 9 // 10,
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


def test_independent_stages_preserve_roles_and_never_publish(lab, monkeypatch):
    root, bank_id, calls = lab
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    cid = compute.compute(bank_id)
    assert len(calls) == 20
    c = inputs.source(root, cid, "compute")
    for p in c.export.rglob("snapshot.npz"):
        with np.load(p) as d:
            assert set(d.files) == {"dt", "n_e", "n_i", "label", "spk_e", "spk_i"}
    monkeypatch.setattr(
        compute, "run_cli", lambda *a, **k: pytest.fail("downstream simulation")
    )
    aid = analyse.analyse(cid)
    a = inputs.source(root, aid, "analyse")
    result = load_json(a.export / "results.json")
    assert len(result["baseline_results"]) == 36
    assert {r["rate_e"] for r in result["baseline_results"]} == {25.0}
    assert {r["epoch"] for r in result["checkpoint_provenance"]} == {43}
    assert {r["role"] for r in result["checkpoint_provenance"]} == {"best_validation"}
    monkeypatch.setattr(
        measurements,
        "summarize_ei_points",
        lambda *a: pytest.fail("presentation aggregation"),
    )
    monkeypatch.setattr(
        measurements, "raster", lambda *a: pytest.fail("presentation measurement")
    )
    pid = present.present(aid)
    p = inputs.source(root, pid, "present")
    labels = load_json(p.export / "numbers.json")["illustrative_labels"]
    assert labels == {"rate_rasters": [7, 7, 7], "ei_rasters": [7, 7]}
    assert {f.name for f in p.export.iterdir()} == {
        "numbers.json",
        "_manifest.json",
        *recipe.FIGURES,
    }
    assert p.record["inputs"] == {"analysis": a.reference}
    assert not (root / ".artifacts").exists()
    assert not (root / ".pingstore/runs" / f".{pid}.tmp").exists()
    bank.check_unchanged()


def test_recipe_retains_full_production_grid():
    jobs = recipe.jobs(recipe.configuration())
    assert len(jobs) == 101
    assert {
        kind: sum(j["kind"] == kind for j in jobs)
        for kind in ("rate_raster", "fi_uniform", "ei_sweep", "ei_raster")
    } == {"rate_raster": 10, "fi_uniform": 52, "ei_sweep": 33, "ei_raster": 6}
    assert len({j["path"] for j in jobs}) == 101
    assert {j["samples"] for j in jobs if "samples" in j} == {1000}
    assert {j["trials"] for j in jobs if "trials" in j} == {32}
    assert (
        recipe.configuration()["rate_rasters"] == np.linspace(0, 100, 40)[:10].tolist()
    )


def test_snapshots_preserve_full_population_rate_and_rng_selection(tmp_path):
    e = np.zeros((20, 256), bool)
    i = np.zeros((20, 128), bool)
    e[::2, :] = True
    i[::4, :] = True
    np.savez(
        tmp_path / "snapshot.npz", spk_e=e[:, None, :], spk_i=i[:, None, :], label=7
    )
    result = measurements.raster(
        tmp_path, {"dt": 0.1, "t_ms": 2.0}, {"kind": "rate_raster", "input_rate": 10.0}
    )
    rng = np.random.default_rng(0)
    ei = np.sort(rng.choice(256, 200, replace=False))
    ii = np.sort(rng.choice(128, 64, replace=False))
    np.testing.assert_array_equal(result["e"], e[:, ei])
    np.testing.assert_array_equal(result["i"], i[:, ii])
    assert result["e_rate_hz"] == 5000.0
    assert result["i_rate_hz"] == 2500.0


@pytest.mark.parametrize("mutation", ["sample_count", "snapshot", "config", "missing"])
def test_analyse_rejects_corrupt_even_resigned_payload(lab, mutation):
    root, bank, _ = lab
    cid = compute.compute(bank)
    c = inputs.source(root, cid, "compute")
    cfg = recipe.configuration(smoke=True)
    jobs = recipe.jobs(cfg)
    if mutation == "snapshot":
        p = c.export / jobs[0]["path"] / "snapshot.npz"
        with np.load(p) as raw:
            data = {k: raw[k] for k in raw.files}
        data["spk_e"] = np.full_like(data["spk_e"], 2, dtype=np.int8)
        np.savez_compressed(p, **data)
    elif mutation == "config":
        p = c.directory / "export/evidence/simulations" / jobs[0]["path"] / "config.json"
        d = load_json(p)
        d["spike_rate"] = 111
        write_json_atomic(p, d)
    else:
        job = next(j for j in jobs if j["kind"] == "ei_sweep")
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


def test_inputs_allow_upstream_metadata_amendment(lab):
    root, bank, _ = lab
    cid = compute.compute(bank)
    p = root / ".pingstore/runs" / bank / "run.json"
    d = load_json(p)
    d["execution"]["extra"] = "changed"
    write_json_atomic(p, d)
    assert analyse.analyse(cid).endswith("-analyse")


def test_failed_simulation_never_completes(lab, monkeypatch):
    root, bank, _ = lab

    def fail(*a, **k):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(compute, "run_cli", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        compute.compute(bank)
    assert not list((root / ".pingstore/runs").glob("exp038-*-compute"))
    assert list((root / ".pingstore/runs").glob(".exp038-*-compute.tmp"))


def test_collection_reserves_dispatches_and_resumes(lab, monkeypatch):
    root, bank, _ = lab
    manifest = root / "bank.json"
    write_json_atomic(manifest, {"pingstore_run_id": bank})
    row = {
        "slug": "exp038",
        "execution": {"mode": "exp038-staged"},
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
    code = "from experiments import exp038; assert exp038.CHECKPOINT_ROLE == 'best_validation'"
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
        [sys.executable, "-m", "experiments.exp038"],
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


def test_source_readme_change_during_compute_is_allowed(lab, monkeypatch):
    root, bank_id, _ = lab
    original = compute.run_cli

    def simulate(args, **kwargs):
        original(args, **kwargs)
        path = root / ".pingstore/runs" / bank_id / "README.md"
        path.write_text("changed while running\n")

    monkeypatch.setattr(compute, "run_cli", simulate)
    identity = compute.compute(bank_id)
    assert (root / ".pingstore/runs" / identity).is_dir()


def test_present_rejects_resigned_incomplete_analysis(lab):
    root, bank_id, _ = lab
    cid = compute.compute(bank_id)
    aid = analyse.analyse(cid)
    source = inputs.source(root, aid, "analyse")
    path = source.export / "results.json"
    result = load_json(path)
    result["ei_sweep"].pop()
    write_json_atomic(path, result)
    resign(source.directory)
    with pytest.raises(PingstoreError, match="incomplete"):
        present.present(aid)
    assert not list((root / ".pingstore/runs").glob("*-present"))


def test_article_renders_only_selected_presentation(lab):
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
        {"exp038": {"exp038": "/" + str(output.export.relative_to(root))}},
    )
    document = root / "document.typ"
    document.write_text(
        '#set page(paper: "a4", margin: 18mm)\n#set text(size: 10pt)\n#import "writings/exp038.typ": body\n#body\n'
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
    # Older v3 presentations lack the optional image-label projection.
    numbers = load_json(output.export / "numbers.json")
    numbers.pop("illustrative_labels")
    write_json_atomic(output.export / "numbers.json", numbers)
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    (output.export / "numbers.json").write_text("broken JSON")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0


def test_raster_labels_use_recorded_class_and_do_not_overlap(tmp_path, monkeypatch):
    from experiments.exp038 import plots

    samples = [
        {
            "e": np.zeros((20, 200), bool),
            "i": np.zeros((20, 64), bool),
            "dt": 0.1,
            "t_ms": 2.0,
            "spike_rate": float(rate),
            "e_rate_hz": 10.0,
            "i_rate_hz": 45.0,
            "label": 7,
        }
        for rate in range(10)
    ]
    figures = []
    monkeypatch.setattr(plots, "save_figure", lambda fig, *a, **k: figures.append(fig))
    plots.plot_rate_rasters(samples, tmp_path / "rasters", "fixture")
    plots.plot_fi_curve(samples, tmp_path / "curve", "fixture")
    raster, curve = figures
    raster.canvas.draw()
    renderer = raster.canvas.get_renderer()
    boxes = [ax.texts[0].get_window_extent(renderer) for ax in raster.axes]
    assert all(a.y0 > b.y1 for a, b in zip(boxes, boxes[1:]))
    assert all(box.x1 < raster.bbox.x1 and box.y0 > 0 for box in boxes)
    assert "label 7" in raster.axes[0].get_title(loc="left")
    assert "label 7" in curve._suptitle.get_text()
