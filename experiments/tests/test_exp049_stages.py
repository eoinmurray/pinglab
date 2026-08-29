"""Synthetic staged probes; no production simulations or historical imports."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from experiments.exp049 import (
    analyse,
    collection,
    compute,
    inputs,
    measurements,
    plots,
    present,
    recipe,
)
from experiments.tests.test_exp044_provenance import _common_config
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
    monkeypatch.setattr(recipe, "N_E", 200)
    monkeypatch.setattr(recipe, "N_I", 64)
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(
        stages, "memberships", lambda _: {"exp022": "demo", "exp049": "demo"}
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
                "ei_strength": {
                    "frozen_ping": 1.0,
                    "trainable_ping_init": 1.0,
                    "trainable_zero_init": 0.0,
                    "trainable_small_init": 0.1,
                }[cell["condition"]],
                "trainable_w_ei": cell["condition"] != "frozen_ping",
                "trainable_w_ie": cell["condition"] != "frozen_ping",
                "ei_ratio": 2.0,
            }
            write_json_atomic(directory / "config.json", cfg)
            cps = {}
            for role, filename, epoch in (
                ("best_validation", "weights.pth", 43),
                ("final_epoch", "weights_final.pth", 50),
            ):
                target = directory / filename
                import torch

                torch.save(
                    {
                        "W_ei.1": torch.full((200, 64), 0.002),
                        "W_ie.1": torch.full((64, 200), 0.002),
                    },
                    target,
                )
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
                            "rate_i": ep / 5,
                            "contrast": 0.7
                            if cell["condition"] == "frozen_ping"
                            else 0.1,
                            "weight_norms": {}
                            if cell["condition"] == "frozen_ping"
                            else {"W_ei.1": ep * 0.01, "W_ie.1": ep * 0.02},
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

        assert Path(value("--load-weights")).name == "weights_final.pth"
        out = Path(value("--out-dir"))
        out.mkdir(parents=True)
        train = load_json(Path(value("--load-config")))
        dump = args[0] == "dump-weights"
        strength = train["ei_strength"]
        rate = train["input_rate"]
        cfg = {
            **train,
            "load_config": value("--load-config"),
            "load_weights": value("--load-weights"),
            "input": "dataset",
            "mode": "dump-weights" if dump else "sim",
            "infer": None if dump else True,
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
        if dump:
            arrays = {
                key: np.full(
                    (200, 64) if key.startswith("W_ei") else (64, 200),
                    0.001 if key.endswith("init") else 0.002,
                    dtype=np.float32,
                )
                for key in recipe.WEIGHT_ARRAYS
            }
            arrays["unused_input_matrix"] = np.zeros((784, 200))
            save(out / "weights_dump.npz", **arrays)
        elif "--sample-index" in args:
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
        else:
            n = cfg["max_samples"]
            pop = np.tile(
                (20 + 10 * np.sin(2 * np.pi * 60 * np.arange(2000) * 0.0001)).astype(
                    np.float32
                ),
                (n, 1),
            )
            save(out / "pop_traces.npz", dt=np.float32(0.1), pop_e=pop, pop_i=pop)
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
    assert not list((root / ".pingstore/runs").glob("exp049-*-compute"))
    assert list((root / ".pingstore/runs").glob(".exp049-*-compute.tmp"))


def test_collection_reserves_dispatches_and_resumes(lab, monkeypatch):
    root, bank, _ = lab
    manifest = root / "bank.json"
    write_json_atomic(manifest, {"pingstore_run_id": bank})
    row = {
        "slug": "exp049",
        "execution": {"mode": "exp049-staged"},
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
    code = (
        "from experiments import exp049; assert exp049.CHECKPOINT_ROLE == 'final_epoch'"
    )
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
        [sys.executable, "-m", "experiments.exp049"],
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
    assert not list((root / ".pingstore/runs").glob("exp049-*-compute"))


def test_present_rejects_resigned_incomplete_analysis(lab):
    root, bank_id, _ = lab
    cid = compute.compute(bank_id)
    aid = analyse.analyse(cid)
    source = inputs.source(root, aid, "analyse")
    path = source.export / "results.json"
    result = load_json(path)
    result["summary"].pop()
    write_json_atomic(path, result)
    resign(source.directory)
    with pytest.raises(PingstoreError, match="incomplete"):
        present.present(aid)
    assert not list((root / ".pingstore/runs").glob("*-present"))


def test_independent_stages_preserve_roles_and_never_publish(lab, monkeypatch):
    root, bank_id, calls = lab
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    cid = compute.compute(bank_id)
    assert len(calls) == 28
    c = inputs.source(root, cid, "compute")
    for filename, keys in recipe.ARRAYS.items():
        for path in c.export.rglob(filename):
            with np.load(path) as data:
                assert set(data.files) == set(keys)
    monkeypatch.setattr(
        compute, "run_cli", lambda *a, **k: pytest.fail("downstream simulation")
    )
    aid = analyse.analyse(cid)
    a = inputs.source(root, aid, "analyse")
    result = load_json(a.export / "results.json")
    assert len(result["summary"]) == 12
    assert {r["f_gamma_hz"] for r in result["summary"]} == {60.0}
    assert {r["epoch"] for r in result["checkpoint_provenance"]} == {50}
    assert result["plot_data"]["cards"]["frozen_ping"]["curves"]["rate_e"]["last"] == 25
    assert result["epoch_curves"]["frozen_ping__seed42"]["rate_e"][-1] == 50 / 3
    for name in ("endpoint", "trajectories", "raster", "card", "weight_distributions"):
        monkeypatch.setattr(
            measurements, name, lambda *a, **k: pytest.fail("presentation measurement")
        )
    save_figure = plots.save_figure

    def check_card_labels(fig, path, **kwargs):
        if path.name.startswith("card__"):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for ax in fig.axes:
                for text in ax.texts:
                    if text.get_text().startswith("mean peak"):
                        bounds = text.get_window_extent(renderer)
                        assert ax.bbox.contains(bounds.x0, bounds.y0)
                        assert ax.bbox.contains(bounds.x1, bounds.y1)
        save_figure(fig, path, **kwargs)

    monkeypatch.setattr(plots, "save_figure", check_card_labels)
    pid = present.present(aid)
    p = inputs.source(root, pid, "present")
    assert {f.name for f in p.export.iterdir()} == {
        "numbers.json",
        "_manifest.json",
        *recipe.FIGURES,
    }
    assert p.record["inputs"] == {"analysis": a.reference}
    assert not (root / ".artifacts").exists()
    bank.check_unchanged()


@pytest.mark.parametrize(
    "fault", ["samples", "weights", "snapshot", "config", "trace", "missing"]
)
def test_invalid_resigned_compute_is_rejected(lab, fault):
    root, bank, _ = lab
    cid = compute.compute(bank)
    c = inputs.source(root, cid, "compute")
    kind = {"weights": "weights_dump", "snapshot": "snapshot"}.get(fault, "infer")
    job = next(
        j for j in recipe.jobs(recipe.configuration(smoke=True)) if j["kind"] == kind
    )
    directory = c.export / job["path"]
    if fault == "config":
        path = c.directory / "provenance/simulations" / job["path"] / "config.json"
        cfg = load_json(path)
        cfg["load_weights"] = cfg["load_weights"].replace(
            "weights_final.pth", "weights.pth"
        )
        write_json_atomic(path, cfg)
    elif fault == "samples":
        path = directory / "metrics.json"
        m = load_json(path)
        m["n_total"] -= 1
        write_json_atomic(path, m)
    elif fault == "missing":
        (directory / "pop_traces.npz").unlink()
    else:
        filename, key = {
            "weights": ("weights_dump.npz", "W_ei_1_trained"),
            "snapshot": ("snapshot.npz", "spk_e"),
            "trace": ("pop_traces.npz", "pop_e"),
        }[fault]
        path = directory / filename
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        arrays[key] = np.full_like(arrays[key], -1, dtype=np.float32)
        np.savez(path, **arrays)
    resign(c.directory)
    with pytest.raises((PingstoreError, OSError, ValueError)):
        analyse.analyse(cid)
    assert not list((root / ".pingstore/runs").glob("*-analyse"))


def test_degenerate_psd_and_history_null_semantics():
    arrays = (np.zeros((2, 2), np.float32),) * 4
    endpoint = measurements.endpoint(
        {"dt": 0.1},
        {"best_acc": 80, "rates_hz": {"hid": 0, "inh": 0}},
        np.zeros((3, 2000), np.float32),
        arrays,
    )
    assert measurements.clean(endpoint)["f_gamma_hz"] is None
    rows = [
        {
            "ep": 1,
            "acc": 80,
            "rate_e": 7,
            "rate_i": 10,
            "test_rate_e": None,
            "contrast": None,
        },
        {"ep": 2, "acc": 90, "rate_e": 9, "rate_i": 11, "contrast": 0.2},
    ]
    curve = measurements.epoch_curve(rows, "trainable_ping_init")
    assert curve["rate_e"] == [None, 9]
    assert (
        "rate_e"
        not in measurements.trajectories({"x": curve})["trainable_ping_init"]["panels"]
    )
    assert measurements.rhythmicity({"x": curve})["epoch1_contrast_trainable"] == 0.2


def test_trajectory_labels_and_limits_preserve_saved_measurements(
    tmp_path, monkeypatch
):
    from matplotlib.collections import LineCollection

    curves = {
        cond: measurements.epoch_curve(
            [
                {
                    "ep": ep,
                    "acc": 80 + ep,
                    "rate_e": 7,
                    "rate_i": 8,
                    "test_rate_e": 120 + ep,
                    "test_rate_i": 160 + ep,
                    "contrast": 0.9 if cond == "frozen_ping" else 0.1,
                }
                for ep in range(1, 4)
            ],
            cond,
        )
        for cond in recipe.COND_ORDER
    }
    data = {"trajectories": measurements.trajectories(curves), "last_epoch": 3}
    figures = []
    monkeypatch.setattr(plots, "save_figure", lambda fig, *a, **k: figures.append(fig))
    plots.fig_training_curves(data, tmp_path / "training", "fixture")
    axes = figures[-1].axes
    assert axes[0].get_ylabel() == "Validation accuracy (%)"
    assert axes[3].get_ylabel() == "Reference contrast R"
    for ax, key in zip(axes[1:3], ("rate_e", "rate_i"), strict=True):
        for line, cond in zip(ax.lines, recipe.COND_ORDER, strict=True):
            saved = data["trajectories"][cond]["panels"][key]
            np.testing.assert_array_equal(line.get_ydata(), saved["mean"])
            assert ax.get_ylim()[1] >= max(saved["hi"])

    for render in (plots.fig_phase_portrait, plots.fig_acc_rate_trajectory):
        render(data, tmp_path / render.__name__, "fixture")
        ax = figures[-1].axes[0]
        assert ax.get_xlabel() == "Validation E rate (Hz)"
        assert not ax.lines  # No inferred rate boundary.
        assert not ax.patches  # No shaded basin.
        assert any("epoch 1" in text.get_text() for text in ax.texts)
        assert all("epoch 0" not in text.get_text() for text in ax.texts)
        segments = [c for c in ax.collections if isinstance(c, LineCollection)]
        conditions = recipe.COND_ORDER
        if render is plots.fig_phase_portrait:
            conditions = [c for c in conditions if c != "frozen_ping"]
        for line, cond in zip(segments, conditions, strict=True):
            saved = data["trajectories"][cond]["phase"]
            key = "p" if render is plots.fig_phase_portrait else "a"
            points = np.column_stack((saved["e"], saved[key]))
            np.testing.assert_array_equal(
                line.get_segments(), np.stack((points[:-1], points[1:]), axis=1)
            )


def test_production_recipe_and_raster_selection(tmp_path):
    jobs = recipe.jobs(recipe.configuration())
    assert len(jobs) == 28 and len({j["path"] for j in jobs}) == 28
    assert {j["samples"] for j in jobs if j["kind"] == "infer"} == {1000}
    e = np.arange(20 * 256).reshape(20, 256) % 3 == 0
    i = np.arange(20 * 128).reshape(20, 128) % 5 == 0
    np.savez(
        tmp_path / "snapshot.npz", spk_e=e[:, None, :], spk_i=i[:, None, :], label=7
    )
    data = measurements.raster(tmp_path, {"dt": 0.1, "t_ms": 2.0})
    rng = np.random.default_rng(42)
    ei = np.sort(rng.choice(256, 200, replace=False))
    ii = np.sort(rng.choice(128, 50, replace=False))
    et, en = np.where(e[:, ei])
    it, inn = np.where(i[:, ii])
    for key, values in (
        ("e_t", et * 0.1),
        ("e_n", en),
        ("i_t", it * 0.1),
        ("i_n", inn),
    ):
        np.testing.assert_array_equal(data[key], values)


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
        {"exp049": {"exp049": "/" + str(output.export.relative_to(root))}},
    )
    document = root / "document.typ"
    document.write_text(
        '#set page(paper: "a4", margin: 18mm)\n#set text(size: 10pt)\n#import "writings/exp049.typ": body\n#body\n'
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
    (output.export / "numbers.json").write_text("broken JSON")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0


@pytest.mark.parametrize("fault", ["training_recipe", "history", "final_epoch"])
def test_bank_contract_rejects_resigned_mismatches(lab, fault):
    root, bank_id, calls = lab
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    directory = bank.export / recipe.cell_name("frozen_ping", 42)
    path = directory / ("config.json" if fault == "training_recipe" else "metrics.json")
    record = load_json(path)
    if fault == "training_recipe":
        record["hidden_sizes"] = [999]
    elif fault == "history":
        record["epochs"].pop()
    else:
        record["checkpoints"]["final_epoch"]["epoch"] = 49
    write_json_atomic(path, record)
    resign(bank.directory)
    with pytest.raises((PingstoreError, RuntimeError)):
        compute.compute(bank_id)
    assert not calls
    assert not list((root / ".pingstore/runs").glob("exp049-*-compute"))
