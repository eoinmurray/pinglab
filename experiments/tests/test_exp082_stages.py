"""Synthetic contract tests; never use the workspace store or real inference."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from experiments.collections.gamma_gated_sparsity import execution, plan, workloads
from experiments.exp082 import (
    analyse,
    collection,
    compute,
    evidence,
    historical,
    import_gold2,
    inference,
    inputs,
    measurements,
    present,
    recipe,
)
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
        stages, "memberships", lambda _: {"exp022": "demo", "exp082": "demo"}
    )

    def code(*args):
        return {"git_commit": "fixture", "dirty": False, "code_dirty": False}

    monkeypatch.setattr(stages, "_capture_code", code)
    monkeypatch.setattr(compute, "_capture_code", code)
    monkeypatch.setenv("PINGLAB_SMOKE", "1")
    for key in ("STREAMS_PER_CELL", "DIGITS_PER_STREAM", "STREAM_BATCH_SIZE"):
        monkeypatch.delenv("PINGLAB_EXP082_" + key, raising=False)
    with stages.stage_run(
        tmp_path, "exp022", "compute", export_root="export/cells"
    ) as run:
        for seed in recipe.SEEDS:
            name = recipe.training_cell_name(seed)
            folder = run.export / "cells" / name
            folder.mkdir(parents=True)
            cfg = {
                "model": "ping",
                "dataset": "mnist",
                "dt": 0.1,
                "t_ms": 200.0,
                "n_in": 784,
                "n_hidden": 1024,
                "n_inh": 256,
                "n_out": 10,
                "epochs": 50,
                "max_samples": 7000,
                "seed": seed,
                "readout_mode": "spike-count",
                "input_rates": list(recipe.TRAINING_RATES_HZ),
                "dataset_split": {
                    "checkpoint_selection_partition": "validation",
                    "official_test_used_during_training": False,
                },
            }
            write_json_atomic(folder / "config.json", cfg)
            cps = {}
            for role, filename, epoch in (
                ("best_validation", "weights.pth", 43),
                ("final_epoch", "weights_final.pth", 50),
            ):
                (folder / filename).write_bytes((name + role).encode())
                cps[role] = {
                    "filename": filename,
                    "epoch": epoch,
                    "sha256": file_sha256(folder / filename),
                }
            write_json_atomic(
                folder / "metrics.json",
                {
                    "config": cfg,
                    "training_cell_name": name,
                    "best_epoch": 43,
                    "checkpoints": cps,
                    "epochs": [{"ep": i} for i in range(1, 51)],
                },
            )
    calls = []

    class Worker:
        dataset = {"fixture": True}

        def __init__(self, bank, directory, cfg):
            self.directory, self.cfg = directory, cfg

        def condition(self, job):
            calls.append(job["id"])
            folder = self.directory / "export" / job["path"]
            folder.mkdir(parents=True)
            shape = (self.cfg["streams_per_cell"], self.cfg["digits_per_stream"])
            out = np.zeros((*shape, 10), dtype=np.int64)
            out[..., 0] = 1
            np.savez_compressed(
                folder / "counts.npz",
                out_counts=out,
                e_counts=np.ones(shape, dtype=np.int64),
                i_counts=np.ones(shape, dtype=np.int64),
                labels=np.zeros(shape, dtype=np.int64),
            )
            attachment = self.directory / "provenance/simulations" / job["path"]
            attachment.mkdir(parents=True)
            write_json_atomic(attachment / "command.json", {"fixture": True})

        def stream(self, name):
            calls.append(name)
            conditions = (
                [[200.0, 5.0]] * 5
                if name == "matched"
                else [list(c) for c in recipe.VARIABLE_STREAM]
            )
            bounds = np.cumsum([0, *[int(d / 0.1) for d, _ in conditions]]).tolist()
            folder = self.directory / "export/streams" / name
            folder.mkdir(parents=True)
            out = np.zeros((bounds[-1], 10), dtype=np.int8)
            out[bounds[:-1], 0] = 1
            np.savez_compressed(
                folder / "recordings.npz",
                pixels=np.zeros((5, 784), dtype=np.float32),
                spikes_e=np.zeros((bounds[-1], 1024), dtype=np.int8),
                spikes_i=np.zeros((bounds[-1], 256), dtype=np.int8),
                spikes_out=out,
            )
            write_json_atomic(
                folder / "stream.json",
                {"labels": [0] * 5, "boundaries": bounds, "conditions": conditions},
            )

    monkeypatch.setattr(compute, "Inference", Worker)
    return tmp_path, run.run_id, calls


def resign(folder):
    record = load_json(folder / "run.json")
    record["payload_digest"] = payload_digest(folder)
    write_json_atomic(folder / "run.json", record)


@pytest.fixture
def historical_lab(lab, monkeypatch):
    root, bank_id, _ = lab
    monkeypatch.setattr(import_gold2, "REPO", root)
    archive = root / "archive"
    archive.mkdir()
    cfg = recipe.configuration()
    rows, files = [], []

    def add(name, value):
        path = archive / name
        write_json_atomic(path, value)
        files.append(
            {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )

    for job in recipe.jobs(cfg):
        row = {k: job[k] for k in ("seed", "duration_ms", "rate_hz")}
        row.update(
            stream_batch_size=5,
            n_correct=200,
            n_total=200,
            accuracy=1.0,
            output_spikes_per_presentation=1.0,
            silent_fraction=0.0,
            class_spike_totals=[200] + [0] * 9,
            rate_e_hz=1.0,
            rate_i_hz=1.0,
        )
        rows.append(row)
        add("state/experiments/exp082/conditions/" + job["id"] + ".json", row)
    old = {
        "config": {
            k: v
            for k, v in cfg.items()
            if k not in ("schema", "profile", "checkpoint_policy")
        },
        "grid_per_seed": rows,
        "repair_run_provenance": {"fixture": True},
    }
    recordings = {}
    for name in ("matched", "variable"):
        conditions = (
            [[200.0, 5.0]] * 5
            if name == "matched"
            else [list(c) for c in recipe.VARIABLE_STREAM]
        )
        bounds = np.cumsum([0, *[int(d / 0.1) for d, _ in conditions]]).tolist()
        old[name + "_stream"] = {
            "labels": [0] * 5,
            "boundaries": bounds,
            "conditions": conditions,
        }
        recordings[name] = {"pixels": np.zeros((5, 784), dtype=np.float32)}
        for key, width in (("spikes_e", 1024), ("spikes_i", 256), ("spikes_out", 10)):
            recordings[name][key] = np.zeros((bounds[-1], width), dtype=np.int8)
        recordings[name]["spikes_out"][bounds[:-1], 0] = 1
    add(import_gold2.DATA + "numbers.json", old)
    for n in range(65):
        add(f"provenance/fixture-{n}.json", {"fixture": n})
    padding = 6079619 - sum(f["size_bytes"] for f in files)
    filler = archive / "provenance/padding.txt"
    filler.write_bytes(b"x" * padding)
    files.append(
        {
            "path": "provenance/padding.txt",
            "size_bytes": padding,
            "sha256": file_sha256(filler),
        }
    )
    assert len(files) == 199
    monkeypatch.setattr(import_gold2, "selection", lambda _: (files, []))
    monkeypatch.setattr(import_gold2, "live_metadata", lambda _: {"fixture": True})
    monkeypatch.setattr(
        import_gold2,
        "verify_bank",
        lambda _, bank, old: evidence.training_contract(bank.export),
    )
    monkeypatch.setattr(
        import_gold2, "reconstruct", lambda *a: (recordings, {"fixture": True})
    )
    return root, bank_id, archive


def test_historical_import_independent_analysis(historical_lab, monkeypatch):
    root, bank, archive = historical_lab
    for module, name in ((compute, "compute"), (inference.Inference, "simulate")):
        monkeypatch.setattr(
            module, name, lambda *a, **k: pytest.fail("upstream execution")
        )
    identity = import_gold2.import_run(archive, bank, root / "unused-mnist")
    run = inputs.source(root, identity, "compute")
    assert run.record["execution"]["operation"] == "historical-import"
    assert run.record["historical_import"]["producer_commit"] == historical.PRODUCER
    assert not (root / ".artifacts").exists()
    assert not list(run.export.rglob("counts.npz"))
    inputs.compute_evidence(root, run)
    derived = analyse.analyse(identity)
    numbers = load_json(inputs.source(root, derived, "analyse").export / "numbers.json")
    assert numbers["condition_evidence"] == "historical-aggregate/v1"
    assert len(numbers["grid_per_seed"]) == 132
    # A checksum-valid execute run cannot masquerade as an aggregate import.
    manifest = run.record
    manifest["execution"]["operation"] = "execute"
    write_json_atomic(run.directory / "run.json", manifest)
    with pytest.raises(PingstoreError, match="explicit historical import"):
        inputs.compute_evidence(root, inputs.source(root, identity, "compute"))


def test_historical_failure_stays_hidden(historical_lab, monkeypatch):
    root, bank, archive = historical_lab

    def fail(*args):
        raise PingstoreError("fixture validation failure")

    monkeypatch.setattr(historical, "validate_import", fail)
    with pytest.raises(PingstoreError, match="fixture validation"):
        import_gold2.import_run(archive, bank, root / "unused-mnist")
    runs = root / ".pingstore/runs"
    assert not list(runs.glob("exp082-*"))
    assert len(list(runs.glob(".exp082-*.tmp"))) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_correct", 201),
        ("accuracy", 0.5),
        ("silent_fraction", 0.001),
        ("rate_e_hz", -1),
        ("class_spike_totals", [0] * 10),
    ],
)
def test_historical_aggregate_rejects_inconsistent_values(historical_lab, field, value):
    _, _, archive = historical_lab
    cfg = recipe.configuration()
    job = recipe.jobs(cfg)[0]
    path = archive / "state/experiments/exp082/conditions" / (job["id"] + ".json")
    row = load_json(path)
    row[field] = value
    write_json_atomic(path, row)
    with pytest.raises(PingstoreError, match="invalid historical"):
        historical.aggregate(path, job, cfg)


def test_live_metadata_rejects_mismatch_without_reservation(tmp_path, monkeypatch):
    name = "run.json"
    (tmp_path / name).write_bytes(b"cached")
    monkeypatch.setattr(
        import_gold2, "HEADERS", {name: (6, file_sha256(tmp_path / name))}
    )
    monkeypatch.setattr(
        import_gold2.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=b"different"),
    )
    with pytest.raises(PingstoreError, match="live R2 metadata differs"):
        import_gold2.live_metadata(tmp_path)
    assert not (tmp_path / ".pingstore").exists()


@pytest.mark.parametrize("name", ["variable", "single_trial"])
def test_raster_labels_and_thumbnails_match_display(historical_lab, monkeypatch, name):
    import matplotlib.pyplot as plt

    root, bank, archive = historical_lab
    identity = import_gold2.import_run(archive, bank, root / "unused-mnist")
    source = inputs.source(root, identity, "compute")
    result = measurements.stream_result(
        *evidence.stream(source.export, "variable" if name == "variable" else "matched")
    )
    if name == "single_trial":
        result = measurements.first_correct_trial_from_stream(result)
    result.update(measurements.display_values(result))
    saved = []
    monkeypatch.setattr(plt, "close", lambda fig: saved.append(fig))
    present.plots.plot_stream_headline(result, root / "plot.png", "must-not-stamp")
    fig = saved[-1]
    fig.canvas.draw()
    by_label = {a.get_ylabel(): a for a in fig.axes if a.get_ylabel()}
    assert by_label["E cell"].get_yticklabels()[-1].get_text() == "200"
    assert by_label["I cell"].get_yticklabels()[-1].get_text() == "64"
    a = by_label["softmax share\n$p_c(u)$"]
    assert (
        a.yaxis.label.get_window_extent().y1 < by_label["I cell"].get_window_extent().y0
    )
    thumbnails = [a for a in fig.axes if a.images]
    assert len(thumbnails) == (5 if name == "variable" else 1)
    label_bottom = min(t.get_window_extent().y0 for t in fig.axes[0].texts)
    assert all(a.get_window_extent().y1 < label_bottom for a in thumbnails)
    widths = [a.get_window_extent().width for a in thumbnails]
    assert max(widths) - min(widths) < 0.01
    assert all(
        abs(a.get_window_extent().width - a.get_window_extent().height) < 0.01
        for a in thumbnails
    )
    assert all(t.get_text() != "must-not-stamp" for t in fig.texts)


def test_compact_rate_ticks_do_not_overlap(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    cfg = recipe.configuration()
    rows = [{**j, "accuracy": 0.5} for j in recipe.jobs(cfg)]
    saved = []
    monkeypatch.setattr(plt, "close", lambda fig: saved.append(fig))
    present.plots.plot_duration_rate_summary(
        measurements.plot_data(rows, cfg), tmp_path / "grid.png", "review"
    )
    fig = saved[-1]
    fig.canvas.draw()
    curve = next(a for a in fig.axes if a.get_ylabel() == "accuracy at 200 ms")
    labels = curve.get_xticklabels()
    assert [t.get_text() for t in labels] == [
        f"{r:g}" for r in cfg["psychometric_rates_hz"]
    ]
    boxes = [t.get_window_extent() for t in labels]
    assert all(a.x1 < b.x0 for a, b in zip(boxes, boxes[1:]))


def test_article_renders_figures_and_equations_from_explicit_input(historical_lab):
    import re
    import shutil
    import subprocess
    import xml.etree.ElementTree as ET

    from demolab_cli import _paths

    root, bank, archive = historical_lab
    cid = import_gold2.import_run(archive, bank, root / "unused-mnist")
    aid = analyse.analyse(cid)
    pid = present.present(aid)
    source = inputs.source(root, pid, "present")
    repo = Path(__file__).resolve().parents[2]
    shutil.copytree(repo / "writings", root / "writings")
    (root / ".demolab").mkdir()
    shutil.copyfile(_paths.TYP / "lib.typ", root / ".demolab/lib.typ")
    write_json_atomic(
        root / "preview.json",
        {"exp082": {"exp082": "/" + str(source.export.relative_to(root))}},
    )
    document = root / "document.typ"
    document.write_text(
        '#set page(paper: "a4", margin: 18mm)\n#set text(size: 10pt)\n#import "writings/exp082.typ": body\n#body\n'
    )
    command = [
        _paths.find_typst(repo),
        "compile",
        "--root",
        str(root),
        "--input",
        "demolab-preview-file=/preview.json",
    ]
    result = subprocess.run(
        [*command, str(document), str(root / "article.pdf")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [
            *command,
            "--features",
            "html",
            "--format",
            "html",
            str(document),
            str(root / "article.html"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    html = (root / "article.html").read_text()
    assert len(re.findall(r"<img\b", html)) == 5
    assert len(re.findall(r"<figcaption\b", html)) == 5
    assert html.count('class="exp082-equation"') == 3
    for equation in re.findall(r"<math\b.*?</math>", html, re.S):
        tree = ET.fromstring(equation)
        for element in tree.iter():
            if element.tag.split("}")[-1] in ("msub", "msup", "msubsup"):
                # Function arguments must stay on the baseline, outside indices.
                assert all(
                    "(" not in "".join(child.itertext()) for child in list(element)[1:]
                )
    assert "Minimum validation cross-entropy" in html
    assert "Validation accuracy selected" not in html
    assert re.search(r'<h3\b[^>]*>Results(?:<|$)', html)
    assert "Results:" not in html
    write_json_atomic(root / "preview.json", {})
    result = subprocess.run(
        [
            *command,
            "--features",
            "html",
            "--format",
            "html",
            str(document),
            str(root / "unavailable.html"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    empty = (root / "unavailable.html").read_text()
    assert "A required run is unavailable" in empty
    assert "<img" not in empty


def fake_plots(monkeypatch):
    for name in (
        "plot_single_trial",
        "plot_single_trial_transition",
        "plot_stream",
        "plot_variable_headline",
        "plot_psychometric",
        "plot_duration_rate_summary",
    ):
        monkeypatch.setattr(
            present.plots, name, lambda data, path, rid: path.write_text("figure")
        )
    monkeypatch.setattr(
        present.plots, "plot_design", lambda path: path.write_text("diagram")
    )


def test_independent_stages_and_flat_export(lab, monkeypatch):
    repo, bank, calls = lab
    cid = compute.compute(bank)
    assert len(calls) == 20
    assert not (repo / ".artifacts").exists()
    monkeypatch.setattr(
        compute, "Inference", lambda *a: pytest.fail("downstream launched inference")
    )
    aid = analyse.analyse(cid)
    fake_plots(monkeypatch)
    pid = present.present(aid)
    source = inputs.source(repo, pid, "present")
    assert all((source.export / f).is_file() for f in recipe.FIGURES)
    assert all(p.is_file() for p in source.export.iterdir())
    assert set(inputs.lineage(repo, pid)) == {bank, cid, aid, pid}
    assert (
        load_json(source.export / "numbers.json")["grid_per_seed"][0]["accuracy"] == 1.0
    )
    assert all(
        name.endswith(stage)
        for name, stage in ((cid, "compute"), (aid, "analyse"), (pid, "present"))
    )


@pytest.mark.parametrize("stage", ["compute", "analyse", "present"])
def test_missing_source_never_allocates(lab, stage):
    repo, _, _ = lab
    before = set((repo / ".pingstore/runs").iterdir())
    with pytest.raises((PingstoreError, OSError)):
        getattr(
            {"compute": compute, "analyse": analyse, "present": present}[stage], stage
        )("exp082-r999-compute")
    assert set((repo / ".pingstore/runs").iterdir()) == before


@pytest.mark.parametrize("target", ["payload", "manifest", "v2", "root", "symlink"])
def test_source_corruption_rejected(lab, target):
    repo, bank, _ = lab
    cid = compute.compute(bank)
    folder = repo / ".pingstore/runs" / bank
    if target == "payload":
        (folder / "export/changed").write_text("corruption")
    elif target == "manifest":
        record = load_json(folder / "run.json")
        record["extra"] = 1
        write_json_atomic(folder / "run.json", record)
    elif target == "v2":
        record = load_json(folder / "run.json")
        record["schema"] = "pingstore.run/v2"
        write_json_atomic(folder / "run.json", record)
    elif target == "root":
        (folder / "unexpected").write_text("invalid root")
        resign(folder)
    else:
        (folder / "export/link").symlink_to(folder / "export/cells")
    with pytest.raises(PingstoreError):
        analyse.analyse(cid)


def test_compute_failure_stays_hidden(lab, monkeypatch):
    repo, bank, _ = lab
    monkeypatch.setattr(
        compute.Inference,
        "condition",
        lambda *a: (_ for _ in ()).throw(RuntimeError("failed simulator")),
    )
    with pytest.raises(RuntimeError, match="failed simulator"):
        compute.compute(bank)
    assert not list((repo / ".pingstore/runs").glob("exp082-*"))
    assert len(list((repo / ".pingstore/runs").glob(".exp082-*.tmp"))) == 1


def test_input_change_during_compute_stays_hidden(lab, monkeypatch):
    repo, bank, _ = lab
    original = compute.Inference.stream

    def mutate(self, name):
        original(self, name)
        (repo / ".pingstore/runs" / bank / "export/changed").write_text("changed")

    monkeypatch.setattr(compute.Inference, "stream", mutate)
    with pytest.raises(PingstoreError):
        compute.compute(bank)
    assert not list((repo / ".pingstore/runs").glob("exp082-*"))


def test_six_shards_reuse_collect_and_no_reexecution(lab):
    repo, bank, calls = lab
    identity = stages.reserve_stage(repo / ".pingstore", "exp082", "compute")
    with pytest.raises((OSError, PingstoreError)):
        compute.compute(bank, run_id=identity, collect=True)
    for index in range(6):
        compute.shard(bank, run_id=identity, index=index)
    assert len(calls) == 18
    compute.shard(bank, run_id=identity, index=0)
    assert len(calls) == 18
    assert compute.compute(bank, run_id=identity, collect=True) == identity
    assert calls[-2:] == ["matched", "variable"]


def test_shard_rejects_changed_payload(lab):
    repo, bank, _ = lab
    identity = stages.reserve_stage(repo / ".pingstore", "exp082", "compute")
    compute.shard(bank, run_id=identity, index=0)
    folder = repo / ".pingstore/runs" / f".{identity}.tmp"
    next((folder / "export/jobs").glob("*/counts.npz")).write_bytes(b"broken")
    with pytest.raises(PingstoreError, match="payload"):
        compute.shard(bank, run_id=identity, index=0)


def test_collection_adapter_and_workload():
    cfg = recipe.configuration()
    assert len(recipe.jobs(cfg)) == 132
    row = next(
        r
        for s in plan.build_plan(Path("/tmp/exp082-campaign"), "fixture")["stages"]
        for r in s["experiments"]
        if r["slug"] == "exp082"
    )
    assert row["execution"]["mode"] == "exp082-staged"
    assert execution._stage_adapter("exp082") is collection
    assert row["execution"]["workload_contract"]["classified_presentations"] == 26400
    with pytest.raises(ValueError, match="explicit v3 bank"):
        workloads.execute_shard("exp082", 0, 6, smoke=False)


def test_batches_preserve_time_axis_and_partial_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        inference,
        "load_mnist_split",
        lambda **k: (None, np.ones((20, 784)), None, np.arange(20) % 10),
    )
    monkeypatch.setattr(inference, "encode_stream", lambda *a: torch.ones((4, 1, 784)))
    cfg = recipe.configuration(streams=5, digits=2, batch=3)
    worker = inference.Inference(SimpleNamespace(export=tmp_path), tmp_path, cfg)
    seen = []

    def simulate(train, spikes, resets, attachments, kind):
        seen.append(tuple(spikes.shape))
        assert resets == (0, 20)
        shape = (spikes.shape[1], 2)
        return {
            "out_counts": np.ones((*shape, 10), dtype=np.int64),
            "e_counts": np.ones(shape, dtype=np.int64),
            "i_counts": np.ones(shape, dtype=np.int64),
        }

    monkeypatch.setattr(worker, "simulate", simulate)
    job = {
        "id": "fixture",
        "path": "jobs/fixture",
        "seed": 42,
        "duration_ms": 2.0,
        "rate_hz": 5.0,
        "cell_name": recipe.training_cell_name(42),
    }
    worker.condition(job)
    assert seen == [(4, 3, 784), (4, 2, 784)]
    counts = evidence.counts(tmp_path / "export/jobs/fixture/counts.npz", cfg)
    assert measurements.condition_row(job, counts, cfg)["n_total"] == 10


def test_analysis_sem_preserves_three_seed_estimator():
    cfg = recipe.configuration()
    rows = [{**j, "accuracy": (j["seed"] - 42) / 2} for j in recipe.jobs(cfg)]
    data = measurements.plot_data(rows, cfg)
    np.testing.assert_allclose(data["means"], 0.5)
    np.testing.assert_allclose(data["sems"], 0.5 / np.sqrt(3))


def test_missing_pixels_is_an_error_not_dataset_fallback(lab):
    repo, bank, _ = lab
    cid = compute.compute(bank)
    root = repo / ".pingstore/runs" / cid
    path = root / "export/streams/matched/recordings.npz"
    data = evidence.arrays(path)
    del data["pixels"]
    np.savez_compressed(path, **data)
    resign(root)
    with pytest.raises(PingstoreError, match="explicit pixels"):
        analyse.analyse(cid)


def test_plot_failure_preserves_sources_and_stays_hidden(lab, monkeypatch):
    repo, bank, _ = lab
    cid = compute.compute(bank)
    aid = analyse.analyse(cid)
    before = {
        p.name: file_sha256(p / "run.json")
        for p in (repo / ".pingstore/runs").iterdir()
    }
    monkeypatch.setattr(
        present.plots,
        "plot_single_trial",
        lambda *a: (_ for _ in ()).throw(RuntimeError("plot failure")),
    )
    with pytest.raises(RuntimeError, match="plot failure"):
        present.present(aid)
    assert {
        p.name: file_sha256(p / "run.json")
        for p in (repo / ".pingstore/runs").iterdir()
        if not p.name.startswith(".")
    } == before
    assert len(list((repo / ".pingstore/runs").glob(".exp082-*-present.tmp"))) == 1


def test_all_saved_figures_render_without_inference(lab, monkeypatch):
    repo, bank, _ = lab
    cid = compute.compute(bank)
    aid = analyse.analyse(cid)
    monkeypatch.setattr(
        inference,
        "load_mnist_split",
        lambda **k: pytest.fail("presentation fetched dataset"),
    )
    pid = present.present(aid)
    output = inputs.source(repo, pid, "present").export
    assert all((output / name).stat().st_size > 100 for name in recipe.FIGURES)


def test_compute_lock_excludes_collector(lab):
    repo, bank, _ = lab
    identity = stages.reserve_stage(repo / ".pingstore", "exp082", "compute")
    directory = repo / ".pingstore/runs" / f".{identity}.tmp"
    with compute._compute_lock(directory, exclusive=False):
        with pytest.raises(PingstoreError, match="busy"):
            compute.compute(bank, run_id=identity, collect=True)


def test_dirty_shards_fail_without_scientific_work(lab, monkeypatch):
    repo, bank, calls = lab
    identity = stages.reserve_stage(repo / ".pingstore", "exp082", "compute")
    monkeypatch.setattr(compute, "_capture_code", lambda *a: {"code_dirty": True})
    with pytest.raises(PingstoreError, match="committed"):
        compute.shard(bank, run_id=identity, index=0)
    assert calls == []


def test_collection_dispatches_explicit_stage_sources(lab, monkeypatch):
    repo, bank, calls = lab
    fake_plots(monkeypatch)
    campaign = plan.build_plan(repo / "campaign", "fixture", smoke=True)
    campaign["profile"] = "smoke"
    campaign["exp022_manifest"] = str(repo / "campaign/exp022/campaign.json")
    write_json_atomic(Path(campaign["exp022_manifest"]), {"pingstore_run_id": bank})
    row = next(
        r for s in campaign["stages"] for r in s["experiments"] if r["slug"] == "exp082"
    )
    commands = []

    def dispatch(command, **kwargs):
        commands.append(command)
        stage = command[2].rsplit(".", 1)[1]
        source = command[command.index("--source") + 1]
        identity = command[command.index("--run-id") + 1]
        getattr(
            {"compute": compute, "analyse": analyse, "present": present}[stage], stage
        )(source, run_id=identity)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(collection.subprocess, "run", dispatch)
    references = collection.execute(repo, campaign, row)
    assert [c[2] for c in commands] == [
        "experiments.exp082." + s for s in collection.STAGES
    ]
    assert references["bank"]["run_id"] == bank
    assert set(references) == {"bank", "compute", "analyse", "present"}
    collection.execute(repo, campaign, row)
    assert len(commands) == 3
    assert len(calls) == 20


def test_simulator_serializes_batched_input_and_resets(tmp_path, monkeypatch):
    monkeypatch.setattr(
        inference,
        "load_mnist_split",
        lambda **k: (None, np.zeros((10, 784)), None, np.arange(10)),
    )
    worker = inference.Inference(
        SimpleNamespace(export=tmp_path), tmp_path, recipe.configuration()
    )

    def simulate(command, **kwargs):
        assert Path(command[1]).is_file()
        assert Path(command[1]).parts[-2:] == ("snnsim", "tool.py")
        assert Path(command[command.index("--load-weights") + 1]).name == "weights.pth"
        assert command[command.index("--device") + 1] == "auto"
        raw = evidence.arrays(Path(command[command.index("--input-file") + 1]))
        assert raw["input_spikes"].shape == (4, 3, 784)
        assert raw["readout_reset"].tolist() == [True, False, True, False]
        out = Path(command[command.index("--out-dir") + 1])
        out.mkdir()
        np.savez_compressed(
            out / "spike_summary.npz",
            dt=np.float32(0.1),
            T=4,
            n_trials=3,
            segment_starts=[0, 2],
            segment_stops=[2, 4],
            out_counts=np.zeros((3, 2, 10), dtype=np.int64),
            e_counts=np.zeros((3, 2), dtype=np.int64),
            i_counts=np.zeros((3, 2), dtype=np.int64),
        )

    monkeypatch.setattr(inference.subprocess, "run", simulate)
    raw = worker.simulate(
        tmp_path,
        torch.ones((4, 3, 784)),
        (0, 2),
        tmp_path / "attachments",
        "spike_summary",
    )
    assert raw["out_counts"].shape == (3, 2, 10)
