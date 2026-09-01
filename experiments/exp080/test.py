"""Focused unit tests for the standalone EXP080 empirical runner."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest
from experiments import exp080
from experiments.exp080 import (
    analyse,
    collection,
    compute,
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
    validate_operational_run_directory,
    write_json_atomic,
)
from pingstore.discovery import discover_runs
from pingstore.layout import canonical_export_file


def test_registered_simulator_validations_pass() -> None:
    record = exp080.validate_simulator()
    assert all(record["checks"].values())


def test_early_spike_contributes_more_than_late_spike() -> None:
    assert exp080.probe_single_spike(20.0) > exp080.probe_single_spike(180.0) > 0.0


def test_direct_features_replay_and_zero_input() -> None:
    import torch

    device = compute.torch_device()
    images = torch.zeros((2, 28, 28), dtype=torch.uint8, device=device)
    rates = torch.tensor([0.5, 25.0], device=device)
    first = compute.direct_features(
        images,
        rates,
        torch.Generator(device=device).manual_seed(123),
    )
    replay = compute.direct_features(
        images,
        rates,
        torch.Generator(device=device).manual_seed(123),
    )
    assert torch.equal(first, replay)
    assert np.all(first.cpu().numpy() == 0.0)


def test_rate_grid_brackets_registered_floor() -> None:
    assert exp080.RATES_HZ == (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
    assert exp080.USEFUL_ACCURACY == 0.5


def test_floor_requires_every_decoder_to_cross_criterion(tmp_path, monkeypatch) -> None:
    correctness = np.zeros((len(exp080.RATES_HZ), len(exp080.SEEDS), 10), dtype=bool)
    correctness[1, :, :6] = True
    correctness[1, 2, 4:6] = False
    correctness[2:, :, :6] = True

    decision = exp080.analyze(correctness)

    assert decision["r_train_hz"] == 0.5
    assert decision["criterion_crossed"] is True
    assert decision["rows"][1]["accuracy"] > exp080.USEFUL_ACCURACY
    assert decision["rows"][1]["minimum_seed_accuracy"] < exp080.USEFUL_ACCURACY


def test_no_crossing_is_recorded_as_a_censored_result(tmp_path, monkeypatch) -> None:
    correctness = np.zeros((len(exp080.RATES_HZ), len(exp080.SEEDS), 10), dtype=bool)

    decision = exp080.analyze(correctness)

    assert decision["criterion_crossed"] is False
    assert decision["r_train_hz"] is None
    assert decision["recommendation"] == {
        "floor_hz": None,
        "ceiling_hz": max(exp080.RATES_HZ),
    }
    assert list(tmp_path.iterdir()) == []


def dataset_record(count, prefix):
    return {
        "source": "synthetic test fixture",
        "image_shape": [count, 28, 28],
        "label_shape": [count],
        "raw_sha256": {
            f"{prefix}-images-idx3-ubyte": "a" * 64,
            f"{prefix}-labels-idx1-ubyte": "b" * 64,
        },
    }


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(stages, "memberships", lambda _: {"exp080": "test"})
    monkeypatch.setattr(
        stages, "_capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    monkeypatch.setenv("PINGLAB_SMOKE", "1")
    monkeypatch.setenv("EXP080_DEVICE", "cpu")
    calls = []
    monkeypatch.setattr(
        compute,
        "load_mnist_training",
        lambda: (
            np.zeros((150, 28, 28), dtype=np.uint8),
            np.zeros(150, dtype=np.int64),
            dataset_record(60000, "train"),
        ),
    )

    def illustration(images, output):
        calls.append("illustration")
        np.savez_compressed(
            output / "feature_samples.npz",
            image=images[0],
            features_mV=np.zeros((3, 784), dtype=np.float32),
            rates_hz=np.asarray([0.5, 5.0, 25.0]),
        )

    def train(images, labels, seed, output, cfg, *, state_cache):
        calls.append(seed)
        assert not list((tmp_path / ".pingstore/runs").glob("exp080-*-compute"))
        directory = output / "models" / f"seed-{seed}"
        directory.mkdir(parents=True)
        state_cache[seed] = {"fixture": seed}
        record = {
            "seed": seed,
            "device": "cpu",
            "runtime_s": 0.1,
            "selected_epoch": 1,
            "selected_validation_accuracy": 0.75,
            "history": [
                {"epoch": i, "train_accuracy": 0.7, "validation_accuracy": 0.75}
                for i in range(1, cfg["epochs"] + 1)
            ],
            "checkpoint_retention": "memory_only",
        }
        write_json_atomic(directory / "training.json", record)
        return record

    def evaluate(records, output, cfg, *, state_cache):
        assert set(state_cache) == set(cfg["seeds"])
        calls.append("evaluation")
        values = np.zeros((8, 3, cfg["test_count"]), dtype=bool)
        values[2:, :, :30] = True
        path = output / "held_out_correctness.npz"
        np.savez_compressed(
            path,
            correctness=values,
            rates_hz=np.asarray(cfg["rates_hz"]),
            seeds=np.asarray(cfg["seeds"]),
            labels=np.zeros(cfg["test_count"], dtype=np.int64),
        )
        return {
            "device": "cpu",
            "runtime_s": 0.1,
            "dataset": dataset_record(cfg["test_count"], "t10k"),
            "arrays_sha256": file_sha256(path),
        }, values

    monkeypatch.setattr(compute, "illustrative_features", illustration)
    monkeypatch.setattr(compute, "train_seed", train)
    monkeypatch.setattr(compute, "evaluate", evaluate)
    return tmp_path, calls


def resign(directory):
    record = load_json(directory / "run.json")
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)


def forbidden(*a, **k):
    pytest.fail("stage crossed its execution boundary")


def test_stages_are_isolated_and_replay_only_explicit_evidence(repo, monkeypatch):
    root, calls = repo
    compute_id = compute.compute()
    assert calls == ["illustration", 42, 43, 44, "evaluation"]
    source = inputs.source(root, compute_id, "compute")
    before = source.reference
    assert source.record["inputs"] == {}
    assert discover_runs(root / ".pingstore/runs") == []
    for name in ("compute", "direct_features", "train_seed", "evaluate"):
        monkeypatch.setattr(compute, name, forbidden)
    monkeypatch.setattr(recipe, "validate_simulator", forbidden)
    monkeypatch.setenv("PINGLAB_SMOKE", "0")
    analysis_id = analyse.analyse(compute_id)
    analysis = inputs.source(root, analysis_id, "analyse")
    result = load_json(analysis.export / "results.json")
    assert result["recipe"]["profile"] == "smoke"
    assert result["decision"]["r_train_hz"] == 0.5
    assert result["decision"]["rows"][2]["per_seed_accuracy"] == [0.6, 0.6, 0.6]
    assert not list(analysis.export.rglob("*.pt"))
    assert discover_runs(root / ".pingstore/runs") == []
    monkeypatch.setattr(measurements, "analyze", forbidden)
    monkeypatch.setattr(analyse, "analyse", forbidden)
    present_id = present.present(analysis_id)
    output = inputs.source(root, present_id, "present")
    assert output.record["inputs"] == {
        "analysis": analysis.reference,
        "compute": source.reference,
    }
    assert load_json(output.presentation / "numbers.json") == result
    assert {p.name for p in output.presentation.iterdir()} >= {
        "numbers.json",
        "decision.json",
        "feature_images.png",
        "psychometric.svg",
        "training_history.svg",
    }
    assert all(p.is_file() for p in output.presentation.iterdir())
    assert not (root / ".artifacts").exists()
    assert inputs.source(root, compute_id, "compute").reference == before
    assert [r["id"] for r in discover_runs(root / ".pingstore/runs")] == [present_id]


@pytest.mark.parametrize("stage", ["compute", "analyse", "present"])
def test_failure_remains_hidden_and_preserves_prior_runs(repo, monkeypatch, stage):
    root, _ = repo

    def fail(*a, **k):
        raise RuntimeError("injected failure")

    source_id = None
    if stage != "compute":
        source_id = compute.compute()
    if stage == "present":
        source_id = analyse.analyse(source_id)
    existing = {
        p.name: file_sha256(p / "run.json")
        for p in (root / ".pingstore/runs").glob("exp080-*")
    }
    module, function = {
        "compute": (compute, "train_seed"),
        "analyse": (measurements, "analyze"),
        "present": (present.plots, "plot_training"),
    }[stage]
    monkeypatch.setattr(module, function, fail)
    with pytest.raises(RuntimeError, match="injected"):
        if stage == "compute":
            compute.compute()
        else:
            getattr({"analyse": analyse, "present": present}[stage], stage)(source_id)
    assert {
        p.name: file_sha256(p / "run.json")
        for p in (root / ".pingstore/runs").glob("exp080-*")
    } == existing
    assert len(list((root / ".pingstore/runs").glob(f".exp080-*-{stage}.tmp"))) == 1


@pytest.mark.parametrize(
    "change",
    [
        "payload",
        "manifest",
        "schema",
        "stage",
        "recipe",
        "checkpoint",
        "selection",
        "arrays",
    ],
)
def test_invalid_sources_fail_closed(repo, change):
    root, _ = repo
    identity = compute.compute()
    source = inputs.source(root, identity, "compute")
    if change == "payload":
        path = source.export / "held_out_correctness.npz"
        path.write_bytes(path.read_bytes() + b"tampered")
    elif change in {"manifest", "schema", "stage", "recipe"}:
        record = load_json(source.directory / "run.json")
        if change == "manifest":
            record["inputs"] = {"unexpected": source.reference}
        elif change == "schema":
            record["schema"] = "pingstore.run/v2"
        elif change == "stage":
            record["stage"] = "analyse"
        else:
            record["execution"]["configuration"]["test_count"] = 51
        write_json_atomic(source.directory / "run.json", record)
    else:
        document = load_json(source.export / "evidence.json")
        if change == "checkpoint":
            source.file("models", "seed-42", "decoder.pt").write_bytes(
                b"other checkpoint"
            )
        elif change == "selection":
            document["training"][0]["selected_epoch"] = 2
            write_json_atomic(
                source.file("models", "seed-42", "training.json"),
                document["training"][0],
            )
        else:
            path = source.export / "held_out_correctness.npz"
            with np.load(path) as data:
                arrays = {k: data[k] for k in data.files}
            arrays["rates_hz"] = arrays["rates_hz"][::-1]
            np.savez_compressed(path, **arrays)
            document["evaluation"]["arrays_sha256"] = file_sha256(path)
        write_json_atomic(source.export / "evidence.json", document)
        resign(source.directory)
    with pytest.raises((PingstoreError, OSError)):
        analyse.analyse(identity)
    assert not list((root / ".pingstore/runs").glob("exp080-*-analyse"))


def test_presentation_allows_transitive_metadata_amendment_during_execution(
    repo, monkeypatch
):
    root, _ = repo
    compute_id = compute.compute()
    analysis_id = analyse.analyse(compute_id)
    source = inputs.source(root, compute_id, "compute")
    original = present.plots.plot_training

    def mutate(*args):
        original(*args)
        record = load_json(source.directory / "run.json")
        record["execution"]["command"] = ["changed"]
        write_json_atomic(source.directory / "run.json", record)

    monkeypatch.setattr(present.plots, "plot_training", mutate)
    presentation_id = present.present(analysis_id)
    assert (root / ".pingstore/runs" / presentation_id).is_dir()


def test_reservations_are_source_neutral_atomic_and_not_reusable(repo):
    root, _ = repo
    identity = stages.reserve_stage(
        root / ".pingstore", "exp080", "compute", origin="slurm"
    )
    assert identity == "exp080-r001-compute"
    assert not (root / ".pingstore/runs" / identity).exists()
    assert compute.compute(run_id=identity) == identity
    source = inputs.source(root, identity, "compute")
    assert source.record["origin"] == "slurm"
    assert not list((root / ".pingstore/runs").glob(".*.tmp"))
    with pytest.raises(PingstoreError, match="unused reserved"):
        compute.compute(run_id=identity)


def test_collection_adapter_reserves_and_dispatches_separate_stages(repo, monkeypatch):
    root, _ = repo
    row = {
        "execution": {"mode": "exp080-staged"},
        "paths": {"state": str(root / "campaign")},
        "required_outputs": [str(root / "campaign/stage-refs.json")],
    }
    commands = []

    def dispatch(command, **kwargs):
        commands.append(command)
        stage = command[2].rsplit(".", 1)[-1]
        identity = command[command.index("--run-id") + 1]
        if stage == "compute":
            compute.compute(run_id=identity)
        else:
            source = command[command.index("--source") + 1]
            getattr({"analyse": analyse, "present": present}[stage], stage)(
                source, run_id=identity
            )
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(collection.subprocess, "run", dispatch)
    refs = collection.execute(root, {"profile": "smoke"}, row)
    assert [c[2] for c in commands] == [
        f"experiments.exp080.{s}" for s in collection.STAGES
    ]
    assert set(refs) == set(collection.STAGES)
    assert collection.completed(root, {}, row).reference == refs["present"]
    collection.execute(root, {"profile": "smoke"}, row)
    assert len(commands) == 3
    with pytest.raises(PingstoreError, match="legacy"):
        collection.require_staged({"execution": {"mode": "monolithic"}})


@pytest.mark.parametrize("memory_only", [False, True])
def test_cpu_checkpoint_keeps_selected_epoch_weights(
    tmp_path, monkeypatch, memory_only
):
    import torch

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(1, 2))
            self.snapshots = []

        def forward(self, features):
            if self.training:
                return self.logits
            self.snapshots.append(self.logits.detach().clone())
            return torch.tensor(
                [[1.0, 0.0]] if len(self.snapshots) == 1 else [[0.0, 1.0]]
            )

    model = Decoder()
    monkeypatch.setenv("EXP080_DEVICE", "cpu")
    monkeypatch.setattr(compute, "make_model", lambda *a: model)
    monkeypatch.setattr(compute, "direct_features", lambda *a: torch.zeros(1, 784))
    cfg = {**recipe.configuration(smoke=True), "train_count": 1, "validation_count": 1}
    cache = {} if memory_only else None
    result = compute.train_seed(
        np.zeros((2, 28, 28), dtype=np.uint8),
        np.zeros(2, dtype=np.int64),
        42,
        tmp_path,
        cfg,
        state_cache=cache,
    )
    if memory_only:
        assert not list(tmp_path.rglob("*.pt"))
        assert result["checkpoint_retention"] == "memory_only"
        checkpoint = {"state_dict": cache[42]}
    else:
        checkpoint = torch.load(tmp_path / result["checkpoint"], weights_only=True)
    assert result["selected_epoch"] == 1
    assert torch.equal(checkpoint["state_dict"]["logits"], model.snapshots[0])
    assert not torch.equal(checkpoint["state_dict"]["logits"], model.snapshots[1])


def test_cli_help_and_legacy_entrypoint_do_not_execute():
    root = Path(__file__).resolve().parents[2]
    for stage in ("compute", "analyse", "present"):
        result = subprocess.run(
            [sys.executable, "-m", f"experiments.exp080.{stage}", "--help"],
            cwd=root,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [sys.executable, "-m", "experiments.exp080"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "independent stages" in result.stderr


def test_atomic_rename_only_exposes_a_valid_complete_run(repo, monkeypatch):
    root, _ = repo
    rename = os.rename
    seen = []

    def checked_rename(source, destination):
        if Path(destination).name.startswith("exp080-"):
            record = validate_operational_run_directory(source)
            assert record["stage"] == "compute"
            assert not Path(destination).exists()
            assert (Path(source) / "export/evidence.json").is_file()
            seen.append(record["run_id"])
        return rename(source, destination)

    monkeypatch.setattr(stages.os, "rename", checked_rename)
    identity = compute.compute()
    assert seen == [identity]
    assert (root / ".pingstore/runs" / identity).is_dir()


def test_importing_package_has_no_execution_or_plotting_side_effects(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from experiments import exp080; "
            "assert 'torch' not in sys.modules; assert 'matplotlib.pyplot' not in sys.modules",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def retain_historical_checkpoints(data, document):
    for record in document["training"]:
        record.pop("checkpoint_retention")
        checkpoint = canonical_export_file(
            data, "models", f"seed-{record['seed']}", "decoder.pt"
        )
        checkpoint.write_bytes(f"synthetic checkpoint {record['seed']}".encode())
        record.update(
            checkpoint=str(checkpoint.relative_to(data)),
            checkpoint_sha256=file_sha256(checkpoint),
        )
        training = canonical_export_file(
            data, "models", f"seed-{record['seed']}", "training.json"
        )
        write_json_atomic(training, record)


def test_historical_illustration_is_carried_without_simulation(repo, monkeypatch):
    from PIL import Image

    root, _ = repo
    identity = compute.compute()
    source = inputs.source(root, identity, "compute")
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(
        source.export / "feature_images.png"
    )
    original = (source.export / "feature_images.png").read_bytes()
    (source.export / "feature_samples.npz").unlink()
    document = load_json(source.export / "evidence.json")
    document["illustration"] = {
        "kind": "historical-image",
        "path": "feature_images.png",
    }
    retain_historical_checkpoints(source.export, document)
    write_json_atomic(source.export / "evidence.json", document)
    record = load_json(source.directory / "run.json")
    record["execution"]["operation"] = "historical-import"
    record["historical_import"] = {"producer": {"campaign_id": "fixture"}}
    write_json_atomic(source.directory / "run.json", record)
    resign(source.directory)
    monkeypatch.setattr(compute, "direct_features", forbidden)
    monkeypatch.setattr(present.plots, "plot_feature_images", forbidden)
    analysis_id = analyse.analyse(identity)
    presentation_id = present.present(analysis_id)
    presentation = inputs.source(root, presentation_id, "present")
    assert (presentation.export / "feature_images.png").read_bytes() == original
    assert (
        presentation.record["retained_figures"]["feature_images.png"]["regenerated"]
        is False
    )


def test_shared_collection_registration_rejects_legacy_rows(tmp_path):
    from experiments.collections.gamma_gated_sparsity import execution, plan

    campaign = plan.build_plan(tmp_path / "campaign", "fixture")
    row = next(
        r for s in campaign["stages"] for r in s["experiments"] if r["slug"] == "exp080"
    )
    assert row["execution"] == {
        "mode": "exp080-staged",
        "stages": ["compute", "analyse", "present"],
    }
    assert row["command"] == []
    assert row["required_outputs"] == [
        str(tmp_path / "campaign/downstream/exp080/stage-refs.json")
    ]
    assert execution._stage_adapter("exp080") is collection
    legacy = {**row, "execution": {"mode": "monolithic"}}
    assert not execution._outputs_valid_for_plan(campaign, legacy)
    with pytest.raises(PingstoreError, match="legacy"):
        execution._run_downstream(campaign, legacy)


@pytest.mark.parametrize("view", ["crossed", "censored", "absent", "broken"])
def test_article_renders_explicit_evidence_without_false_branches(repo, view):
    from demolab_cli import _paths

    root, _ = repo
    project = Path(__file__).resolve().parents[2]
    presentation_id = present.present(analyse.analyse(compute.compute()))
    run = inputs.source(root, presentation_id, "present")
    # Changes below affect a disposable rendering fixture, never a completed run.
    shutil.copytree(run.export, root / "render-data")
    numbers = root / "render-data/numbers.json"
    if view == "censored":
        record = load_json(numbers)
        record["decision"] = measurements.analyze(np.zeros((8, 3, 50), dtype=bool))
        write_json_atomic(numbers, record)
    elif view == "broken":
        numbers.write_text("not json")
    (root / "writings").mkdir()
    for name in ("exp080.typ", "contents.typ", "run-inputs.typ", "run-view.typ"):
        shutil.copyfile(project / "writings" / name, root / "writings" / name)
    (root / ".demolab").mkdir()
    for name in ("lib.typ", "style.css"):
        shutil.copyfile(_paths.TYP / name, root / ".demolab" / name)
    (root / ".demolab/VERSION").write_text("test")
    write_json_atomic(
        root / "preview.json",
        {} if view == "absent" else {"exp080": {"exp080": "/render-data"}},
    )
    document = root / "article.typ"
    document.write_text(
        '#import "/.demolab/lib.typ": entry-page\n'
        '#import "/writings/exp080.typ": meta, body\n'
        '#entry-page(meta, body, id: "exp080")\n'
    )
    command = [
        _paths.find_typst(project),
        "compile",
        "--root",
        str(root),
        "--input",
        "demolab-preview-file=/preview.json",
        "--features",
        "html",
        "--format",
        "html",
        str(document),
        str(root / "article.html"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if view == "broken":
        assert result.returncode != 0 and "json" in result.stderr.lower()
        return
    assert result.returncode == 0, result.stderr
    html = (root / "article.html").read_text()
    assert html.count('<nav aria-label="Table of Contents">') == 1
    assert "else [" not in html
    if view == "absent":
        assert 'class="pinglab-numbered-equation"' not in html
        assert "All three nonlinear decoders reached" not in html
        assert "A required run is unavailable" in html
        return
    headings = re.findall(r"<h3\b[^>]*>(.*?)</h3>", html, re.S)
    assert sum(heading.startswith("References") for heading in headings) == 1
    assert html.count('class="pinglab-numbered-equation"') == 3
    for number in range(1, 4):
        assert f'<span class="pinglab-equation-number">({number})</span>' in html
    assert len(re.findall(r"<img\b", html)) == 3
    for math in re.findall(r"<math\b.*?</math>", html, re.S):
        tree = ElementTree.fromstring(math)
        for subscript in tree.iter("msub"):
            assert "(" not in "".join(subscript[1].itertext())
    assert "over 50 decoder-training epochs" not in html
    if view == "crossed":
        assert "The selected floor was 0.5 Hz" in html
        assert "No rate met the criterion" not in html
    else:
        assert "No rate met the criterion" in html
        assert "The selected floor is" not in html


def test_cached_decoder_load_matches_checkpoint_predictions(tmp_path):
    import torch

    seed = 42
    model = compute.make_model(torch.device("cpu"), seed)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    path = tmp_path / "decoder.pt"
    torch.save({"state_dict": state}, path)
    records = [
        {"seed": seed, "checkpoint": path.name, "checkpoint_sha256": file_sha256(path)}
    ]
    disk = compute.load_models(records, torch.device("cpu"), tmp_path)[0]
    cached = compute.load_models(
        records, torch.device("cpu"), tmp_path, state_cache={seed: state}
    )[0]
    features = torch.arange(784, dtype=torch.float32)[None] / 784
    with torch.no_grad():
        assert torch.equal(disk(features), cached(features))
