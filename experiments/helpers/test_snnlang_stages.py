"""Synthetic acquisition checks the four integration stages without production work."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from experiments.helpers import snnlang_stages as boundary
from pingstore import stages
from pingstore.contracts import (
    PingstoreError,
    load_json,
    validate_operational_run_directory,
    write_json_atomic,
)
from pingstore.discovery import discover_runs

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["exp074", "exp075", "exp076", "exp077"])
def lab(request, tmp_path, monkeypatch):
    slug = request.param
    modules = {
        name: importlib.import_module(f"experiments.{slug}.{name}")
        for name in ("compute", "analyse", "present", "recipe", "plots")
    }
    recipe = modules["recipe"]
    registry = tmp_path / "experiments/collections/registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_bytes((ROOT / "experiments/collections/registry.json").read_bytes())
    for name in ("compute", "analyse", "present"):
        monkeypatch.setattr(modules[name], "REPO", tmp_path)
    monkeypatch.setattr(
        stages,
        "_capture_code",
        lambda *_: {
            "git_commit": "a" * 40,
            "dirty": False,
            "code_dirty": False,
            "patch": None,
        },
    )

    def command(repo, provenance, name, argv):
        out = Path(argv[argv.index("--out-dir") + 1])
        out.mkdir()
        row = {"ep": 1, "loss": 2.0, "test_loss": 2.1, "acc": 20.0, "samples": 128}
        metrics = {
            "epochs": [
                dict(row, ep=i + 1) for i in range(getattr(recipe, "EPOCHS", 1))
            ],
            "best_acc": 20.0,
            "best_epoch": 1,
            "total_elapsed_s": 1.0,
            "rate_e_hz": 3.0,
            "rate_i_hz": 4.0,
            "n_total": 160,
            "config": {
                "dataset_split": {
                    "validation_samples": 100,
                    "optimizer_train_samples": 900,
                },
                "validation_encoder_draws": {"count": 3},
                "input_rate": 25.0,
                "evaluation_partition": "official_mnist_test",
            },
        }
        write_json_atomic(out / "metrics.json", metrics)
        if slug == "exp074":
            np.savez_compressed(
                out / "rasters.npz",
                dt=recipe.DT_MS,
                e_trial=[0, 1],
                i_trial=[0, 1],
                e_t=[1, 2],
                i_t=[2, 3],
                e_cell=[0, 1],
                i_cell=[0, 1],
            )
        else:
            shapes = {
                "W_ff.0": (784, recipe.N_E),
                "W_ff.1": (recipe.N_E, 10),
                "W_ee.1": (recipe.N_E, recipe.N_E),
                "W_ei.1": (recipe.N_E, recipe.N_I),
                "W_ie.1": (recipe.N_I, recipe.N_E),
                "W_ii.1": (recipe.N_I, recipe.N_I),
            }
            for name in ("weights.pth", "weights_final.pth"):
                torch.save(
                    {key: torch.zeros(shape) for key, shape in shapes.items()},
                    out / name,
                )
        return {"elapsed_s": 1.0, "returncode": 0}

    monkeypatch.setattr(boundary, "command", command)
    monkeypatch.setattr(
        boundary,
        "test_evidence",
        lambda repo, run, name, nodes: {
            "nodes": nodes,
            "tests": len(nodes),
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "passed": True,
            "elapsed_s": 1.0,
        },
    )
    if slug == "exp077":

        def simulation(spec):
            return SimpleNamespace(
                recordings={
                    f"population_{i}": torch.zeros((recipe.STEPS, recipe.BATCH, size))
                    for i, size in enumerate((16, 4, 12, 3))
                }
            )

        monkeypatch.setattr(modules["compute"], "simulate", simulation)

        def acquisition(directory):
            keys = [
                "legacy_parameter__a",
                "native_parameter__a",
                "legacy_output",
                "native_output",
                "replay_output",
                "compiled_first",
                "compiled_warm",
                "legacy_e",
                "native_e",
                "legacy_i",
                "native_i",
            ]
            np.savez_compressed(
                directory / "parity.npz", **{key: np.zeros(2) for key in keys}
            )
            return {
                "timings": {"legacy": [1, 1, 1], "graph": [2, 2, 2]},
                "peaks": {"legacy": 10, "graph": 20},
                "compiled_times": [1, 1, 1],
                "compile_setup_s": 1,
                "compile_first_s": 2,
                "compile_workload_steps": 20,
                "compile_workload_batch": 2,
            }

        monkeypatch.setattr(modules["compute"], "acquire_parity", acquisition)

    # Real plotting is checked by production presentation; fixtures isolate storage.
    def plot(*args, **kwargs):
        Path(args[-1]).write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')

    for name in (
        "plot_rasters",
        "plot_training",
        "write_lifecycle_svg",
        "render_rasters",
    ):
        if hasattr(modules["plots"], name):
            monkeypatch.setattr(modules["plots"], name, plot)
    from snnlang.compiler import Bundle

    monkeypatch.setattr(
        Bundle,
        "visualise",
        lambda self, path, **kw: Path(path).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"/>'
        ),
    )
    return tmp_path, modules


def chain(lab):
    repo, modules = lab
    compute = modules["compute"].compute()
    analyse = modules["analyse"].analyse(compute)
    return repo, modules, compute, analyse


def test_independent_stages_and_explicit_immutable_ancestry(lab, monkeypatch):
    repo, modules, compute, analyse = chain(lab)
    before = {
        name: (repo / ".pingstore/runs" / name / "run.json").read_bytes()
        for name in (compute, analyse)
    }

    def forbidden(*a, **kw):
        raise AssertionError("presentation launched upstream work")

    monkeypatch.setattr(modules["compute"], "compute", forbidden)
    monkeypatch.setattr(modules["analyse"], "analyse", forbidden)
    monkeypatch.setattr(boundary, "command", forbidden)
    present = modules["present"].present(analyse)
    record = validate_operational_run_directory(repo / ".pingstore/runs" / present)
    assert record["stage"] == "present"
    assert record["inputs"]["analysis"]["run_id"] == analyse
    assert all(
        path.is_file()
        for path in (repo / ".pingstore/runs" / present / "export").iterdir()
    )
    assert {row["id"] for row in discover_runs(repo / ".pingstore/runs")} == {present}
    assert not (repo / ".artifacts").exists()
    for name, original in before.items():
        assert (repo / ".pingstore/runs" / name / "run.json").read_bytes() == original
    if modules["recipe"].SLUG == "exp077":
        result = load_json(repo / ".pingstore/runs" / present / "export/numbers.json")
        assert result["parity_performance"]["parity_pass"] is True
        assert result["parity_performance"]["performance_gate_pass"] is False


@pytest.mark.parametrize(
    "corruption",
    ["payload", "v2", "wrong-stage", "wrong-experiment", "recipe"],
)
def test_invalid_evidence_is_rejected_before_allocation(lab, corruption):
    repo, modules, compute, analyse = chain(lab)
    path = repo / ".pingstore/runs" / compute
    record = load_json(path / "run.json")
    if corruption == "payload":
        (path / "export/changed.txt").write_text("changed")
    else:
        if corruption == "manifest":
            record["execution"]["extra"] = "changed"
        if corruption == "v2":
            record["schema"] = "pingstore.run/v2"
        if corruption == "wrong-stage":
            record["stage"] = "present"
        if corruption == "wrong-experiment":
            record["experiment"] = "exp099"
        if corruption == "recipe":
            record["execution"]["configuration"] = {}
        write_json_atomic(path / "run.json", record)
    before = {p.name for p in path.parent.iterdir()}
    with pytest.raises(PingstoreError):
        modules["present"].present(analyse)
    assert {p.name for p in path.parent.iterdir()} == before


def test_plot_failure_keeps_hidden_run_and_preserves_sources(lab, monkeypatch):
    repo, modules, compute, analyse = chain(lab)

    def broken(*a, **kw):
        raise RuntimeError("plot failed")

    plotname = {
        "exp074": "plot_rasters",
        "exp075": "plot_training",
        "exp076": "plot_training",
        "exp077": "render_rasters",
    }[modules["recipe"].SLUG]
    monkeypatch.setattr(modules["plots"], plotname, broken)
    with pytest.raises(RuntimeError, match="plot failed"):
        modules["present"].present(analyse)
    roots = repo / ".pingstore/runs"
    assert len(list(roots.glob(".*-present.tmp"))) == 1
    assert not list(roots.glob("*-present"))
    for name in (compute, analyse):
        validate_operational_run_directory(roots / name)


def test_transitive_source_change_during_presentation_aborts(lab, monkeypatch):
    repo, modules, compute, analyse = chain(lab)
    plotname = {
        "exp074": "plot_rasters",
        "exp075": "plot_training",
        "exp076": "plot_training",
        "exp077": "render_rasters",
    }[modules["recipe"].SLUG]
    original = getattr(modules["plots"], plotname)

    def mutate(*args, **kwargs):
        original(*args, **kwargs)
        (repo / ".pingstore/runs" / compute / "export/changed.txt").write_text(
            "changed"
        )

    monkeypatch.setattr(modules["plots"], plotname, mutate)
    with pytest.raises(PingstoreError):
        modules["present"].present(analyse)
    assert not list((repo / ".pingstore/runs").glob("*-present"))


def test_selected_article_renders_results_before_methods(lab):
    import re
    import shutil
    import subprocess

    from demolab_cli import _paths
    from PIL import Image

    repo, modules, compute, analyse = chain(lab)
    present = modules["present"].present(analyse)
    slug = modules["recipe"].SLUG
    # Synthetic images are only layout fixtures, never retained scientific runs.
    selected = repo / "selected"
    shutil.copytree(repo / ".pingstore/runs" / present / "export", selected)
    for path in selected.glob("*.png"):
        Image.new("RGB", (2, 2), "white").save(path)
    shutil.copytree(ROOT / "writings", repo / "writings")
    (repo / ".demolab").mkdir()
    (repo / ".demolab/VERSION").write_text("test")
    for filename in ("lib.typ", "style.css"):
        shutil.copy2(_paths.TYP / filename, repo / ".demolab" / filename)
    write_json_atomic(repo / "selection.json", {slug: {slug: "/selected"}})
    (repo / "article.typ").write_text(
        '#import "/.demolab/lib.typ": entry-page\n'
        f'#import "/writings/{slug}.typ": meta, body\n#entry-page(meta, body)\n'
    )
    result = subprocess.run(
        [
            _paths.find_typst(ROOT),
            "compile",
            "--features",
            "html",
            "--format",
            "html",
            "--root",
            str(repo),
            "--input",
            "demolab-preview-file=/selection.json",
            str(repo / "article.typ"),
            str(repo / "article.html"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    html = re.sub(r"<style\b.*?</style>", "", (repo / "article.html").read_text(), flags=re.S)
    headings = [
        re.sub("<[^>]+>", "", text).replace("#", "").strip()
        for text in re.findall(r"<h3\b[^>]*>(.*?)</h3>", html, re.S)
    ]
    assert headings == [
        "Table of Contents",
        "Abstract",
        "Datasets",
        "Results",
        "Methods",
        "References",
    ]
    assert "A required run is unavailable" not in html
    assert html.count("<figcaption") >= 2


def test_failed_compute_never_exposes_a_run(lab, monkeypatch):
    repo, modules = lab

    def fail(*args, **kwargs):
        raise RuntimeError("acquisition failed")

    if modules["recipe"].SLUG == "exp077":
        monkeypatch.setattr(modules["compute"], "simulate", fail)
    else:
        monkeypatch.setattr(boundary, "command", fail)
    with pytest.raises(RuntimeError, match="acquisition failed"):
        modules["compute"].compute()
    roots = repo / ".pingstore/runs"
    assert len(list(roots.glob(".*-compute.tmp"))) == 1
    assert not list(roots.glob("*-compute"))


def test_supplied_reservation_is_single_use(lab):
    repo, modules = lab
    identity = stages.reserve_stage(
        repo / ".pingstore", modules["recipe"].SLUG, "compute"
    )
    assert modules["compute"].compute(run_id=identity) == identity
    with pytest.raises(PingstoreError, match="unused reserved"):
        modules["compute"].compute(run_id=identity)


def test_simulator_execution_attachments_are_retained_as_provenance(tmp_path):
    import sys

    output = tmp_path / "export"
    output.mkdir()
    provenance = tmp_path / "export/evidence"
    provenance.mkdir()
    for name in ("run.sh", "run.jsonl", "output.log", "metrics.json"):
        (output / name).write_text(name)
    boundary.command(
        tmp_path,
        provenance,
        "simulator",
        [sys.executable, "-c", "pass", "--out-dir", str(output)],
    )
    assert sorted(path.name for path in output.iterdir()) == ["evidence", "metrics.json"]
    assert {path.name for path in (provenance / "simulator").iterdir()} == {
        "run.jsonl",
        "output.log",
    }
