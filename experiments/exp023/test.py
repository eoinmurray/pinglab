from __future__ import annotations

from pathlib import Path

from experiments.exp023 import recipe as exp023

"""Isolated exp023 pipeline fixtures; never execute a scientific simulation."""

import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from experiments.exp023 import analyse, collection, compute, inputs, present, recipe
from experiments.helpers import run_cli
from pingstore import stages
from pingstore.contracts import (
    LEGACY_RUN_SCHEMA,
    PingstoreError,
    load_json,
    payload_digest,
    write_json_atomic,
)
from pingstore.discovery import discover_runs
from pingstore.layout import initialize_layout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(stages, "memberships", lambda _: {"exp023": "demo"})
    monkeypatch.setattr(
        stages, "_capture_code", lambda *args: {"git_commit": "fixture", "dirty": False}
    )
    monkeypatch.setattr(recipe, "N_E", 4)
    monkeypatch.setattr(recipe, "N_I", 2)
    monkeypatch.setenv("PINGLAB_SMOKE", "1")
    calls = []

    def simulate(args, **kwargs):
        calls.append(args)

        def arg(flag):
            return args[args.index(flag) + 1]

        destination = Path(arg("--out-dir"))
        destination.mkdir(parents=True)
        steps = int(float(arg("--t-ms")) / float(arg("--dt")))
        e = np.zeros((steps, recipe.N_E), dtype=bool)
        i = np.zeros((steps, recipe.N_I), dtype=bool)
        e[::100, 1] = True
        if arg("--ei-strength") != "0":
            i[5::100, 0] = True
        from config import save_snapshot_npz

        selected = args[args.index("--output-fields") + 1 :]
        selected = selected[
            : next(
                (i for i, x in enumerate(selected) if x.startswith("--")), len(selected)
            )
        ]
        save_snapshot_npz(
            destination / "snapshot.npz",
            {
                "hid": e,
                "inh": i,
                "input": np.zeros((steps, int(arg("--n-in"))), dtype=bool),
                "v_e_1": np.full(e.shape, -60.0),
                "ge_e_1": np.full(e.shape, 0.01),
                "gi_e_1": np.full(e.shape, 0.02 if i.any() else 0.0),
                "v_i_1": np.full(i.shape, -55.0),
                "ge_i_1": np.full(i.shape, 0.03),
            },
            float(arg("--dt")),
            recipe.N_E,
            recipe.N_I,
            output_fields=selected,
        )
        write_json_atomic(destination / "config.json", {"arguments": args})
        (destination / "run.sh").write_text("fixture only\n")

    monkeypatch.setattr(compute, "run_cli", simulate)
    return tmp_path, calls


def resign(directory):
    record = load_json(directory / "run.json")
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)


def test_independent_v3_stages_preserve_measurements_and_never_publish(
    repo, monkeypatch
):
    root, calls = repo
    compute_id = compute.compute()
    assert len(calls) == 16
    source = inputs.source(root, compute_id, "compute")
    assert source.record["inputs"] == {}
    assert not list(source.export.rglob("run.sh"))
    assert (source.directory / "provenance/simulations/scope/ping/run.sh").is_file()
    before = source.reference
    monkeypatch.setenv("PINGLAB_SMOKE", "0")
    monkeypatch.setattr(
        compute, "run_cli", lambda *a, **k: pytest.fail("downstream simulation")
    )
    analysis_id = analyse.analyse(compute_id)
    analysis = inputs.source(root, analysis_id, "analyse")
    results = load_json(analysis.export / "results.json")
    assert results["config"]["profile"] == "smoke"
    assert results["config"]["drive"]["fi_sweep"]["t_ms"] == 200
    assert results["raster"]["coba"]["e_rate_hz"] == pytest.approx(25)
    assert results["raster"]["ping"]["i_rate_hz"] == pytest.approx(50)
    assert results["raster"]["ping"]["e_index"] == 1
    assert results["f_gamma_hz"]["coba"] is None
    with np.load(analysis.export / "traces.npz") as data:
        np.testing.assert_allclose(data["ping__ii_e"], -0.4)
        np.testing.assert_allclose(data["ping__ie_e"], 0.6)
    assert discover_runs(root / ".pingstore/runs") == []
    monkeypatch.setattr(
        analyse, "population_psd", lambda *a: pytest.fail("presentation remeasurement")
    )
    monkeypatch.setattr(
        analyse, "select_traces", lambda *a: pytest.fail("presentation selection")
    )
    presentation_id = present.present(analysis_id)
    output = inputs.source(root, presentation_id, "present")
    assert output.record["inputs"] == {
        "analysis": analysis.reference,
        "compute": before,
    }
    assert (
        load_json(output.export / "numbers.json")["fi_curves"] == results["fi_curves"]
    )
    assert all(p.is_file() for p in output.export.iterdir())
    assert (output.export / "overview_compound.png").is_file()
    assert (output.export / "traces__ping__i_i.svg").is_file()
    assert not (output.directory / "presentation").exists()
    assert not (root / ".artifacts").exists()
    assert inputs.source(root, compute_id, "compute").reference == before
    assert [row["id"] for row in discover_runs(root / ".pingstore/runs")] == [
        presentation_id
    ]


def test_failure_leaves_hidden_run_and_never_runs_downstream(repo, monkeypatch):
    root, _ = repo

    def fail(*a, **k):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(compute, "run_cli", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        compute.compute()
    assert not list((root / ".pingstore/runs").glob("exp023-*"))
    assert len(list((root / ".pingstore/runs").glob(".exp023-*.tmp"))) == 1
    with pytest.raises(PingstoreError):
        analyse.analyse(".exp023-r001-compute-local.tmp")


def test_sources_and_authoritative_pins_are_validated(repo):
    root, _ = repo
    identity = compute.compute()
    with pytest.raises(PingstoreError, match="not a analyse"):
        present.present(identity)
    analysis_id = analyse.analyse(identity)
    source = inputs.source(root, identity, "compute")
    record_path = source.directory / "run.json"
    record = load_json(record_path)
    record["execution"]["note"] = "changed manifest, unchanged payload"
    write_json_atomic(record_path, record)
    with pytest.raises(PingstoreError, match="checksum changed"):
        present.present(analysis_id)


def test_missing_or_corrupt_snapshot_is_not_silently_recomputed(repo):
    root, calls = repo
    identity = compute.compute()
    source = inputs.source(root, identity, "compute")
    (source.export / "scope/ping/snapshot.npz").write_bytes(b"bad fixture")
    with pytest.raises(PingstoreError, match="checksum"):
        analyse.analyse(identity)
    resign(source.directory)
    with pytest.raises(ValueError):
        analyse.analyse(identity)
    assert len(calls) == 16


def test_unsupported_schema_is_rejected_before_scientific_consumption(repo):
    root, _ = repo
    identity = "exp023-r001-compute-local"
    directory = root / ".pingstore/runs" / identity
    initialize_layout(directory, "exp023", schema=LEGACY_RUN_SCHEMA)
    write_json_atomic(
        directory / "run.json",
        {
            "schema": LEGACY_RUN_SCHEMA,
            "run_id": identity,
            "experiment": "exp023",
            "stage": "compute",
            "inputs": {},
            "origin": "local",
            "collection": "demo",
            "created_at": "2026-08-27T12:00:00+00:00",
            "execution": {},
            "provenance": {},
            "payload_digest": payload_digest(directory),
        },
    )
    with pytest.raises(PingstoreError, match="requires v3"):
        analyse.analyse(identity)


def test_shared_cli_targets_existing_simulator():
    assert run_cli.SNN_TOOL.is_file()
    with patch("sh.uv") as command:
        run_cli.run_cli(recipe.raster_args("ping"), no_sync=True)
    assert str(run_cli.SNN_TOOL) in command.call_args.args


@pytest.mark.parametrize("flag", [[], ["--plot-only"], ["--skip-training"]])
def test_retired_launcher_rejects_all_combined_modes(flag):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "experiments.exp023", *flag],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "independent stages" in result.stderr


def test_collection_plans_use_reserved_stages_and_reject_legacy(repo):
    from experiments.collections.gamma_gated_sparsity.plan import build_plan

    root, _ = repo
    plan = build_plan(root / "campaign", "fixture")
    row = next(
        row
        for stage in plan["stages"]
        for row in stage["experiments"]
        if row["slug"] == "exp023"
    )
    assert row["execution"]["mode"] == "exp023-staged"
    assert row["command"] == []
    reservations = collection.reserve(root, row, origin="slurm-wilkes")
    assert set(reservations) == {"compute", "analyse", "present"}
    for stage, identity in reservations.items():
        assert identity.endswith("-" + stage)
        reservation = load_json(
            root
            / ".pingstore/runs"
            / f".{identity}.tmp"
            / "provenance/reservation.json"
        )
        assert reservation["origin"] == "slurm-wilkes"
    assert collection.reserve(root, row) == reservations
    with pytest.raises(PingstoreError, match="legacy exp023"):
        collection.require_staged({"execution": {"mode": "monolithic"}})


def test_collection_dispatches_explicit_stages_and_reuses_completed_chain(
    repo, monkeypatch
):
    from types import SimpleNamespace

    from experiments.collections.gamma_gated_sparsity.plan import build_plan

    root, _ = repo
    plan = build_plan(root / "campaign", "fixture", smoke=True)
    row = next(
        row
        for stage in plan["stages"]
        for row in stage["experiments"]
        if row["slug"] == "exp023"
    )
    calls = []

    def dispatch(command, **kwargs):
        calls.append(command)
        stage = command[2].rsplit(".", 1)[-1]
        run_id = command[command.index("--run-id") + 1]
        if stage == "compute":
            compute.compute(run_id=run_id)
        else:
            source_id = command[command.index("--source") + 1]
            getattr({"analyse": analyse, "present": present}[stage], stage)(
                source_id, run_id=run_id
            )
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(collection.subprocess, "run", dispatch)
    refs = collection.execute(root, plan, row)
    assert len(calls) == 3
    assert collection.execute(root, plan, row) == refs
    assert len(calls) == 3
    assert not (root / ".artifacts").exists()


def test_spectrum_and_selection_preserve_original_rules():
    spikes = np.zeros((4000, 4), dtype=bool)
    spikes[::250, :] = True
    _, _, peak = analyse.population_psd(spikes, 0.1, (5, 150))
    assert peak == pytest.approx(40, abs=0.01)
    assert analyse.pick_active(spikes) == 0
    assert analyse.pick_active(np.zeros_like(spikes)) is None
    assert analyse.population_psd(np.zeros_like(spikes), 0.1, (5, 150))[2] is None


def test_article_renders_selected_fixture_numbers_and_all_sections(repo):
    from demolab_cli import _paths

    root, _ = repo
    source_root = Path(__file__).resolve().parents[2]
    compute_id = compute.compute()
    analysis_id = analyse.analyse(compute_id)
    present_id = present.present(analysis_id)
    output = inputs.source(root, present_id, "present")
    shutil.copytree(source_root / "writings", root / "writings")
    (root / ".demolab").mkdir()
    shutil.copy2(_paths.TYP / "lib.typ", root / ".demolab/lib.typ")
    write_json_atomic(
        root / "preview.json",
        {"exp023": {"exp023": "/" + str(output.export.relative_to(root))}},
    )
    document = root / "document.typ"
    document.write_text(
        '#set page(paper: "a4", margin: 18mm)\n#set text(size: 10pt)\n'
        '#import "writings/exp023.typ": body\n#body\n'
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
        "100",
        str(document),
        str(root / "article-{p}.png"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert list(root.glob("article-*.png"))
    text = (root / "writings/exp023.typ").read_text()
    headings = [
        "== Abstract",
        "== Results",
        "== Methods",
        "#reference-list",
    ]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)
    for removed in ("Inputs and outputs", "Design Scope", "Prior art"):
        assert removed not in text
    assert "#cite(1)" in text
    assert "#cite(2)" not in text
    assert "same input as" not in text
    assert "≈ 30 Hz" not in text
    # Selected corrupt evidence must error, never show the unavailable notice.
    (output.export / "numbers.json").write_text("invalid fixture JSON")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0


@pytest.mark.parametrize("silent", [False, True])
def test_compact_snapshots_preserve_scope_analysis_and_fi_rates(tmp_path, silent):
    from config import save_snapshot_npz

    cfg = recipe.configuration(smoke=True)
    cfg.update(n_e=4, n_i=2)
    point = {"t_ms": 10, "dt_ms": 0.1, "n_in": 3}
    rng = np.random.default_rng(42)
    e = (rng.random((100, 4)) < 0.1) & (not silent)
    i = (rng.random((100, 2)) < 0.1) & (not silent)
    rec = {"hid": e, "inh": i, "input": np.zeros((100, 3), bool)}
    for p, n in (("e", 4), ("i", 2)):
        rec[f"v_{p}_1"] = rng.uniform(-70, -50, (100, n))
        for g in ("ge", "gi"):
            rec[f"{g}_{p}_1"] = rng.uniform(0, 0.2, (100, n))
    save_snapshot_npz(tmp_path / "full.npz", rec, 0.1, 4, 2)
    args = recipe.raster_args("ping")
    fields = args[args.index("--output-fields") + 1 :]
    save_snapshot_npz(tmp_path / "scope.npz", rec, 0.1, 4, 2, output_fields=fields)
    full = analyse.snapshot(tmp_path / "full.npz", cfg, point, traces=True)
    lean = analyse.snapshot(tmp_path / "scope.npz", cfg, point, traces=True)
    full_traces, full_selection = analyse.select_traces(full, cfg["biophysics"])
    lean_traces, lean_selection = analyse.select_traces(lean, cfg["biophysics"])
    assert lean_selection == full_selection
    assert "input_spikes" not in lean and "v_e_1" not in lean
    for key in full_traces:
        np.testing.assert_array_equal(lean_traces[key], full_traces[key])
    save_snapshot_npz(
        tmp_path / "fi.npz",
        rec,
        0.1,
        4,
        2,
        output_fields=["spk_e_count", "spk_i_count"],
    )
    counts = analyse.snapshot(tmp_path / "fi.npz", cfg, point)
    assert set(counts) == {"dt", "T", "n_e", "n_i", "spk_e_count", "spk_i_count"}
    for p in ("e", "i"):
        rate = float(
            counts[f"spk_{p}_count"]
            / (cfg[f"n_{p}"] * int(counts["T"]) * counts["dt"] / 1000.0)
        )
        assert rate == analyse.population_rate(full[f"spk_{p}"], full["dt"])


def _arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_raster_drive_provenance_matches_exact_execution_arguments() -> None:
    provenance = exp023.drive_provenance()["raster_operating_points"]
    for cell in exp023.CELLS:
        executed = exp023.raster_args(cell, Path("/tmp/ignored"))
        recorded = provenance[cell]
        assert recorded["input"] == _arg(executed, "--input") == "synthetic-spikes"
        assert recorded["input_rate_hz"] == float(_arg(executed, "--input-rate"))
        assert recorded["ei_strength"] == float(_arg(executed, "--ei-strength"))
        assert recorded["t_ms"] == float(_arg(executed, "--t-ms"))
        assert recorded["dt_ms"] == float(_arg(executed, "--dt"))


def test_fi_provenance_is_separate_and_matches_executed_grid() -> None:
    executed = [args for name, args in exp023.simulations() if name.startswith("fi/")]

    recorded = exp023.drive_provenance()["fi_sweep"]
    assert len(executed) == len(exp023.FI_EI) * len(exp023.FI_RATES_HZ)
    assert recorded["input_rates_hz"] == exp023.FI_RATES_HZ
    assert {float(_arg(args, "--input-rate")) for args in executed} == set(
        recorded["input_rates_hz"]
    )
    assert all(_arg(args, "--input") == recorded["input"] for args in executed)
    assert {cell: float(exp023.FI_EI[cell]) for cell in exp023.FI_EI} == recorded[
        "ei_strength_by_cell"
    ]


def test_geometry_and_smoke_duration_preserve_the_live_recipe():
    assert _arg(exp023.raster_args("coba"), "--n-in") == "1024"
    assert _arg(exp023.fi_args("coba", 2), "--n-in") == "784"
    assert _arg(exp023.raster_args("ping", smoke=True), "--t-ms") == "200"
    assert _arg(exp023.fi_args("ping", 2), "--t-ms") == "400"
