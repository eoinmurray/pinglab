from __future__ import annotations

import numpy as np
from experiments.exp085 import DT_MS, N_INPUT, author_network
from experiments.exp086 import (
    K_VALUES,
    PHASE_BINS,
    analyse_trajectory,
    circular_distance,
    instantaneous_frequency,
    make_inputs,
)

"""Exp086 contracts with synthetic stages and one 20-step simulator regression."""

import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from execution import GraphRuntimeState, load_runtime_state
from experiments.exp086 import (
    analyse,
    compute,
    evidence,
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


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(
        stages, "memberships", lambda _: {"exp086": "demo", "exp085": "demo"}
    )
    monkeypatch.setattr(
        stages, "_capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    # Test-only duration; no production CLI or smoke recipe is introduced.
    monkeypatch.setattr(recipe, "T_MS", 1_500.0)
    monkeypatch.setitem(recipe.SCALE, "t_ms", 1_500.0)
    calls = []

    def fake_simulate(spec, *, runtime_state=None):
        steps = next(iter(spec.inputs.values())).shape[0]
        calls.append((spec, runtime_state))
        assert spec.seed == 85
        if runtime_state is not None:
            assert runtime_state.completed_steps == 5_000
            assert runtime_state.voltages["fixture"].item() == 1.0
            runtime_state.voltages["fixture"].zero_()
        recordings = {}
        for index, (name, shape) in enumerate(evidence.recording_shapes(steps).items()):
            spikes = torch.zeros(shape, dtype=torch.uint8)
            spikes[50 :: (200 if index < 2 else 220), 0, :] = 1
            recordings[name] = spikes
        state = GraphRuntimeState(
            signature="synthetic-fixture",
            compatibility={},
            completed_steps=steps,
            voltages={"fixture": torch.ones(1)},
            refractory={},
            conductances={},
            population_histories={},
            input_histories={},
        )
        return SimpleNamespace(recordings=recordings, runtime_state=state)

    monkeypatch.setattr(compute, "simulate", fake_simulate)
    return tmp_path, calls


def resign(directory):
    record = load_json(directory / "run.json")
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)


def fail(*args, **kwargs):
    raise RuntimeError("deliberate test failure")


def test_stages_are_independent_and_preserve_arrays(repo, monkeypatch):
    root, calls = repo
    cid = compute.compute()
    assert cid == "exp086-r001-compute"
    assert len(calls) == 10
    prefix, _ = calls[0]
    assert len({id(spec.inputs) for spec, _ in calls[1:]}) == 1
    assert len({id(state) for _, state in calls[1:]}) == 9
    for spec, _ in calls[1:]:
        for key, tensor in spec.inputs.items():
            assert (
                tensor.untyped_storage().data_ptr()
                == prefix.inputs[key].untyped_storage().data_ptr()
            )
    source = inputs.source(root, cid, "compute")
    original = source.reference
    assert source.record["inputs"] == {}
    state = load_runtime_state(source.export / "prefix-state")
    assert state.completed_steps == 5_000
    assert state.voltages["fixture"].item() == 1.0
    assert not list(source.export.rglob("*.png"))
    assert not (source.export / "results.json").exists()
    with np.load(source.export / "inputs.npz") as raw:
        for key, expected in recipe.make_inputs().items():
            np.testing.assert_array_equal(raw[key], expected.numpy())
    monkeypatch.setattr(compute, "simulate", fail)
    aid = analyse.analyse(cid)
    analysis = inputs.source(root, aid, "analyse")
    assert analysis.record["inputs"] == {"compute": original}
    result = load_json(analysis.export / "results.json")
    with np.load(source.file("branches", "k_0p070", "spikes.npz")) as raw:
        expected = measurements.analyse_trajectory(dict(raw), k=0.07)
    assert result["trajectories"][1] == measurements.public_summary(expected)
    with np.load(analysis.export / "k_0p070.npz") as raw:
        assert set(raw.files) == set(recipe.ARRAY_KEYS)
        for key in recipe.ARRAY_KEYS:
            np.testing.assert_array_equal(raw[key], expected[key])
    assert discover_runs(root / ".pingstore/runs") == []
    monkeypatch.setattr(measurements, "analyse_trajectory", fail)
    monkeypatch.setattr(measurements, "choose_intermediate", fail)
    pid = present.present(aid)
    output = inputs.source(root, pid, "present")
    numbers = load_json(output.presentation / "numbers.json")
    assert all(numbers[k] == v for k, v in result.items())
    assert set(recipe.FIGURES) <= {p.name for p in output.presentation.iterdir()}
    assert all(p.is_file() for p in output.presentation.iterdir())
    assert output.record["inputs"]["analysis"] == analysis.reference
    assert set(output.record["inputs"]) == {"analysis"}
    assert not (output.presentation / "coupling_regimes.svg").exists()
    assert not (output.presentation / "intermittent_attraction.svg").exists()
    assert inputs.source(root, cid, "compute").reference == original
    assert not (root / ".artifacts").exists()
    assert len(calls) == 10


@pytest.mark.parametrize("stage", ["compute", "analyse", "present"])
def test_failure_stays_hidden_and_cannot_resume(repo, monkeypatch, stage):
    root, _ = repo
    if stage == "compute":
        monkeypatch.setattr(compute, "simulate", fail)
        command = compute.compute
    else:
        cid = compute.compute()
        if stage == "analyse":
            monkeypatch.setattr(measurements, "analyse_trajectory", fail)
            command = partial(analyse.analyse, cid)
        else:
            aid = analyse.analyse(cid)
            monkeypatch.setattr(present.plots, "plot_uncoupled", fail)
            command = partial(present.present, aid)
    runs = root / ".pingstore/runs"
    before = {p.name for p in runs.glob("exp086-*")}
    with pytest.raises(RuntimeError, match="deliberate test failure"):
        command()
    assert {p.name for p in runs.glob("exp086-*")} == before
    hidden = list(runs.glob(".exp086-*.tmp"))
    assert len(hidden) == 1
    with pytest.raises(PingstoreError, match="interrupted"):
        command(run_id=hidden[0].name[1:-4])


@pytest.mark.parametrize(
    "corruption", ["payload", "nested", "root", "symlink", "v2"]
)
def test_transitive_ancestry_corruption_is_rejected(repo, corruption):
    root, _ = repo
    cid = compute.compute()
    aid = analyse.analyse(cid)
    source = inputs.source(root, cid, "compute")
    if corruption == "payload":
        (source.export / "inputs.npz").write_bytes(b"changed")
    elif corruption == "manifest":
        record = load_json(source.directory / "run.json")
        record["execution"]["operation"] = "changed"
        write_json_atomic(source.directory / "run.json", record)
    elif corruption == "nested":
        source.file(
            "branches", "k_0p080", "network.bundle", "manifest.json"
        ).write_text(
            "{}"
        )
    elif corruption == "root":
        (source.directory / "unexpected").write_text("invalid")
    elif corruption == "symlink":
        (source.export / "link").symlink_to(source.export / "inputs.npz")
    else:
        record = load_json(source.directory / "run.json")
        record["schema"] = "pingstore.run/v2"
        write_json_atomic(source.directory / "run.json", record)
    before = set((root / ".pingstore/runs").iterdir())
    with pytest.raises((PingstoreError, OSError)):
        present.present(aid)
    assert set((root / ".pingstore/runs").iterdir()) == before


def test_ancestor_mutation_during_presentation_blocks_completion(repo, monkeypatch):
    root, _ = repo
    cid = compute.compute()
    aid = analyse.analyse(cid)
    source = inputs.source(root, cid, "compute")

    def mutate(*args):
        (source.export / "inputs.npz").write_bytes(b"changed during presentation")

    monkeypatch.setattr(present.plots, "plot_uncoupled", mutate)
    monkeypatch.setattr(present.plots, "plot_coupling_regimes", lambda *a: None)
    monkeypatch.setattr(present.plots, "plot_intermittent_attraction", lambda *a: None)
    with pytest.raises(PingstoreError):
        present.present(aid)
    assert len(list((root / ".pingstore/runs").glob(".exp086-*.tmp"))) == 1


def test_wrong_stage_missing_source_and_changed_recipe_are_rejected(repo):
    root, _ = repo
    cid = compute.compute()
    aid = analyse.analyse(cid)
    with pytest.raises(PingstoreError):
        analyse.analyse(aid)
    with pytest.raises((PingstoreError, OSError)):
        analyse.analyse("exp086-r999-compute")
    source = inputs.source(root, cid, "compute")
    doc = load_json(source.export / "evidence.json")
    doc["recipe"]["t_ms"] += 1
    write_json_atomic(source.export / "evidence.json", doc)
    resign(source.directory)
    with pytest.raises(PingstoreError, match="recipe"):
        analyse.analyse(cid)


def test_invalid_spikes_rejected_with_valid_payload_digest(repo):
    root, _ = repo
    cid = compute.compute()
    source = inputs.source(root, cid, "compute")
    path = source.file("branches", "k_0p080", "spikes.npz")
    with np.load(path) as raw:
        arrays = dict(raw)
    arrays["population_0"][0, 0, 0] = 2
    np.savez_compressed(path, **arrays)
    resign(source.directory)
    with pytest.raises(PingstoreError, match="binary array"):
        analyse.analyse(cid)


def test_reservation_and_atomic_visibility(repo, monkeypatch):
    root, _ = repo
    rid = stages.reserve_stage(root / ".pingstore", "exp086", "compute", origin="local")
    real = compute.simulate

    def inspect(*args, **kwargs):
        assert not (root / ".pingstore/runs" / rid).exists()
        assert (root / ".pingstore/runs" / f".{rid}.tmp").is_dir()
        return real(*args, **kwargs)

    monkeypatch.setattr(compute, "simulate", inspect)
    assert compute.compute(run_id=rid) == rid
    assert (root / ".pingstore/runs" / rid).is_dir()
    assert not (root / ".pingstore/runs" / f".{rid}.tmp").exists()
    with pytest.raises(PingstoreError):
        compute.compute(run_id=rid)


def test_rejects_absent_prefix_state(repo, monkeypatch):
    real = compute.simulate

    def no_state(*args, **kwargs):
        result = real(*args, **kwargs)
        result.runtime_state = None
        return result

    monkeypatch.setattr(compute, "simulate", no_state)
    with pytest.raises(RuntimeError, match="no reusable runtime state"):
        compute.compute()


def test_selection_preserves_tie_order_and_rejects_no_candidate():
    row = {
        "k": 0.07,
        "phase_slips": 2,
        "phase_alignment_error_rad": 0.2,
        "phase_concentration": 0.4,
        "density_peak_to_mean": 2.0,
        "slowing_fraction": 0.5,
    }
    other = {**row, "k": 0.06}
    assert measurements.choose_intermediate([row, other]) is row
    with pytest.raises(RuntimeError, match="no intermediate"):
        measurements.choose_intermediate([{**row, "phase_slips": 1}])


@pytest.mark.parametrize("module", ["compute", "analyse", "present"])
@pytest.mark.parametrize("mode", ["module", "file"])
def test_stage_help_without_execution(module, mode):
    root = Path(__file__).resolve().parents[2]
    command = (
        ["-m", f"experiments.exp086.{module}"]
        if mode == "module"
        else [str(root / f"experiments/exp086/{module}.py")]
    )
    result = subprocess.run(
        [sys.executable, *command, "--help"], cwd=root, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert "--run-id" in result.stdout


def test_real_recipe_and_collection_membership():
    cfg = recipe.configuration()
    assert cfg["t_ms"] == 5_000.0
    assert cfg["dt_ms"] == 0.1
    assert cfg["input_seeds"] == [8501, 8502]
    assert cfg["network_seed"] == 85
    root = Path(__file__).resolve().parents[2]
    assert stages.memberships(root)["exp086"] == "demo"
    assert len(recipe.branches()) == 9


def test_combined_runner_refuses_implicit_execution():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "experiments.exp086"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "requires an explicit stage" in result.stderr


def test_real_simulator_full_profile_retains_all_exposed_spikes(tmp_path):
    steps = 20
    drives = {
        f"drive_A_{recipe.INPUT_RATE_A_HZ:g}_Hz": recipe.poisson_input(
            rate_hz=recipe.INPUT_RATE_A_HZ, seed=recipe.INPUT_SEEDS[0], steps=steps
        ),
        f"drive_B_{recipe.INPUT_RATE_B_HZ:g}_Hz": recipe.poisson_input(
            rate_hz=recipe.INPUT_RATE_B_HZ, seed=recipe.INPUT_SEEDS[1], steps=steps
        ),
    }
    result = compute.simulate(
        compute.ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=recipe.author_network().graph,
            inputs=drives,
            seed=recipe.NETWORK_SEED,
        )
    )
    assert any(key.endswith(".voltage") for key in result.recordings)
    assert any(key.endswith(".conductance") for key in result.recordings)
    path = tmp_path / "spikes.npz"
    compute.save_recordings(path, result.recordings, steps)
    actual = evidence.binary_arrays(path, evidence.recording_shapes(steps), np.uint8)
    for key, array in actual.items():
        np.testing.assert_array_equal(array, result.recordings[key].cpu().numpy())
    missing = {k: v for k, v in result.recordings.items() if k != "population_0"}
    with pytest.raises(PingstoreError, match="recorded populations"):
        compute.save_recordings(tmp_path / "missing.npz", missing, steps)


def test_article_renders_selected_run_and_handles_missing_data(repo):
    import re
    import shutil

    from demolab_cli import _paths
    from pingstore.presentation_inputs import projection

    root, _ = repo
    pid = present.present(analyse.analyse(compute.compute()))
    output = inputs.source(root, pid, "present")
    source_root = Path(__file__).resolve().parents[2]
    (root / "writings").mkdir()
    for name in ("exp086.typ", "contents.typ", "run-inputs.typ", "run-view.typ"):
        shutil.copy2(source_root / "writings" / name, root / "writings" / name)
    (root / ".demolab").mkdir()
    shutil.copy2(_paths.TYP / "lib.typ", root / ".demolab/lib.typ")
    catalogue = projection(root)
    inventory = root / ".demolab/pinglab-inputs.json"
    write_json_atomic(inventory, catalogue)
    document = root / "document.typ"
    document.write_text(
        '#set page(paper: "a4", margin: 18mm)\n#set text(size: 10pt)\n'
        '#import "writings/exp086.typ": body\n#body\n'
    )
    command = [
        _paths.find_typst(source_root),
        "compile",
        "--root",
        str(root),
        "--input",
        "demolab-url-render=true",
        "--input",
        "demolab-url-article=exp086",
        "--input",
        "source.exp086=/" + str(output.export.relative_to(root)),
    ]
    html_command = command + [
        "--features",
        "html",
        "--format",
        "html",
        str(document),
        str(root / "article.html"),
    ]
    rendered = subprocess.run(html_command, capture_output=True, text=True, timeout=60)
    assert rendered.returncode == 0, rendered.stderr
    assert "ignored during MathML export" not in rendered.stderr
    html = (root / "article.html").read_text()
    images = re.findall(r"<img\b[^>]*>", html)
    assert len(images) == 4
    assert html.count('class="exp086-figure"') == 4
    assert ".exp086-figure img {height:auto;max-width:100%;}" in html
    assert all('alt="' in tag and 'src="' in tag for tag in images)
    assert len(re.findall(r"<figcaption\b", html)) == 4
    assert len(re.findall(r"<h\d\b[^>]*>References</h\d>", html)) == 1
    assert html.count('<mover accent="true"><mi>𝑓</mi>') == 3
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    assert set(re.findall(r'href="#([^"]+)"', html)) <= ids
    assert pid in html and "No presentation runs available" not in html
    assert "whole net windings" in html
    assert "coupling_regimes.svg" not in html
    assert "intermittent_attraction.svg" not in html
    paged = subprocess.run(
        command
        + [
            "--format",
            "png",
            "--ppi",
            "80",
            str(document),
            str(root / "article-{p}.png"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert paged.returncode == 0, paged.stderr
    assert list(root.glob("article-*.png"))
    # Corruption is confined to this synthetic test store.
    (output.export / "numbers.json").write_text("broken JSON")
    rendered = subprocess.run(html_command, capture_output=True, text=True, timeout=60)
    assert rendered.returncode != 0
    catalogue["runs"] = []
    catalogue["display_runs"] = []
    write_json_atomic(inventory, catalogue)
    empty_command = html_command.copy()
    index = empty_command.index(
        "source.exp086=/" + str(output.export.relative_to(root))
    )
    del empty_command[index - 1 : index + 1]
    rendered = subprocess.run(empty_command, capture_output=True, text=True, timeout=60)
    assert rendered.returncode == 0, rendered.stderr
    html = (root / "article.html").read_text()
    assert "A required run is unavailable" in html
    assert "whole net windings" not in html and "<img " not in html


def test_coupling_sweep_has_locked_endpoint_intermediate_values_and_zero() -> None:
    assert K_VALUES[0] == 0.08
    assert K_VALUES[-1] == 0.0
    assert np.all(np.diff(K_VALUES) < 0)
    assert len(K_VALUES) > 2


def test_each_branch_uses_equal_e_to_e_and_e_to_i_strength() -> None:
    for k in (K_VALUES[0], K_VALUES[len(K_VALUES) // 2], K_VALUES[-1]):
        graph = author_network(k_ee=float(k), k_ei=float(k)).graph
        parameters = {row["id"]: row for row in graph["parameters"]}
        for source, target in (("PING_A", "PING_B"), ("PING_B", "PING_A")):
            ee = parameters[f"{source}_E_to_{target}_E_K_EE.weight"]["initializer"][
                "mean"
            ]
            ei = parameters[f"{source}_E_to_{target}_I_K_EI.weight"]["initializer"][
                "mean"
            ]
            assert ee == ei == float(k)


def test_fixed_input_generation_is_reproducible() -> None:
    first = make_inputs()
    second = make_inputs()
    assert first.keys() == second.keys()
    for name in first:
        assert first[name].shape[-1] == N_INPUT
        assert np.array_equal(first[name].numpy(), second[name].numpy())


def test_instantaneous_frequency_follows_intervolley_intervals() -> None:
    interval = round(25.0 / DT_MS)
    peaks = np.arange(0, 4 * interval, interval)
    frequency = instantaneous_frequency(peaks, steps=5 * interval)
    assert np.all(frequency[: 3 * interval] == 40.0)
    assert np.isnan(frequency[3 * interval :]).all()


def test_circular_distance_wraps_at_pi() -> None:
    np.testing.assert_allclose(
        circular_distance(-np.pi + 0.1, np.pi - 0.1),
        0.2,
        atol=1e-12,
    )


def test_phase_analysis_reports_expected_number_of_bins() -> None:
    steps = 10_000
    recordings = {}
    for index, period in enumerate((200, 200, 220, 220)):
        population = 80 if index % 2 == 0 else 20
        spikes = np.zeros((steps, 1, population), dtype=np.uint8)
        spikes[np.arange(50, steps, period), 0, :] = 1
        recordings[f"population_{index}"] = spikes
    analysis = analyse_trajectory(recordings, k=0.01)
    assert len(analysis["phase_bin_centres"]) == PHASE_BINS
    assert len(analysis["phase_density"]) == PHASE_BINS
