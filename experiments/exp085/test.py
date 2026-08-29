from __future__ import annotations

import numpy as np
import pytest
from experiments.exp085 import (
    COUPLING_DELAY_MS,
    CROSS_FAN_IN,
    CROSS_ZERO_FRACTION,
    DT_MS,
    E_REFRACTORY_MS,
    E_TO_I_TAU_MS,
    E_TO_I_WEIGHT,
    I_REFRACTORY_MS,
    INPUT_RATE_A_HZ,
    INPUT_RATE_B_HZ,
    K_EE,
    K_EI,
    N_E,
    N_I,
    N_INPUT,
    T_MS,
    analyse_event_aligned_mechanism,
    author_network,
    author_phase_response_network,
    inhibitory_cycle_summary,
    interpolated_phase,
    make_phase_response_inputs,
    make_uncoupled_inputs,
    population_volley_events,
    rhythm_summary,
)

"""Exp085 contracts use synthetic evidence, never production simulation."""

import subprocess
import sys
from pathlib import Path

import torch
from experiments.exp085 import (
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
    LEGACY_RUN_SCHEMA,
    PingstoreError,
    file_sha256,
    load_json,
    payload_digest,
    validate_operational_run_directory,
    write_json_atomic,
)
from pingstore.discovery import discover_runs
from pingstore.layout import initialize_layout


def synthetic(spec, *, runtime_state=None):
    """Only the executor is replaced; acquisition, measurement and plotting are real."""
    from execution import (
        ExecutionResult,
        GraphRuntimeState,
        plan_graph,
        runtime_state_compatibility,
        runtime_state_signature,
    )

    steps = len(next(iter(spec.inputs.values())))
    populations = {row["id"]: row["size"] for row in spec.graph["populations"]}
    prc = len(populations) == 2

    def spike_train(size, events):
        values = torch.zeros((steps, 1, size))
        events = np.asarray(events)
        values[events[(events >= 0) & (events < steps)]] = 1
        return values

    a_events = np.arange(100, steps, 250)
    i_events = a_events + 20
    if prc:
        left, right = 6850, 7100
        for target in ("E", "I"):
            pulse = spec.inputs[f"coupling_matched_pulse_to_{target}"]
            occupied = np.flatnonzero(pulse[:, 0, 0].numpy())
            if occupied.size:
                arrival = occupied[0] + round(recipe.COUPLING_DELAY_MS / recipe.DT_MS)
                fraction = (arrival - left) / (right - left)
                if target == "E" and fraction >= 0.5:
                    a_events = np.where(a_events == right, right - 15, a_events)
                if target == "I" and 0.10 <= fraction <= 0.14:
                    a_events = np.where(a_events == right, right + 40, a_events)
                    i_events = np.where(i_events == right + 20, right + 60, i_events)
                    i_events = np.append(i_events, left + 50)
    data = {
        "population_0": spike_train(recipe.N_E, a_events),
        "population_1": spike_train(recipe.N_I, i_events),
    }
    if not prc:
        params = {row["id"]: row for row in spec.graph["parameters"]}
        coupled = (
            runtime_state is not None
            and params["PING_A_E_to_PING_B_E_K_EE.weight"]["initializer"]["mean"] > 0
        )
        b_events = np.arange(180 if coupled else 190, steps, 250 if coupled else 270)
        data.update(
            population_2=spike_train(recipe.N_E, b_events),
            population_3=spike_train(recipe.N_I, b_events + 20),
        )
    for name, size in populations.items():
        data[f"{name}.voltage"] = (
            torch.full((steps, 1, size), -61.0) + torch.arange(size) / 100
        )
    for row in spec.graph["projections"]:
        target = row["target"].split(".")[0]
        data[f"{row['id']}.conductance"] = (
            torch.ones((steps, 1, populations[target])) * 0.125
        )
    plan = plan_graph(spec.graph)
    completed = steps + (runtime_state.completed_steps if runtime_state else 0)
    state = GraphRuntimeState(
        signature=runtime_state_signature(plan),
        compatibility=runtime_state_compatibility(plan),
        completed_steps=completed,
        voltages={
            name: torch.full((1, size), -61.0) for name, size in populations.items()
        },
        refractory={},
        conductances={},
        population_histories={},
        input_histories={},
    )
    return ExecutionResult(
        executor="graph",
        recordings=data
        if spec.recording_fields is None
        else {k: data[k] for k in spec.recording_fields},
        parameters={
            row["id"]: torch.ones(tuple(reversed(row["shape"])))
            for row in spec.graph["parameters"]
        },
        runtime_state=state,
        metrics={"synthetic_fixture": True},
    )


@pytest.fixture
def lab(tmp_path, monkeypatch):
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(stages, "memberships", lambda _: {"exp085": "demo"})
    monkeypatch.setattr(
        stages, "_capture_code", lambda *args: {"git_commit": "fixture", "dirty": False}
    )
    monkeypatch.setattr(compute, "simulate", synthetic)
    return tmp_path


@pytest.fixture(scope="module")
def completed_fixture(tmp_path_factory):
    """Build once and copy immutable synthetic evidence for independent failure tests."""
    root = tmp_path_factory.mktemp("exp085-fixture")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(compute, "REPO", root)
        patch.setattr(analyse, "REPO", root)
        patch.setattr(stages, "memberships", lambda _: {"exp085": "demo"})
        patch.setattr(
            stages,
            "_capture_code",
            lambda *args: {"git_commit": "fixture", "dirty": False},
        )
        patch.setattr(compute, "simulate", synthetic)
        identity = compute.compute()
        analysis = analyse.analyse(identity)
    return root, identity, analysis


@pytest.fixture
def retained(lab, completed_fixture):
    import shutil

    root, compute_id, analysis_id = completed_fixture
    shutil.copytree(root / ".pingstore", lab / ".pingstore")
    return lab, compute_id, analysis_id


def directory(repo, identity):
    return repo / ".pingstore/runs" / identity


def forbid(*args, **kwargs):
    pytest.fail("stage isolation violated")


def refresh(root):
    record = load_json(root / "run.json")
    record["payload_digest"] = payload_digest(root)
    write_json_atomic(root / "run.json", record)


def test_acquisition_preserves_all_probes_inputs_weights_and_branch_state(
    lab, monkeypatch
):
    calls = []

    def observe(spec, *, runtime_state=None):
        result = synthetic(spec, runtime_state=runtime_state)
        calls.append((spec, runtime_state))
        if runtime_state is not None:
            assert runtime_state.completed_steps == 5000
            # Every branch receives a detached copy, never another branch's state.
            assert runtime_state.voltages["PING_A_E"][0, 0] == -61
            runtime_state.voltages["PING_A_E"][0, 0] = 999
        return result

    monkeypatch.setattr(compute, "simulate", observe)
    monkeypatch.setattr(analyse, "analyse", forbid)
    monkeypatch.setattr(present, "present", forbid)
    monkeypatch.setattr(measurements, "analyse_phase_response", forbid)
    reserved = stages.reserve_stage(
        lab / ".pingstore", "exp085", "compute", origin="local"
    )
    identity = compute.compute(run_id=reserved)
    root = directory(lab, identity)
    record = validate_operational_run_directory(root)
    acquisition = load_json(root / "export/evidence.json")
    assert identity == "exp085-r001-compute"
    assert record["inputs"] == {} and record["collection"] == "demo"
    assert len(calls) == 49 and len(acquisition["probes"]) == 42
    assert [
        row["fraction"] for row in acquisition["probes"][:21]
    ] == recipe.PRC_PHASE_FRACTIONS.tolist()
    assert len({id(state) for _, state in calls[-4:]}) == 4
    for spec, _ in calls[-4:]:
        for name, tensor in spec.inputs.items():
            torch.testing.assert_close(tensor, calls[0][0].inputs[name][5000:])
    assert not list((root / "export").glob("*.png"))
    assert not (lab / ".artifacts").exists()
    assert discover_runs(lab / ".pingstore/runs") == []
    assert len(list((root / "export/evidence/simulations").glob("*.json"))) == 49


def check_article_render(lab, output):
    """Catch duplicate headings and punctuation lost at Typst code boundaries."""
    import json
    import re
    import shutil

    from demolab_cli import _paths

    repo = Path(__file__).resolve().parents[2]
    root = lab / "article-preview"
    (root / "writings").mkdir(parents=True)
    (root / ".demolab").mkdir()
    for name in ("exp085.typ", "contents.typ", "run-inputs.typ", "run-view.typ"):
        shutil.copy2(repo / "writings" / name, root / "writings" / name)
    shutil.copy2(_paths.TYP / "lib.typ", root / ".demolab/lib.typ")
    shutil.copytree(output / "export", root / "selected")
    (root / "preview.json").write_text(json.dumps({"exp085": {"exp085": "/selected"}}))
    source = root / "article.typ"
    source.write_text('#import "writings/exp085.typ": body\n#body\n')
    command = [
        str(_paths.find_typst(repo)),
        "compile",
        "--root",
        str(root),
        "--input",
        "demolab-preview-file=/preview.json",
    ]
    for filename, flags in (
        ("article.html", ["--features", "html", "--format", "html"]),
        ("article.pdf", []),
    ):
        compiled = subprocess.run(
            [*command, *flags, str(source), str(root / filename)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert compiled.returncode == 0, compiled.stderr
    html = (root / "article.html").read_text()
    headings = [
        re.sub("<[^>]+>", "", text).strip()
        for text in re.findall(r"<h[1-6]\b[^>]*>(.*?)</h[1-6]>", html, re.S)
    ]
    assert headings.count("References") == 1
    assert headings.index("Results") < headings.index("Methods")
    assert "; the lower panels" in re.sub("<[^>]+>", "", html)
    assert len(re.findall(r"<img\b", html)) == 6
    assert len(re.findall(r'<math display="block">', html)) == 1
    assert re.search(r'class="numbered-equation".*?<span>\(1\)</span>', html, re.S)
    abstract = re.search(
        r"<h[1-6][^>]*>Abstract</h[1-6]>(.*?)<h[1-6]", html, re.S
    ).group(1)
    assert 'href="#ref-' not in abstract
    assert 60 <= len(re.sub("<[^>]+>", " ", abstract).split()) <= 120
    results = re.search(
        r"<h[1-6][^>]*>Results</h[1-6]>(.*?)<h[1-6][^>]*>Methods", html, re.S
    ).group(1)
    without_figures = re.sub(r"<figure\b.*?</figure>", "", results, flags=re.S)
    without_headings = re.sub(r"<h[1-6]\b.*?</h[1-6]>", "", without_figures, flags=re.S)
    assert not re.sub("<[^>]+>", "", without_headings).strip()
    methods = re.search(
        r"<h[1-6][^>]*>Methods</h[1-6]>(.*?)<h[1-6][^>]*>References", html, re.S
    ).group(1)
    assert len(re.findall(r"<ol\b", methods)) == 1
    assert len(re.findall(r"<li\b", methods)) == 5
    assert len(re.findall(r"<strong\b", methods)) == 5
    assert (root / "article.pdf").read_bytes().startswith(b"%PDF")
    (root / "preview.json").write_text("{}")
    for filename, flags in (
        ("unavailable.html", ["--features", "html", "--format", "html"]),
        ("unavailable.pdf", []),
    ):
        compiled = subprocess.run(
            [*command, *flags, str(source), str(root / filename)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert compiled.returncode == 0, compiled.stderr
    unavailable = (root / "unavailable.html").read_text()
    assert "A required run is unavailable" in unavailable
    assert "Table of Contents" in unavailable
    assert not re.search(r"<(?:img|math)\b", unavailable)
    assert not re.search(r"<h[1-6][^>]*>(?:Abstract|Results|Methods)</h", unavailable)


def test_analysis_and_real_presentation_are_isolated_and_pinned(retained, monkeypatch):
    lab, compute_id, analysis_id = retained
    raw, analysis_root = directory(lab, compute_id), directory(lab, analysis_id)
    before = {
        path.name: (payload_digest(path), file_sha256(path / "run.json"))
        for path in (raw, analysis_root)
    }
    monkeypatch.setattr(compute, "compute", forbid)
    monkeypatch.setattr(compute, "simulate", forbid)
    reanalysis_id = analyse.analyse(compute_id)
    assert load_json(
        directory(lab, reanalysis_id) / "export/results.json"
    ) == load_json(analysis_root / "export/results.json")
    result = load_json(analysis_root / "export/results.json")["results"]
    assert result["uncoupled"]["PING_A"]["frequency_hz"] == 40
    assert result["uncoupled"]["PING_B"]["frequency_hz"] == pytest.approx(1000 / 27)
    assert len(result["phase_response"]["responses"]["E"]) == 21
    assert [
        row["phase_locked"] for row in result["pathway_comparison"]["conditions"]
    ] == [False, True, False, True]
    assert result["event_aligned_mechanism"]["next_target_volley_advance_ms"] == 1
    monkeypatch.setattr(analyse, "measure", forbid)
    monkeypatch.setattr(analyse, "analyse", forbid)
    monkeypatch.setattr(measurements, "analyse_phase_response", forbid)
    identity = present.present(analysis_id)
    output = directory(lab, identity)
    manifest = validate_operational_run_directory(output)
    assert manifest["inputs"] == {
        "analysis": stages.source_run(lab / ".pingstore", analysis_id).reference
    }
    assert set(inputs.lineage(lab, identity)) == {compute_id, analysis_id, identity}
    assert all(p.is_file() for p in (output / "export").iterdir())
    assert all(
        (output / "export" / name).stat().st_size > 1000 for name in recipe.FIGURES
    )
    numbers = load_json(output / "export/numbers.json")
    assert {key: numbers[key] for key in result} == result
    check_schematic_labels_and_geometry(lab, output, analysis_root)
    assert load_json(output / "export/protocol.json") == result
    assert [row["id"] for row in discover_runs(lab / ".pingstore/runs")] == [identity]
    assert before == {
        path.name: (payload_digest(path), file_sha256(path / "run.json"))
        for path in (raw, analysis_root)
    }
    assert not (lab / ".artifacts").exists()
    check_article_render(lab, output)


def check_schematic_labels_and_geometry(lab, output, analysis_root):
    from xml.etree import ElementTree

    graph = load_json(analysis_root / "export/network.json")
    original = lab / "original-network.svg"
    present.snn.Bundle(
        graph=graph, training=None, manifest={}, diagnostics=[]
    ).visualise(original, view="circuit", expand_groups=recipe.PING_GROUPS)
    before = ElementTree.parse(original)
    after = ElementTree.parse(output / "export/network.svg")

    def geometry(tree):
        return [
            (
                node.tag,
                node.attrib,
                None if node.tag.rsplit("}", 1)[-1] in {"text", "title"} else node.text,
            )
            for node in tree.iter()
        ]

    assert geometry(before) == geometry(after)
    labels = [
        node.text
        for node in after.iter()
        if node.tag.rsplit("}", 1)[-1] in {"text", "title"}
    ]
    assert all("_" not in label and "batch" not in label for label in labels)
    assert {
        "Drive A: 300 Hz",
        "Drive B: 260 Hz",
        "128 spike channels",
        "80 neurons",
        "20 neurons",
        "CONDUCTANCE LIF",
        "A excitatory",
        "A inhibitory",
        "B excitatory",
        "B inhibitory",
    } <= set(labels)


@pytest.mark.parametrize("tag", ["text", "title"])
def test_unrecognized_schematic_labels_fail_before_completion(
    retained, monkeypatch, tag
):
    lab, _, analysis_id = retained

    def unexpected_label(self, path, **kwargs):
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg"><{tag}>unexpected</{tag}></svg>'
        )

    monkeypatch.setattr(present.snn.Bundle, "visualise", unexpected_label)
    with pytest.raises(PingstoreError, match="unrecognized schematic"):
        present.present(analysis_id)
    assert discover_runs(lab / ".pingstore/runs") == []
    assert len(list((lab / ".pingstore/runs").glob(".*-present.tmp"))) == 1


@pytest.mark.parametrize("stage", ["compute", "analyse"])
def test_reject_v2_before_allocation(lab, stage):
    identity = f"exp085-r001-{stage}-local"
    root = directory(lab, identity)
    initialize_layout(root, "exp085", schema=LEGACY_RUN_SCHEMA)
    write_json_atomic(
        root / "run.json",
        {
            "schema": LEGACY_RUN_SCHEMA,
            "run_id": identity,
            "experiment": "exp085",
            "collection": "demo",
            "stage": stage,
            "origin": "local",
            "created_at": "2026-08-28T12:00:00+00:00",
            "inputs": {},
            "provenance": {},
            "execution": {},
            "payload_digest": payload_digest(root),
        },
    )
    with pytest.raises(PingstoreError, match="requires v4"):
        (analyse.analyse if stage == "compute" else present.present)(identity)
    assert list(root.parent.iterdir()) == [root]


def test_missing_wrong_stage_and_recipe_inputs_do_not_allocate(retained):
    lab, compute_id, analysis_id = retained
    runs = lab / ".pingstore/runs"
    before = sorted(p.name for p in runs.iterdir())
    for identity, function in [
        (analysis_id, analyse.analyse),
        (compute_id, present.present),
    ]:
        with pytest.raises(PingstoreError):
            function(identity)
    with pytest.raises((PingstoreError, FileNotFoundError)):
        analyse.analyse("exp085-r999-compute")
    record = load_json(directory(lab, compute_id) / "run.json")
    record["execution"]["configuration"]["network_seed"] = 99
    write_json_atomic(directory(lab, compute_id) / "run.json", record)
    with pytest.raises(PingstoreError, match="recipe"):
        analyse.analyse(compute_id)
    assert sorted(p.name for p in runs.iterdir()) == before


@pytest.mark.parametrize("target", ["payload", "layout", "symlink"])
def test_complete_ancestry_rejects_tampering(retained, target):
    lab, compute_id, analysis_id = retained
    root = directory(lab, compute_id)
    if target == "manifest":
        record = load_json(root / "run.json")
        record["execution"]["extra"] = "changed"
        write_json_atomic(root / "run.json", record)
    elif target == "payload":
        (root / "export/evidence/extra.txt").write_text("changed")
    elif target == "layout":
        (root / "unexpected.txt").write_text("bad root")
        refresh(root)
    else:
        (root / "export/link").symlink_to(root / "export/evidence.json")
    with pytest.raises(PingstoreError):
        present.present(analysis_id)
    assert not list(root.parent.glob("*present*"))


@pytest.mark.parametrize(
    "broken", ["grid", "recording", "pulse", "weights", "branch-input", "state"]
)
def test_reject_scientifically_incomplete_even_with_valid_checksum(retained, broken):
    lab, compute_id, _ = retained
    root = directory(lab, compute_id)
    if broken == "grid":
        path = root / "export/evidence.json"
        record = load_json(path)
        record["probes"].pop()
        write_json_atomic(path, record)
    elif broken == "state":
        path = root / "export/prefix-state/manifest.json"
        record = load_json(path)
        record["completed_steps"] -= 1
        write_json_atomic(path, record)
    else:
        name, filename = {
            "recording": ("prc-I-05", "recordings.npz"),
            "pulse": ("prc-I-05", "inputs.npz"),
            "weights": ("uncoupled", "parameters.npz"),
            "branch-input": ("pathway-both", "inputs.npz"),
        }[broken]
        path = root / "export/jobs" / name / filename
        data = evidence.arrays(path)
        if broken == "recording":
            del data["PING_A_I.voltage"]
        elif broken == "pulse":
            data["coupling_matched_pulse_to_I"][:] = 0
        elif broken == "weights":
            data.pop(next(iter(data)))
        else:
            data["drive_A_300_Hz"][0, 0, 0] = 1 - data["drive_A_300_Hz"][0, 0, 0]
        np.savez_compressed(path, **data)
    refresh(root)
    with pytest.raises(PingstoreError):
        analyse.analyse(compute_id)
    assert len(list(root.parent.glob("exp085-*-analyse"))) == 1


@pytest.mark.parametrize("stage", ["compute", "analyse", "present"])
def test_failed_stages_stay_hidden_and_do_not_resume(retained, monkeypatch, stage):
    lab, compute_id, analysis_id = retained
    module = {"compute": compute, "analyse": analyse, "present": present}[stage]
    worker = {"compute": "acquire", "analyse": "measure", "present": "render"}[stage]

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(module, worker, fail)
    function = getattr(module, stage)
    args = (
        ()
        if stage == "compute"
        else (compute_id if stage == "analyse" else analysis_id,)
    )
    identity = stages.reserve_stage(lab / ".pingstore", "exp085", stage)
    with pytest.raises(RuntimeError, match="injected"):
        function(*args, run_id=identity)
    assert not directory(lab, identity).exists()
    hidden = directory(lab, f".{identity}.tmp")
    assert hidden.is_dir()
    before = payload_digest(hidden)
    with pytest.raises(PingstoreError, match="interrupted"):
        function(*args, run_id=identity)
    assert payload_digest(hidden) == before
    assert discover_runs(lab / ".pingstore/runs") == []


@pytest.mark.parametrize("stage", ["analyse", "present"])
def test_source_mutation_during_stage_never_completes(retained, monkeypatch, stage):
    lab, compute_id, analysis_id = retained
    root = directory(lab, compute_id)
    module, worker = (analyse, "measure") if stage == "analyse" else (present, "render")
    original = getattr(module, worker)

    def mutate(*args):
        if stage == "present":
            for name in recipe.FIGURES:
                (args[0] / name).write_text("fixture")
            result = None
        else:
            result = original(*args)
        (root / "export/evidence/late-change.txt").write_text("changed ancestor")
        return result

    monkeypatch.setattr(module, worker, mutate)
    before = {p.name for p in root.parent.iterdir() if not p.name.startswith(".")}
    with pytest.raises(PingstoreError):
        getattr(module, stage)(compute_id if stage == "analyse" else analysis_id)
    assert {
        p.name for p in root.parent.iterdir() if not p.name.startswith(".")
    } == before
    assert list(root.parent.glob(f".exp085-*-{stage}.tmp"))


@pytest.mark.parametrize("stage", ["compute", "analyse", "present"])
def test_cli_requires_explicit_stages_and_inputs(tmp_path, stage):
    root = Path(__file__).resolve().parents[2]
    script = root / "experiments/exp085" / f"{stage}.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--run-id" in result.stdout
    if stage != "compute":
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0 and "--source" in result.stderr
    assert not (tmp_path / ".pingstore").exists()


def test_imports_and_retired_entrypoints_never_dispatch(tmp_path):
    root = Path(__file__).resolve().parents[2]
    expression = (
        "import sys; from unittest.mock import patch; "
        f"sys.path[:0] = [{str(root)!r}, {str(root / 'tools')!r}]; "
        "patch('subprocess.run', side_effect=AssertionError('dispatch')).start(); "
        "from experiments.exp085 import recipe, measurements, compute, analyse, present; "
        "import experiments.exp086"
    )
    result = subprocess.run(
        [sys.executable, "-c", expression],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [sys.executable, "-m", "experiments.exp085"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0 and "requires independent stages" in result.stderr


@pytest.mark.parametrize("kind", ["none", "prc"])
def test_real_executor_retention_interface_in_three_timesteps(tmp_path, kind):
    """A 0.3 ms interface check, not the 49-job scientific experiment."""
    from execution import (
        ExecutionSpec,
        load_runtime_state,
        save_runtime_state,
        simulate,
    )

    graph = recipe.graphs()[kind]
    drive = (
        recipe.make_phase_response_inputs()
        if kind == "prc"
        else recipe.make_uncoupled_inputs()
    )
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            device="cpu",
            inputs={name: value[:3] for name, value in drive.items()},
            seed=recipe.NETWORK_SEED,
        )
    )
    parameters = compute.numpy_tensors(result.parameters)
    assert {name: value.shape for name, value in parameters.items()} == {
        row["id"]: tuple(reversed(row["shape"])) for row in graph["parameters"]
    }
    raw = compute.numpy_tensors(result.recordings)
    assert raw["population_0"].shape == (3, 1, recipe.N_E)
    assert raw["PING_A_I.voltage"].shape == (3, 1, recipe.N_I)
    job = {
        "id": "prc-baseline" if kind == "prc" else "pathway-none",
        "graph": kind,
        "steps": 3,
    }
    folder = tmp_path / "jobs" / job["id"]
    folder.mkdir(parents=True)
    np.savez_compressed(folder / "recordings.npz", **raw)
    assert set(evidence.recording(tmp_path, job)) == set(raw)
    assert result.runtime_state is not None
    save_runtime_state(tmp_path / "state", result.runtime_state)
    assert load_runtime_state(tmp_path / "state").completed_steps == 3
    state = load_json(tmp_path / "state/manifest.json")
    assert state["tensors_digest"] == "sha256:" + file_sha256(
        tmp_path / "state/tensors.npz"
    )
    write_json_atomic(tmp_path / "execution-metrics.json", result.metrics)


def test_completion_is_validated_before_atomic_rename(retained, monkeypatch):
    lab, compute_id, _ = retained
    rename = stages.os.rename
    completed = []

    def inspect(source, destination):
        assert source.name.startswith(".") and not destination.exists()
        record = validate_operational_run_directory(source)
        assert record["stage"] == "analyse"
        assert record["inputs"]["compute"]["run_id"] == compute_id
        assert (source / "export/plot-arrays.npz").is_file()
        completed.append(destination)
        return rename(source, destination)

    monkeypatch.setattr(stages.os, "rename", inspect)
    identity = analyse.analyse(compute_id)
    assert completed == [directory(lab, identity)]
    assert completed[0].is_dir()


@pytest.mark.parametrize("broken", ["pin", "results", "graph", "plot-array"])
def test_presentation_rejects_inconsistent_analysis_before_reservation(
    retained, broken
):
    lab, _, analysis_id = retained
    root = directory(lab, analysis_id)
    if broken == "pin":
        record = load_json(root / "run.json")
        record["inputs"]["compute"]["payload_digest"] = "sha256:" + "0" * 64
        write_json_atomic(root / "run.json", record)
    elif broken in ("results", "graph"):
        path = (
            root
            / "export"
            / ("results.json" if broken == "results" else "network.json")
        )
        record = load_json(path)
        if broken == "results":
            record["recipe"]["network_seed"] += 1
        else:
            record["populations"][0]["size"] += 1
        write_json_atomic(path, record)
    else:
        path = root / "export/plot-arrays.npz"
        arrays = evidence.arrays(path)
        arrays.pop(next(iter(arrays)))
        np.savez_compressed(path, **arrays)
    refresh(root)
    with pytest.raises(PingstoreError):
        present.present(analysis_id)
    assert not list(root.parent.glob("*present*"))


@pytest.fixture(scope="module")
def graph() -> dict:
    return author_network().graph


def test_network_contains_two_matched_ping_circuits(graph: dict) -> None:
    populations = {row["id"]: row for row in graph["populations"]}
    assert populations["PING_A_E"]["size"] == N_E
    assert populations["PING_B_E"]["size"] == N_E
    assert populations["PING_A_I"]["size"] == N_I
    assert populations["PING_B_I"]["size"] == N_I

    assert populations["PING_A_E"]["neuron"] == populations["PING_B_E"]["neuron"]
    assert populations["PING_A_I"]["neuron"] == populations["PING_B_I"]["neuron"]
    assert populations["PING_A_E"]["neuron"]["refractory_steps"] == round(
        E_REFRACTORY_MS / DT_MS
    )
    assert populations["PING_A_I"]["neuron"]["refractory_steps"] == round(
        I_REFRACTORY_MS / DT_MS
    )


def test_network_has_only_local_e_to_i_to_e_ping_loops(graph: dict) -> None:
    projections = {row["id"]: row for row in graph["projections"]}
    parameters = {row["id"]: row for row in graph["parameters"]}
    for name in ("PING_A", "PING_B"):
        assert projections[f"{name}_E_to_I"]["target"] == f"{name}_I.excitatory"
        assert projections[f"{name}_I_to_E"]["target"] == f"{name}_E.inhibitory"
        assert f"{name}_E_to_E" not in projections
        assert f"{name}_I_to_I" not in projections
        e_to_i = projections[f"{name}_E_to_I"]
        assert e_to_i["synapse"]["tau"]["value"] == E_TO_I_TAU_MS
        initializer = parameters[f"{name}_E_to_I.weight"]["initializer"]
        assert initializer["mean"] == E_TO_I_WEIGHT


def test_cross_network_paths_are_reciprocal_and_separately_weighted(
    graph: dict,
) -> None:
    projections = {row["id"]: row for row in graph["projections"]}
    parameters = {row["id"]: row for row in graph["parameters"]}
    expected = {
        "PING_A_E_to_PING_B_E_K_EE": K_EE,
        "PING_A_E_to_PING_B_I_K_EI": K_EI,
        "PING_B_E_to_PING_A_E_K_EE": K_EE,
        "PING_B_E_to_PING_A_I_K_EI": K_EI,
    }

    for projection_id, strength in expected.items():
        projection = projections[projection_id]
        assert projection["connection"] == "feedback"
        assert projection["delay"]["value"] == COUPLING_DELAY_MS
        initializer = parameters[f"{projection_id}.weight"]["initializer"]
        assert initializer["mean"] == strength
        assert initializer["initial_zero_fraction"] == pytest.approx(
            CROSS_ZERO_FRACTION
        )
        assert initializer["zeroing"] == "exact_k"

    realised_fan_in = round((1.0 - CROSS_ZERO_FRACTION) * N_E)
    assert realised_fan_in == CROSS_FAN_IN


def test_pathway_branches_can_share_one_runtime_state() -> None:
    from execution import plan_graph, runtime_state_signature

    signatures = {
        runtime_state_signature(plan_graph(author_network(k_ee=k_ee, k_ei=k_ei).graph))
        for k_ee, k_ei in (
            (0.0, 0.0),
            (K_EE, 0.0),
            (0.0, K_EI),
            (K_EE, K_EI),
        )
    }

    assert len(signatures) == 1


def test_phase_response_paths_match_the_two_coupling_paths() -> None:
    probe_graph = author_phase_response_network().graph
    projections = {row["id"]: row for row in probe_graph["projections"]}
    parameters = {row["id"]: row for row in probe_graph["parameters"]}
    expected = {
        "probe_E_to_PING_A_E_K_EE": ("PING_A_E.excitatory", K_EE),
        "probe_E_to_PING_A_I_K_EI": ("PING_A_I.excitatory", K_EI),
    }
    for projection_id, (target, strength) in expected.items():
        assert projections[projection_id]["target"] == target
        initializer = parameters[f"{projection_id}.weight"]["initializer"]
        assert initializer["mean"] == strength
        assert initializer["initial_zero_fraction"] == pytest.approx(
            CROSS_ZERO_FRACTION
        )


def test_uncoupled_inputs_use_the_two_design_rates() -> None:
    inputs = make_uncoupled_inputs()
    duration_s = T_MS / 1_000.0
    realised_a = (
        float(inputs[f"drive_A_{INPUT_RATE_A_HZ:g}_Hz"].sum()) / N_INPUT / duration_s
    )
    realised_b = (
        float(inputs[f"drive_B_{INPUT_RATE_B_HZ:g}_Hz"].sum()) / N_INPUT / duration_s
    )
    assert realised_a == pytest.approx(INPUT_RATE_A_HZ, rel=0.03)
    assert realised_b == pytest.approx(INPUT_RATE_B_HZ, rel=0.03)


def test_phase_response_input_places_one_full_probe_volley() -> None:
    arrival_step = 100
    inputs = make_phase_response_inputs(target="I", arrival_step=arrival_step)
    pulse_e = inputs["coupling_matched_pulse_to_E"]
    pulse_i = inputs["coupling_matched_pulse_to_I"]
    delay_steps = round(COUPLING_DELAY_MS / DT_MS)

    assert pulse_e.sum() == 0
    assert pulse_i.sum() == N_E
    assert pulse_i[arrival_step - delay_steps].sum() == N_E


def test_phase_and_frequency_follow_detected_volley_intervals() -> None:
    interval_steps = round(25.0 / DT_MS)
    peaks = np.arange(0, 5 * interval_steps, interval_steps)
    summary = rhythm_summary(peaks)
    phase = interpolated_phase(peaks, steps=6 * interval_steps)

    assert summary["frequency_hz"] == 40.0
    assert summary["iei_cv"] == 0.0
    np.testing.assert_allclose(
        phase[:interval_steps],
        2.0 * np.pi * np.arange(interval_steps) / interval_steps,
    )


def test_inhibitory_cycle_summary_counts_each_neuron_between_volleys() -> None:
    spikes = np.zeros((8, 1, 2), dtype=np.uint8)
    spikes[1, 0] = 1
    spikes[5, 0] = 1

    summary = inhibitory_cycle_summary(spikes, np.array([0, 4, 8]))

    assert summary == {
        "cycles": 2,
        "mean_spikes_per_neuron": 1.0,
        "minimum": 1,
        "maximum": 1,
    }


def test_population_volley_events_groups_adjacent_timesteps() -> None:
    spikes = np.zeros((10, 1, 3), dtype=np.uint8)
    spikes[2, 0, :2] = 1
    spikes[3, 0, 2] = 1
    spikes[8, 0, :] = 1

    events = population_volley_events(spikes, start=0, stop=10)

    assert len(events) == 2
    assert events[0]["spikes"] == 3
    assert events[1]["spikes"] == 3


def test_event_aligned_mechanism_measures_target_volley_advance() -> None:
    steps = 500

    def spikes(size: int, at: int) -> np.ndarray:
        values = np.zeros((steps, 1, size), dtype=np.uint8)
        values[at, 0] = 1
        return values

    def conductance(size: int) -> np.ndarray:
        return np.zeros((steps, 1, size), dtype=np.float32)

    baseline = {
        "population_0": spikes(N_E, 100),
        "population_2": spikes(N_E, 300),
        "population_3": spikes(N_I, 310),
        "PING_A_E_to_PING_B_E_K_EE.conductance": conductance(N_E),
        "PING_B_I_to_E.conductance": conductance(N_E),
    }
    coupled = {
        "population_0": spikes(N_E, 100),
        "population_2": spikes(N_E, 290),
        "population_3": spikes(N_I, 300),
        "PING_A_E_to_PING_B_E_K_EE.conductance": conductance(N_E),
        "PING_B_I_to_E.conductance": conductance(N_E),
    }

    record, traces = analyse_event_aligned_mechanism(
        {"none": baseline, "e_to_e": coupled}
    )

    assert record["next_target_volley_advance_ms"] == pytest.approx(1.0)
    assert traces["time_from_arrival_ms"][0] == pytest.approx(-5.0)
