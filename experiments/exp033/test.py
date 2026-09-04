from __future__ import annotations

import numpy as np
from experiments.exp033 import numerics as exp033

"""Synthetic staged evidence and bounded numerical checks; no production runs."""

import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.exp033 import (
    analyse,
    appearance,
    collection,
    compute,
    evidence,
    inputs,
    measurements,
    numerics,
    plots,
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


def synthetic():
    cfg = recipe.configuration()

    def continuation(grid):
        rows = [
            {
                "I_ext": float(x),
                "fp": [0.004, 0.001, 0.008, 0.012],
                "eigs": [
                    [float(x) - 0.596, 0.17],
                    [float(x) - 0.596, -0.17],
                    [-1, 0],
                    [-2, 0],
                ],
            }
            for x in grid
        ]
        coarse = measurements.coarse_hopf(rows)
        h = {**coarse, "I_ext_star": 0.596, "leading_eigenvalue": [0, 0.17]}
        return {"sweep": rows, "hopf": h}

    def series(t, dimension, amplitude=0.001):
        return {
            "t_ms": t,
            "Y": np.array(
                [0.004 + amplitude * np.sin(t / 5 - i) for i in range(dimension)]
            ),
        }

    def ramp(h):
        grid = np.linspace(h["I_ext_star"] - 0.1, h["I_ext_star"] + 0.55, 25)
        rows = [
            {
                "I_ext": float(x),
                **series(
                    np.linspace(0, 2000, 101),
                    4,
                    np.sqrt(max(x - h["I_ext_star"], 0)) * 0.005,
                ),
            }
            for x in grid
        ]
        return {"up": rows, "down": copy.deepcopy(rows)}

    def cycle(h):
        period = 1000 / h["freq_star_Hz"]
        return {
            "I_ext": h["I_ext_star"] + 0.4,
            **series(np.linspace(0, 700, 2001), 4),
            "waveform": series(np.linspace(700 - 3 * period, 700, 1500), 4),
            "phase": series(np.linspace(700 - 4 * period, 700, 2000), 4),
        }

    ref = continuation(np.linspace(*cfg["drive_grid"]))
    ref.update(ramp=ramp(ref["hopf"]), cycle=cycle(ref["hopf"]))
    reductions = {}
    for key in (
        "three_d_qss",
        "keep_E_I (Wilson-Cowan)",
        "keep_ge_gi (QSS rates)",
        "keep_E_gi (fast/slow)",
        "keep_E_ge",
        "keep_I_gi",
        "keep_I_ge",
    ):
        rows = copy.deepcopy(ref["sweep"])
        dim = 3 if key == "three_d_qss" else 2
        for row in rows:
            row["fp"] = row["fp"][:dim]
            row["eigs"] = row["eigs"][:dim]
            if dim == 2:
                row["eigs"] = [[-1, 0.1], [-1, -0.1]]
        reductions[key] = rows
    sensitivities = []
    for sigma in cfg["sigma_grid_mV"]:
        r = continuation(np.linspace(*cfg["sensitivity_grid"]))
        r.update(
            sigma_V_mV=sigma,
            convergence=continuation(np.linspace(*cfg["convergence_grid"])),
            ramp=ramp(r["hopf"]),
            cycle=cycle(r["hopf"]),
        )
        sensitivities.append(r)
    return {
        "schema": "exp033.compute/v1",
        "recipe": cfg,
        "reference": ref,
        "comparison": {
            "I_ext": ref["hopf"]["I_ext_star"] + 1,
            "fp": [0.004, 0.001, 0.008, 0.012],
            "4d": series(np.linspace(0, 300, 601), 4),
            "2d": series(np.linspace(0, 300, 601), 2, 1e-10),
        },
        "ladder": {
            key: {"fp": [0.004] * dim, **series(np.linspace(0, 400, 801), dim)}
            for key, dim in (("4d", 4), ("3d", 3), ("2d", 2))
        },
        "reductions": reductions,
        "frequency": [
            {
                "tau_gaba_ms": tau,
                **copy.deepcopy(continuation(np.linspace(*cfg["drive_grid"]))),
            }
            for tau in cfg["tau_grid_ms"]
        ],
        "sensitivity": sensitivities,
    }


def frequencies():
    return {
        "results": [
            {"tau_gaba_ms": tau, "seed": seed, "f_gamma_hz": 100 / tau + offset}
            for tau in recipe.TAU_GRID_MS
            for seed, offset in ((42, 0), (43, 1), (44, 10))
        ]
    }


@pytest.fixture
def lab(tmp_path, monkeypatch):
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(
        stages,
        "memberships",
        lambda _: {s: "demo" for s in ("exp022", "exp033", "exp041")},
    )
    monkeypatch.setattr(
        stages, "_capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    with stages.stage_run(tmp_path, "exp022", "compute") as bank:
        (bank.export / "fixture.txt").write_text("synthetic bank")
    b = inputs.source(tmp_path, bank.run_id, "compute", experiment="exp022")
    with stages.stage_run(
        tmp_path, "exp041", "compute", inputs={"bank": b}
    ) as upstream:
        (upstream.export / "fixture.txt").write_text("synthetic traces")
    u = inputs.source(tmp_path, upstream.run_id, "compute", experiment="exp041")
    with stages.stage_run(
        tmp_path, "exp041", "analyse", inputs={"compute": u, "bank": b}
    ) as freq:
        write_json_atomic(freq.export / "results.json", frequencies())
    raw = synthetic()
    calls = []

    def simulate():
        calls.append("compute")
        return copy.deepcopy(raw)

    monkeypatch.setattr(compute, "simulate", simulate)
    return tmp_path, freq.run_id, bank.run_id, calls


def resign(directory):
    r = load_json(directory / "run.json")
    r["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", r)


def fail(*a, **k):
    pytest.fail("implicit upstream work")


def test_independent_stages_and_lossless_arrays(lab, monkeypatch):
    root, frequency, _, calls = lab
    identity = compute.compute()
    source = inputs.source(root, identity, "compute")
    assert source.record["inputs"] == {}
    assert source.record["origin"] == "local"
    assert identity == "exp033-r001-compute"
    original = synthetic()
    saved = evidence.read(source.export)
    np.testing.assert_array_equal(
        saved["reference"]["cycle"]["waveform"]["Y"],
        original["reference"]["cycle"]["waveform"]["Y"],
    )
    assert not list(source.export.glob("*.svg"))
    monkeypatch.setattr(compute, "simulate", fail)
    for name in ("solve_ivp", "fsolve", "brentq", "sweep", "find_hopf"):
        monkeypatch.setattr(numerics, name, fail)
    analysis_id = analyse.analyse(identity, frequency)
    analysis_source = inputs.source(root, analysis_id, "analyse")
    assert set(analysis_source.record["inputs"]) == {"compute", "frequencies"}
    expected = load_json(analysis_source.export / "results.json")
    assert (
        expected["results"]["frequency_vs_tau_gaba"]["spiking_exp041"]["6.0"]
        == 100 / 6 + 1
    )
    monkeypatch.setattr(measurements, "analyse", fail)
    monkeypatch.setattr(analyse, "analyse", fail)
    presented = inputs.source(root, present.present(analysis_id), "present")
    assert set(presented.record["inputs"]) == {"analysis"}
    assert all((presented.presentation / name).is_file() for name in recipe.FIGURES)
    displayed = load_json(presented.presentation / "numbers.json")
    assert displayed == appearance.article_numbers(expected)
    assert displayed["results"] == expected["results"]
    assert [c["passed"] for c in displayed["success_criteria"]] == [
        c["passed"] for c in expected["success_criteria"]
    ]
    assert all(p.is_file() for p in presented.presentation.iterdir())
    assert calls == ["compute"]
    assert not (root / ".artifacts").exists()
    assert not list((root / ".pingstore/runs").glob(".*.tmp"))


@pytest.mark.parametrize("damage", ["payload", "v2", "root_file", "symlink"])
def test_source_corruption_rejected_before_reservation(lab, damage):
    root, frequency, bank, _ = lab
    identity = compute.compute()
    directory = root / ".pingstore/runs" / bank
    if damage == "payload":
        (directory / "export/fixture.txt").write_text("changed")
    elif damage in ("manifest", "v2"):
        r = load_json(directory / "run.json")
        if damage == "v2":
            r["schema"] = "pingstore.run/v2"
        else:
            r["execution"]["changed"] = True
        write_json_atomic(directory / "run.json", r)
    elif damage == "root_file":
        (directory / "unexpected").write_text("x")
    else:
        (directory / "export/link").symlink_to("fixture.txt")
    before = set((root / ".pingstore/runs").iterdir())
    with pytest.raises(PingstoreError):
        analyse.analyse(identity, frequency)
    assert set((root / ".pingstore/runs").iterdir()) == before


@pytest.mark.parametrize("target", ["ancestor", "compute"])
def test_metadata_amendment_during_analysis_is_allowed(lab, monkeypatch, target):
    root, frequency, bank, _ = lab
    identity = compute.compute()
    original = measurements.analyse

    def mutate(*args):
        output = original(*args)
        path = (
            root
            / ".pingstore/runs"
            / (bank if target == "ancestor" else identity)
            / "run.json"
        )
        record = load_json(path)
        record["execution"]["changed"] = True
        write_json_atomic(path, record)
        return output

    monkeypatch.setattr(measurements, "analyse", mutate)
    analysis_id = analyse.analyse(identity, frequency)
    assert (root / ".pingstore/runs" / analysis_id).is_dir()


def test_compute_failure_and_reserved_identity_reuse(lab, monkeypatch):
    root, _, _, _ = lab
    identity = stages.reserve_stage(root / ".pingstore", "exp033", "compute")

    def broken():
        raise RuntimeError("solver failure")

    monkeypatch.setattr(compute, "simulate", broken)
    with pytest.raises(RuntimeError, match="solver failure"):
        compute.compute(run_id=identity)
    assert not (root / ".pingstore/runs" / identity).exists()
    with pytest.raises(PingstoreError, match="interrupted execution"):
        compute.compute(run_id=identity)


def test_presentation_failure_never_completes(lab, monkeypatch):
    root, frequency, _, _ = lab
    aid = analyse.analyse(compute.compute(), frequency)

    def broken(*a, **k):
        raise RuntimeError("renderer failure")

    monkeypatch.setattr(plots, "fig_bifurcation_compound", broken)
    with pytest.raises(RuntimeError, match="renderer failure"):
        present.present(aid)
    assert not list((root / ".pingstore/runs").glob("exp033-*-present"))


@pytest.mark.parametrize(
    "damage", ["ramp", "cycle", "reductions", "tau", "sigma", "grid", "recipe"]
)
def test_incomplete_measurements_fail(lab, damage):
    root, frequency, _, _ = lab
    identity = compute.compute()
    source = inputs.source(root, identity, "compute")
    raw = evidence.read(source.export)
    if damage == "ramp":
        raw["reference"]["ramp"]["up"].pop()
    elif damage == "cycle":
        raw["reference"]["cycle"]["waveform"]["Y"] = np.zeros((4, 2))
    elif damage == "reductions":
        raw["reductions"].pop("keep_E_ge")
    elif damage == "tau":
        raw["frequency"].pop()
    elif damage == "sigma":
        raw["sensitivity"].pop()
    elif damage == "grid":
        raw["reference"]["sweep"][0]["I_ext"] = -1
    else:
        raw["recipe"]["sigma_V_mV"] = 99
    evidence.write(source.export, raw)
    resign(source.directory)
    with pytest.raises(PingstoreError):
        analyse.analyse(identity, frequency)


def test_wrong_stage_or_experiment_and_missing_inputs(lab):
    root, frequency, bank, _ = lab
    identity = compute.compute()
    for source in (identity, bank):
        with pytest.raises(PingstoreError):
            analyse.analyse(identity, source)
    with pytest.raises(PingstoreError):
        present.present(identity)
    with pytest.raises(PingstoreError):
        analyse.analyse(bank, frequency)


def test_frequency_coverage_and_median():
    d = frequencies()
    assert measurements.spiking_medians(d)[6.0] == 100 / 6 + 1
    d["results"].pop()
    with pytest.raises(PingstoreError):
        measurements.spiking_medians(d)


def test_failed_integrator_does_not_return_partial_evidence(monkeypatch):
    monkeypatch.setattr(
        compute,
        "solve_ivp",
        lambda *a, **k: SimpleNamespace(success=False, message="failed"),
    )
    with pytest.raises(PingstoreError, match="integration failed"):
        compute.integrate(numerics.rhs_4d, np.ones(4), 1.0, 1.0)


def test_bounded_ramp_and_cycle_measurement_equivalence(monkeypatch):
    h = synthetic()["reference"]["hopf"]

    def solve(rhs, span, initial, args=(), **kwargs):
        t = np.linspace(0, span[1], 101)
        amp = np.sqrt(max(args[0] - h["I_ext_star"], 0)) * 0.005

        def values(tt):
            return np.array(
                [initial[k] + amp * np.sin(tt / 5 - k) for k in range(len(initial))]
            )

        return SimpleNamespace(
            t=t, y=values(t), sol=values, success=True, message="synthetic"
        )

    monkeypatch.setattr(compute, "solve_ivp", solve)
    monkeypatch.setattr(numerics, "solve_ivp", solve)
    monkeypatch.setattr(
        numerics, "fixed_point", lambda *a, **k: np.array([0.004, 0.001, 0.008, 0.012])
    )
    for sigma in recipe.SIGMA_V_GRID_MV:
        old = numerics.hysteresis_sweep(h["I_ext_star"], sigma=sigma)
        new = measurements.hysteresis(compute.ramp(h, sigma), h)
        assert old == new
        old_cycle = numerics.limit_cycle_metrics(h, sigma=sigma)
        new_cycle = measurements.cycle(compute.cycle(h, sigma))
        assert new_cycle == {
            k: old_cycle[k] for k in ("I_ext", "e_leads_i_ms", "e_peak_to_peak")
        }
    old_comp = numerics.compute_2d_vs_4d(h)
    comp = compute.comparison(h)
    for key in ("4d", "2d"):
        t, y = comp[key]["t_ms"], comp[key]["Y"]
        d = y[0] - comp["fp"][0]
        assert old_comp["pp_" + key] == float(d[t > 150].max() - d[t > 150].min())


def test_imports_do_not_create_storage_or_import_renderers(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(repo),
        "PINGLAB_RUN_STATE_DIR": str(tmp_path / "state"),
        "PINGLAB_RUN_DERIVED_DIR": str(tmp_path / "derived"),
    }
    code = "from experiments import exp033; import sys; assert 'matplotlib.pyplot' not in sys.modules; assert not hasattr(exp033, 'RUN_PATHS')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_collection_dispatch_pins_frequencies_and_never_publishes(lab, monkeypatch):
    root, frequency, _, calls = lab
    from experiments.collections.gamma_gated_sparsity import execution
    from experiments.collections.gamma_gated_sparsity.plan import build_plan

    plan = build_plan(root / "campaign", "fixture")
    row = next(
        r for s in plan["stages"] for r in s["experiments"] if r["slug"] == "exp033"
    )
    assert row["command"] == [] and row["execution"]["mode"] == "exp033-staged"
    assert execution._stage_adapter("exp033") is collection
    monkeypatch.setattr(
        collection,
        "campaign_frequencies",
        lambda *a: inputs.source(root, frequency, "analyse", experiment="exp041"),
    )
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        stage = command[2].rsplit(".", 1)[-1]
        run_id = command[command.index("--run-id") + 1]
        if stage == "compute":
            identity = compute.compute(run_id=run_id)
        elif stage == "analyse":
            identity = analyse.analyse(
                command[command.index("--source") + 1],
                command[command.index("--frequency-source") + 1],
                run_id=run_id,
            )
        else:
            identity = present.present(
                command[command.index("--source") + 1], run_id=run_id
            )
        return SimpleNamespace(stdout=identity + "\n")

    monkeypatch.setattr(collection.subprocess, "run", execute)
    refs = collection.execute(root, plan, row)
    assert set(refs) == {"compute", "analyse", "present", "frequencies"}
    assert [c[2].rsplit(".", 1)[-1] for c in commands] == [
        "compute",
        "analyse",
        "present",
    ]
    assert (
        collection.completed(root, plan, row).record["run_id"]
        == refs["present"]["run_id"]
    )
    collection.execute(root, plan, row)
    assert len(commands) == 3 and calls == ["compute"]
    assert not (root / ".artifacts").exists()
    row["execution"]["mode"] = "monolithic"
    with pytest.raises(PingstoreError, match="legacy exp033"):
        collection.execute(root, plan, row)


def test_collection_waits_for_exp041_without_launching_it(lab, monkeypatch):
    root, _, _, _ = lab
    from experiments.collections.gamma_gated_sparsity.plan import build_plan

    plan = build_plan(root / "campaign", "fixture")
    row = next(
        r for s in plan["stages"] for r in s["experiments"] if r["slug"] == "exp033"
    )
    monkeypatch.setattr(collection.subprocess, "run", fail)
    with pytest.raises(PingstoreError, match="completed exp041 analysis"):
        collection.execute(root, plan, row)


def test_array_nan_and_pickle_are_rejected(tmp_path):
    with pytest.raises(PingstoreError):
        evidence.write(tmp_path, {"x": np.array([np.nan])})
    np.savez(tmp_path / "arrays.npz", a0000=np.array([object()], dtype=object))
    write_json_atomic(tmp_path / "evidence.json", {"x": {"array": "a0000"}})
    with pytest.raises(ValueError, match="Object arrays"):
        evidence.read(tmp_path)


def test_compute_orchestration_preserves_every_grid(monkeypatch):
    expected = synthetic()
    calls = []

    def continuation(grid, *, sigma=recipe.SIGMA_V_MV, tau=recipe.TAU_GABA_MS):
        calls.append((grid.tolist(), sigma, tau))
        return copy.deepcopy(expected["reference"])

    monkeypatch.setattr(compute, "continuation", continuation)
    monkeypatch.setattr(compute, "ramp", lambda *a: expected["reference"]["ramp"])
    monkeypatch.setattr(compute, "cycle", lambda *a: expected["reference"]["cycle"])
    monkeypatch.setattr(compute, "comparison", lambda *a: expected["comparison"])
    monkeypatch.setattr(compute, "ladder", lambda: expected["ladder"])
    reductions = []

    def reduction(rhs, fp, grid):
        reductions.append((rhs.__name__, fp.__name__, grid.tolist()))
        return []

    monkeypatch.setattr(numerics, "reduction_sweep", reduction)
    result = compute.simulate()
    assert len(calls) == 15
    grid = np.linspace(0, 4, 401).tolist()
    assert calls[:7] == [(grid, 4.0, tau) for tau in (6.0, *recipe.TAU_GRID_MS)]
    assert calls[7:] == [
        (np.linspace(0, 1.2, count).tolist(), sigma, 6.0)
        for sigma in recipe.SIGMA_V_GRID_MV
        for count in (121, 241)
    ]
    assert len(reductions) == 7
    assert all(row[2] == grid for row in reductions)
    assert len(result["sensitivity"]) == 4


def test_failed_campaign_reservation_is_not_replaced(lab, monkeypatch):
    root, _, _, _ = lab
    row = {
        "execution": {"mode": "exp033-staged"},
        "paths": {"state": str(root / "campaign")},
        "required_outputs": [str(root / "campaign/stage-refs.json")],
    }
    identities = collection.reserve(root, row, origin="slurm-wilkes")

    def broken():
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(compute, "simulate", broken)
    with pytest.raises(RuntimeError):
        compute.compute(run_id=identities["compute"])
    with pytest.raises(PingstoreError, match="explicit recovery"):
        collection.reserve(root, row)
    assert load_json(root / "campaign/stage-reservations.json") == identities


def test_hpc_without_prior_reservation_fails_before_work(lab, monkeypatch):
    root, _, _, calls = lab
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    with pytest.raises(PingstoreError, match="before submission"):
        compute.compute()
    row = {
        "execution": {"mode": "exp033-staged"},
        "paths": {"state": str(root / "campaign")},
        "required_outputs": [str(root / "campaign/stage-refs.json")],
    }
    with pytest.raises(PingstoreError, match="before submission"):
        collection.reserve(root, row)
    assert calls == []
    assert not list((root / ".pingstore/runs").glob("*exp033*"))


def test_historical_regression_tolerance_preserves_evidence():
    numbers, _ = measurements.analyse(synthetic(), frequencies())
    r = numbers["results"]
    original = evidence.amplitude_summary(r["criticality"], r["hopf"]["I_ext_star"])
    recorded = copy.deepcopy(original)
    recorded["A2_slope"] = float(np.nextafter(original["A2_slope"], np.inf))
    before = copy.deepcopy(recorded)
    assert evidence.verify_amplitudes(recorded, r["hopf"]["I_ext_star"]) == original
    assert recorded == before
    recorded["A2_slope"] *= 1.01
    with pytest.raises(PingstoreError, match="regression"):
        evidence.verify_amplitudes(recorded, r["hopf"]["I_ext_star"])


def test_frequency_axis_does_not_magnify_roundoff(tmp_path, monkeypatch):
    data = synthetic()
    numbers, _ = measurements.analyse(data, frequencies())
    rows = numbers["results"]["sigma_sensitivity"]["rows"]
    for i, row in enumerate(rows):
        row["hopf"]["freq_star_Hz"] = 27.566 + i * 1e-9
        row["limit_cycle"]["e_peak_to_peak"] = (8.3 - 1.5 * i) / 1000
    figures = []
    original = plots.plt.subplots

    def capture(*a, **k):
        fig, axes = original(*a, **k)
        figures.append(fig)
        return fig, axes

    monkeypatch.setattr(plots.plt, "subplots", capture)
    plots.plot_sigma_sensitivity(
        numbers["results"]["sigma_sensitivity"],
        tmp_path / "sigma.svg",
        "exp033-fixture",
    )
    axis = figures[0].axes[1]
    assert axis.get_ylim() == (0, 40)
    assert not axis.yaxis.get_major_formatter().get_useOffset()
    np.testing.assert_array_equal(
        axis.lines[0].get_ydata(), [r["hopf"]["freq_star_Hz"] for r in rows]
    )
    assert "exp033-fixture" not in (tmp_path / "sigma.svg").read_text()
    amplitude = figures[0].axes[3]
    line = amplitude.lines[0]
    path = line.get_transform().transform_path(line.get_path())
    text_box = amplitude.texts[0].get_window_extent(figures[0].canvas.get_renderer())
    assert not path.intersects_bbox(text_box, filled=False)


def test_historical_svg_changes_only_stamp_and_legend(tmp_path):
    import xml.etree.ElementTree as ET

    source, destination = tmp_path / "source.svg", tmp_path / "output.svg"
    source.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 573.113688 325.44" height="325.44pt"><g id="axes_1"><path id="trace" d="M 1 2 L 3 4"/><g id="legend_1"><text x="300" y="20">4D</text></g></g><g id="stamp"><!-- exp033-numerics --><text>old</text></g></svg>'
    )
    before = source.read_bytes()
    appearance.historical_svg(source, destination, move_legend=True)
    root = ET.parse(destination).getroot()
    assert root.get("viewBox") == "0 -60 573.113688 385.44"
    assert root.get("height") == "385.44pt"
    assert root.find('.//*[@id="trace"]').attrib == {"id": "trace", "d": "M 1 2 L 3 4"}
    assert root.find('.//*[@id="legend_1"]').get("transform") == "translate(0 -60)"
    assert root.find('.//*[@id="stamp"]') is None
    assert source.read_bytes() == before
    with pytest.raises(PingstoreError, match="exactly one"):
        appearance.historical_svg(destination, tmp_path / "twice.svg")


def test_article_selected_inputs_equations_and_absent_data(lab):
    import re
    import shutil
    import xml.etree.ElementTree as ET

    from demolab_cli import _paths

    root, frequency, _, _ = lab
    aid = analyse.analyse(compute.compute(), frequency)
    output = inputs.source(root, present.present(aid), "present")
    repo = Path(__file__).resolve().parents[2]
    (root / "writings").mkdir()
    for name in (
        "exp033.typ", "templates/dataset.typ", "templates/abstract.typ",
        "templates/methods.typ", "templates/article-layout.typ",
        "templates/result-card.typ", "templates/references.typ",
        "templates/contents.typ", "templates/equations.typ", "templates/status.typ",
    ):
        target = root / "writings" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / "writings" / name, target)
    (root / ".demolab").mkdir()
    shutil.copyfile(_paths.TYP / "lib.typ", root / ".demolab/lib.typ")
    write_json_atomic(
        root / "preview.json",
        {"exp033": {"exp033": "/" + str(output.export.relative_to(root))}},
    )
    doc = root / "doc.typ"
    doc.write_text(
        '#set page(paper: "a4", margin: 18mm)\n#set text(size: 10pt)\n#import "writings/exp033.typ": body\n#body'
    )
    base = [_paths.find_typst(repo), "compile", "--root", str(root)]
    selected = ["--input", "demolab-preview-file=/preview.json"]
    for fmt, extra in [
        ("pdf", []),
        ("html", ["--features", "html", "--format", "html"]),
    ]:
        result = subprocess.run(
            [*base, *selected, *extra, str(doc), str(root / ("article." + fmt))],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert "does not exist" not in result.stderr
    html = (root / "article.html").read_text()
    assert len(re.findall(r"<img\b", html)) == 6
    assert len(re.findall(r"<figcaption\b", html)) == 6
    assert len(re.findall(r"<math\b", html)) > 100
    gains = []
    for expression in re.findall(r"<math\b.*?</math>", html, re.S):
        for node in ET.fromstring(expression).iter("msub"):
            if "".join(node[0].itertext()) == "Φ":
                gains.append(node)
    assert gains
    # A missing space after Phi_E puts the entire gain argument in its subscript.
    assert all(g[1].tag == "mi" and g[1].text in ("𝐸", "𝐼") for g in gains)
    assert html.index("Results") < html.index("Methods") < html.index("Appendix:")
    assert "Figure 8" not in (root / "writings/exp033.typ").read_text()
    assert "exp033-fixture" not in html
    assert "[(!) The present model supplies that candidate mechanism" in html
    assert 'href="https://www.ma.ic.ac.uk/~dturaev/kuznetsov.pdf"' in html
    result = subprocess.run(
        [
            *base,
            "--features",
            "html",
            "--format",
            "html",
            str(doc),
            str(root / "pending.html"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Oscillatory onset" not in (root / "pending.html").read_text()
    (output.export / "numbers.json").write_text("broken")
    result = subprocess.run(
        [*base, *selected, str(doc), str(root / "broken.pdf")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0


def test_refined_hopf_lies_inside_coarse_bracket() -> None:
    grid = np.linspace(0.0, 1.2, 25)
    hopf = exp033.find_hopf(exp033.sweep(grid, sigma=4.0), sigma=4.0)
    assert hopf is not None
    lo, hi = hopf["coarse_bracket_nA"]
    assert lo <= hopf["I_ext_star"] <= hi
    assert abs(hopf["leading_eigenvalue"][0]) < 1e-8


def test_hysteresis_propagates_sigma(monkeypatch) -> None:
    observed: list[float] = []

    def fake_fixed_point(_drive, _tau=exp033.TAU_GABA_MS, x0=None, sigma=4.0):
        observed.append(sigma)
        return np.ones(4)

    def fake_settle(drive, state, _tau=exp033.TAU_GABA_MS, sigma=4.0, **_kwargs):
        observed.append(sigma)
        return max(0.0, drive - 0.5), state

    monkeypatch.setattr(exp033, "fixed_point", fake_fixed_point)
    monkeypatch.setattr(exp033, "settle", fake_settle)
    exp033.hysteresis_sweep(0.5, sigma=5.5, span=(-0.05, 0.05), n=3)
    assert observed and set(observed) == {5.5}


def test_tau_gaba_sweep_propagates_sigma(monkeypatch) -> None:
    observed: list[tuple[float, float]] = []

    def fake_sweep(_grid, tau_gaba=0.0, sigma=0.0):
        observed.append((tau_gaba, sigma))
        return []

    monkeypatch.setattr(exp033, "sweep", fake_sweep)
    monkeypatch.setattr(exp033, "find_hopf", lambda *_args, **_kwargs: None)
    exp033.frequency_vs_tau_gaba([4.5, 9.0], np.array([0.0, 1.0]), sigma=6.0)
    assert observed == [(4.5, 6.0), (9.0, 6.0)]


def test_exp054_explicitly_selects_reference_sigma() -> None:
    from experiments.exp054.recipe import configuration

    assert configuration()["mean_field"]["sigma_V_mV"] == exp033.SIGMA_V_MV


def test_publication_text_does_not_claim_fully_fitted_scale() -> None:
    corpus = "\n".join(
        (exp033.REPO / path).read_text()
        for path in (
            "writings/exp033.typ",
            "writings/exp054.typ",
            "writings/exp110.typ",
        )
    ).lower()
    assert "no fitted scale" not in corpus
    assert "absolute scale is fixed by the biophysics" not in corpus
