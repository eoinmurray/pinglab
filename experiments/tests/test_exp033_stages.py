"""Synthetic staged evidence and bounded numerical checks; no production runs."""

import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.exp033 import (
    analyse,
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
    assert load_json(presented.presentation / "numbers.json") == expected
    assert all(p.is_file() for p in presented.presentation.iterdir())
    assert calls == ["compute"]
    assert not (root / ".artifacts").exists()
    assert not list((root / ".pingstore/runs").glob(".*.tmp"))


@pytest.mark.parametrize(
    "damage", ["payload", "manifest", "v2", "root_file", "symlink"]
)
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
def test_mutation_during_analysis_never_completes(lab, monkeypatch, target):
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
    with pytest.raises(PingstoreError):
        analyse.analyse(identity, frequency)
    assert not list((root / ".pingstore/runs").glob("exp033-*-analyse"))
    assert list((root / ".pingstore/runs").glob(".exp033-*-analyse.tmp"))


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

    monkeypatch.setattr(plots, "plot_hysteresis", broken)
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


@pytest.fixture
def historical_archive(lab, monkeypatch):
    import hashlib
    import json

    from experiments.exp033 import historical, import_gold2
    from pingstore.contracts import file_sha256

    root, frequency, _, _ = lab
    monkeypatch.setattr(import_gold2, "REPO", root)
    archive = root / "archive"
    for name in import_gold2.selected_paths():
        path = archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture evidence")
    old, _ = measurements.analyse(synthetic(), frequencies())
    old["results"]["criticality"] = historical.amplitude_summary(
        old["results"]["criticality"], old["results"]["hopf"]["I_ext_star"]
    )
    producer = {
        "campaign": "fixture",
        "git_commit": "fixture",
        "exp033_job": "33913627",
        "exp054_cache_job": "33913631",
    }
    old["collection_provenance"] = {
        "source_git_commit": "fixture",
        "campaign_id": "fixture",
        "dependencies": ["exp041"],
    }
    write_json_atomic(archive / import_gold2.DERIVED / "numbers.json", old)
    r = old["results"]
    subset = {
        "sweep": synthetic()["reference"]["sweep"],
        "hopf": r["hopf"],
        "criticality": r["criticality"],
        "frequency_vs_tau_gaba": r["frequency_vs_tau_gaba"]["mean_field"],
        "spiking_exp041": {
            str(k): v for k, v in r["frequency_vs_tau_gaba"]["spiking_exp041"].items()
        },
    }
    payload = np.empty(6, dtype=object)
    payload[:] = [{"excluded": "empirical grid"}, *subset.values()]
    np.savez(archive / import_gold2.CACHE, payload=payload)
    raw = json.dumps(
        subset, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    base = archive / import_gold2.BASE
    write_json_atomic(archive / "run.json", {"archive": {"uri": "fixture"}})
    write_json_atomic(
        base / "run.json", {"run_id": "fixture", "source": {"git_commit": "fixture"}}
    )
    write_json_atomic(
        base / "collection-plan.json",
        {
            "stages": [
                {
                    "experiments": [
                        {"slug": s, "command": ["python", s]}
                        for s in ("exp033", "exp054")
                    ]
                }
            ]
        },
    )
    write_json_atomic(
        base / "submissions/collection-submission.json",
        {
            "jobs": [
                {"name": "ggs-" + s, "job_id": j}
                for s, j in (("exp033", "33913627"), ("exp054", "33913631"))
            ]
        },
    )
    for s, j in (("exp033", "33913627"), ("exp054", "33913631")):
        write_json_atomic(
            base / "collection-status" / (s + ".json"),
            {"state": "complete", "experiment": s},
        )
        (base / "logs/collection" / f"ggs-{s}_{j}.out").write_text(
            f"job={j} host=fixture action=run-experiment experiment={s}\ndevice=cpu\n"
        )

    def row(name):
        p = archive / name
        return {"path": name, "size_bytes": p.stat().st_size, "sha256": file_sha256(p)}

    write_json_atomic(
        archive / "inventory.json",
        {
            "files": [
                row(n)
                for n in sorted(import_gold2.selected_paths())
                if n not in ("inventory.json", "run.json", "lineage.json")
            ]
        },
    )
    rows = [row(n) for n in sorted(import_gold2.selected_paths())]
    code = root / "producer.py"
    code.write_text("# fixture scientific producer\n")
    plan = {
        "schema": "exp033.selective-import-plan/v1",
        "archive": "fixture",
        "source_files": rows,
        "source_file_count": len(rows),
        "source_bytes": sum(r["size_bytes"] for r in rows),
        "carry_forward_figures": list(historical.CARRY),
        "producer_code_sha256": file_sha256(code),
        "producer": producer,
        "borrowed_cache": {"selected_json_sha256": hashlib.sha256(raw).hexdigest()},
        "upstream_references": {
            frequency: inputs.source(
                root, frequency, "analyse", experiment="exp041"
            ).reference
        },
        "frequency_comparison": {
            "deltas_hz": {str(t): 0.0 for t in recipe.TAU_GRID_MS}
        },
        "missing_evidence": ["raw trajectories"],
    }
    plan_path = root / "plan.json"
    write_json_atomic(plan_path, plan)
    live = root / "live"
    live.mkdir()
    for name in ("inventory.json", "run.json", "lineage.json"):
        (live / ("live-" + name)).write_bytes((archive / name).read_bytes())
    return import_gold2, (archive, plan_path, file_sha256(plan_path), code, live)


def test_historical_import_and_independent_derived_stages(
    lab, historical_archive, monkeypatch
):
    from experiments.exp033 import historical
    from pingstore.contracts import file_sha256

    root, frequency, _, calls = lab
    importer, args = historical_archive
    monkeypatch.setattr(compute, "simulate", fail)
    monkeypatch.setattr(numerics, "solve_ivp", fail)
    identity = importer.import_subset(*args)
    source = inputs.source(root, identity, "compute")
    assert source.record["origin"] == "local"
    assert source.record["historical_import"]["producer"]["job_id"] == "33913627"
    assert source.record["historical_import"]["cache_producer"]["job_id"] == "33913631"
    assert (
        len(load_json(source.directory / "provenance/file-mapping.json")["files"]) == 19
    )
    analysis_id = analyse.analyse(identity, frequency)
    monkeypatch.setattr(historical, "analyse", fail)
    for name in (
        "plot_limit_cycle",
        "plot_timeseries",
        "plot_phase_planes",
        "plot_reduction_ladder",
    ):
        monkeypatch.setattr(plots, name, fail)
    output = inputs.source(root, present.present(analysis_id), "present")
    for name in historical.CARRY:
        assert file_sha256(output.export / name) == file_sha256(
            source.export / "retained-figures" / name
        )
    assert not calls
    assert not (root / ".artifacts").exists()


@pytest.mark.parametrize("damage", ["plan", "archive", "live", "producer", "copy"])
def test_historical_import_rejects_changed_evidence(
    lab, historical_archive, monkeypatch, damage
):
    importer, args = historical_archive
    root = lab[0]
    archive, plan, _, code, live = args
    target = {
        "plan": plan,
        "archive": archive / importer.CACHE,
        "live": live / "live-run.json",
        "producer": code,
    }
    if damage == "copy":
        original = importer.shutil.copyfile

        def broken(src, dst):
            original(src, dst)
            Path(dst).write_text("corrupted copy")

        monkeypatch.setattr(importer.shutil, "copyfile", broken)
    else:
        target[damage].write_text("changed")
    with pytest.raises(PingstoreError):
        importer.import_subset(*args)
    assert not list((root / ".pingstore/runs").glob("exp033-*"))
    if damage == "copy":
        assert list((root / ".pingstore/runs").glob(".exp033-*.tmp"))


def test_historical_regression_tolerance_preserves_evidence():
    from experiments.exp033 import historical

    numbers, _ = measurements.analyse(synthetic(), frequencies())
    r = numbers["results"]
    original = historical.amplitude_summary(r["criticality"], r["hopf"]["I_ext_star"])
    recorded = copy.deepcopy(original)
    recorded["A2_slope"] = float(np.nextafter(original["A2_slope"], np.inf))
    before = copy.deepcopy(recorded)
    assert historical.verify_amplitudes(recorded, r["hopf"]["I_ext_star"]) == original
    assert recorded == before
    recorded["A2_slope"] *= 1.01
    with pytest.raises(PingstoreError, match="regression"):
        historical.verify_amplitudes(recorded, r["hopf"]["I_ext_star"])
