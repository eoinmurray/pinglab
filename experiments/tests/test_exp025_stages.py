"""Synthetic v3 stage fixtures; no production simulations or archive imports."""

from pathlib import Path

import numpy as np
import pytest
from experiments.exp025 import (
    analyse,
    compute,
    evidence,
    inputs,
    measurements,
    present,
    recipe,
)
from experiments.tests.test_exp044_provenance import _common_config
from pingstore import stages
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(
        stages, "memberships", lambda _: {"exp022": "demo", "exp025": "demo"}
    )
    monkeypatch.setattr(
        stages, "_capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    monkeypatch.setenv("PINGLAB_SMOKE", "1")
    with stages.stage_run(
        tmp_path, "exp022", "compute", export_root="export/cells"
    ) as run:
        for cell in recipe.bank_cells():
            name = cell["cell_name"]
            directory = run.export / "cells" / name
            directory.mkdir(parents=True)
            cfg = {
                **_common_config(),
                "dt": 0.1,
                "seed": cell["seed"],
                "n_hidden": 4,
                "n_inh": 2,
                "hidden_sizes": [4],
                "ei_strength": float(cell["model"] == "ping"),
                "v_grad_dampen": 1000.0 if cell["model"] == "ping" else 1.0,
                "w_in": [cell["w_in"], cell["w_in"] * 0.1],
                "fr_reg_upper_strength": 0.0
                if cell["rate_target_hz"] is None
                else 0.041,
                "fr_reg_upper_target_hz": cell["rate_target_hz"] or 0.0,
            }
            write_json_atomic(directory / "config.json", cfg)
            checkpoints = {}
            for role, filename in (
                ("final_epoch", "weights_final.pth"),
                ("best_validation", "weights.pth"),
            ):
                target = directory / filename
                target.write_bytes((name + role).encode())
                checkpoints[role] = {
                    "filename": filename,
                    "epoch": 50,
                    "sha256": file_sha256(target),
                }
            write_json_atomic(
                directory / "metrics.json",
                {
                    "config": cfg,
                    "training_cell_name": name,
                    "best_epoch": 50,
                    "best_acc": 85.0,
                    "checkpoints": checkpoints,
                    "epochs": [
                        {
                            "ep": ep,
                            "acc": 80 + ep / 10,
                            "test_rate_e": ep / 2,
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

        calls.append(args)

        def arg(key):
            return args[args.index(key) + 1]

        cfg = load_json(Path(arg("--load-config")))
        out = Path(arg("--out-dir"))
        out.mkdir(parents=True)
        assert arg("--device") == "auto"
        assert Path(arg("--load-weights")).name == "weights_final.pth"
        assert kwargs == {"no_sync": True}
        if "--digit" in args:
            assert "--max-samples" not in args
            e, i = np.zeros((4000, 4), bool), np.zeros((4000, 2), bool)
            e[::200] = True
            i[::200] = True
            save(out / "snapshot.npz", dt=0.1, spk_e=e, spk_i=i, unused=np.zeros(20))
        else:
            n = int(arg("--max-samples"))
            write_json_atomic(
                out / "metrics.json",
                {
                    "config": {
                        **cfg,
                        "evaluation_partition": "official_mnist_test",
                        "evaluation_samples": n,
                    },
                    "best_acc": 80.0,
                    "n_total": n,
                    "n_correct": n * 8 // 10,
                    "rates_hz": {"hid": 2.0, "inh": 0.1},
                    "ce_loss": 2.3,
                },
            )
            if "per_cell_rates" in args:
                save(
                    out / "per_cell_rates.npz",
                    rate_e_per_sample=np.linspace(0.5, 2.0, n, dtype=np.float32),
                    unused=np.zeros(20),
                )
            if "pop_traces" in args:
                t = np.arange(2000) * 0.1 / 1000
                save(
                    out / "pop_traces.npz",
                    dt=0.1,
                    pop_e=np.tile(0.1 + 0.05 * np.sin(2 * np.pi * 50 * t), (n, 1)),
                )
                tr, ts = np.meshgrid(
                    np.arange(n), np.arange(100, 2000, 200), indexing="ij"
                )
                data = {"dt": 0.1, "T": 2000, "n_e": 4, "n_i": 2, "n_trials": n}
                for p in ("e", "i"):
                    data.update(
                        {
                            p + "_trial": tr.ravel().astype(np.int32),
                            p + "_t": ts.ravel().astype(np.int32),
                            p + "_cell": np.zeros(tr.size, np.int32),
                        }
                    )
                save(out / "rasters.npz", **data)
        simulation_config = {
            **cfg,
            "infer": True,
            "input": "dataset",
            "tau_gaba": cfg["tau_gaba_ms"],
            "t_ms": 400.0 if "--digit" in args else cfg["t_ms"],
            "max_samples": None if "--digit" in args else int(arg("--max-samples")),
            "digit": 0,
            "sample": 0,
            "scale_w_in": float(arg("--scale-w-in")) if "--scale-w-in" in args else 1.0,
            "scale_w_ei": 1.0,
            "scale_w_ie": 1.0,
            "intervention": [],
            "scale_projection": [],
            "load_weights": arg("--load-weights"),
            "load_config": arg("--load-config"),
        }
        if "--digit" not in args:
            metrics = load_json(out / "metrics.json")
            metrics["config"].pop("seed")
            metrics["config"].pop("tau_gaba_ms")
            metrics["config"]["load_weights"] = arg("--load-weights")
            write_json_atomic(out / "metrics.json", metrics)
        write_json_atomic(out / "config.json", simulation_config)
        (out / "run.sh").write_text("synthetic command\n")

    monkeypatch.setattr(compute, "run_cli", simulate)
    return tmp_path, run.run_id, calls


def test_independent_stages_preserve_bank_and_do_not_publish(lab, monkeypatch):
    root, bank_id, calls = lab
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    identity = compute.compute(bank_id)
    assert len(calls) == 56
    run = inputs.source(root, identity, "compute")
    assert set(run.record["inputs"]) == {"bank"}
    original = load_json(
        run.directory
        / "provenance/simulations/frontier/coba__off__seed42/metrics.original.json"
    )
    assert "seed" not in original["config"] and "tau_gaba_ms" not in original["config"]
    corrected = load_json(run.export / "frontier/coba__off__seed42/metrics.json")
    assert (
        corrected["config"]["seed"] == 42 and corrected["config"]["tau_gaba_ms"] == 6.0
    )
    with np.load(run.export / "snapshot/coba/snapshot.npz") as raw:
        assert set(raw.files) == {"dt", "spk_e", "spk_i"}
    monkeypatch.setattr(
        compute, "run_cli", lambda *a, **k: pytest.fail("implicit simulation")
    )
    measured_id = analyse.analyse(identity)
    measured = inputs.source(root, measured_id, "analyse")
    data = load_json(measured.export / "results.json")
    assert len(data["results"]) == 36
    assert len(data["frontier_statistics"]) == 12
    assert len(data["rate_target_p_fgamma"]) == 12
    assert len(data["w_in_scale_sweep"]) == 6
    assert data["rate_target_p_fgamma"][0]["p"] is None
    assert data["rate_target_p_fgamma"][-1]["p"] == 0.25
    for name in (
        "measure_p_fgamma",
        "scaled_metrics",
        "aggregate_frontier",
        "aggregate_low_w_in_seed_rows",
    ):
        monkeypatch.setattr(
            measurements,
            name,
            lambda *a, **k: pytest.fail("presentation measured data"),
        )
    shown = inputs.source(root, present.present(measured_id), "present")
    numbers = load_json(shown.export / "numbers.json")
    assert all(numbers[k] == v for k, v in data.items())
    assert all((shown.export / name).is_file() for name in recipe.FIGURES)
    assert not (root / ".artifacts").exists()
    bank.check_unchanged()


@pytest.mark.parametrize(
    "field,value",
    [
        ("v_grad_dampen", 1000.0),
        ("dt", 0.2),
        ("seed", 99),
        ("fr_reg_upper_strength", 0.04),
    ],
)
def test_wrong_training_recipe_rejected(lab, field, value):
    root, bank_id, calls = lab
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    path = bank.export / recipe.cell_name("coba", None, 42) / "config.json"
    cfg = load_json(path)
    cfg[field] = value
    write_json_atomic(path, cfg)
    with pytest.raises(PingstoreError):
        evidence.training_contract(bank.export)
    assert not calls


def test_raw_payload_corruption_blocks_analysis(lab):
    root, bank_id, _ = lab
    identity = compute.compute(bank_id)
    run = inputs.source(root, identity, "compute")
    (run.export / "snapshot/ping/snapshot.npz").write_bytes(b"corrupt")
    with pytest.raises(PingstoreError, match="checksum"):
        analyse.analyse(identity)


def test_ancestor_manifest_drift_blocks_analysis(lab):
    root, bank_id, _ = lab
    identity = compute.compute(bank_id)
    path = root / ".pingstore/runs" / bank_id / "run.json"
    record = load_json(path)
    record["execution"]["note"] = "changed"
    write_json_atomic(path, record)
    with pytest.raises(PingstoreError, match="checksum"):
        analyse.analyse(identity)


def test_inference_failure_stays_hidden(lab, monkeypatch):
    root, bank_id, _ = lab

    def fail(*a, **k):
        raise RuntimeError("simulation failed")

    monkeypatch.setattr(compute, "run_cli", fail)
    with pytest.raises(RuntimeError, match="simulation failed"):
        compute.compute(bank_id)
    assert not list((root / ".pingstore/runs").glob("exp025-*"))
    assert list((root / ".pingstore/runs").glob(".exp025-*.tmp"))


def test_penalty_retains_float32_samplewise_definition(tmp_path):
    rates = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32)
    np.savez(tmp_path / "per_cell_rates.npz", rate_e_per_sample=rates)
    write_json_atomic(
        tmp_path / "metrics.json",
        {"best_acc": 80.0, "ce_loss": 2.0, "rates_hz": {"hid": 1.0, "inh": 0.5}},
    )
    value = measurements.scaled_metrics(tmp_path, 1.0, 0.041)
    assert value[2] == float(0.041 * (np.maximum(rates - 1.0, 0.0) ** 2).mean())


def test_collection_dispatch_and_resume_use_pinned_stages(lab, monkeypatch):
    from types import SimpleNamespace

    from experiments.collections.gamma_gated_sparsity.plan import build_plan
    from experiments.exp025 import collection

    root, bank_id, _ = lab
    plan = build_plan(root / "campaign", "fixture", smoke=True)
    plan["profile"] = "smoke"
    plan["exp022_manifest"] = str(root / "campaign/exp022-manifest.json")
    write_json_atomic(Path(plan["exp022_manifest"]), {"pingstore_run_id": bank_id})
    # Build-plan rows are grouped by collection phase.
    from experiments.collections.gamma_gated_sparsity.execution import rows_in_order

    row = next(r for r in rows_in_order(plan) if r["slug"] == "exp025")
    seen = []

    def dispatch(command, **kwargs):
        stage = command[2].split(".")[-1]
        seen.append(stage)
        source = command[command.index("--source") + 1]
        identity = command[command.index("--run-id") + 1]
        {
            "compute": compute.compute,
            "analyse": analyse.analyse,
            "present": present.present,
        }[stage](source, run_id=identity)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(collection.subprocess, "run", dispatch)
    refs = collection.execute(root, plan, row)
    assert seen == ["compute", "analyse", "present"]
    assert refs == collection.execute(root, plan, row)
    assert len(seen) == 3
    plan["profile"] = "production"
    with pytest.raises(PingstoreError, match="profile"):
        collection.execute(root, plan, row)


@pytest.mark.parametrize("fault", ["duplicate", "shape", "nan", "samples"])
def test_recording_semantics_reject_corrupt_but_resigned_payload(lab, fault):
    from pingstore.contracts import payload_digest

    root, bank_id, _ = lab
    identity = compute.compute(bank_id)
    run = inputs.source(root, identity, "compute")
    directory = run.export / "pfg/ping__off__seed42"
    if fault == "samples":
        path = directory / "metrics.json"
        data = load_json(path)
        data["n_total"] = 99
        write_json_atomic(path, data)
    elif fault in ("shape", "nan"):
        path = directory / "pop_traces.npz"
        with np.load(path) as data:
            arrays = {k: data[k].copy() for k in data.files}
        if fault == "shape":
            arrays["pop_e"] = arrays["pop_e"][:-1]
        else:
            arrays["pop_e"][0, 0] = np.nan
        np.savez(path, **arrays)
    else:
        path = directory / "rasters.npz"
        with np.load(path) as data:
            arrays = {k: data[k].copy() for k in data.files}
        for key in ("e_trial", "e_t", "e_cell"):
            arrays[key][1] = arrays[key][0]
        np.savez(path, **arrays)
    record = load_json(run.directory / "run.json")
    record["payload_digest"] = payload_digest(run.directory)
    write_json_atomic(run.directory / "run.json", record)
    with pytest.raises(PingstoreError):
        analyse.analyse(identity)
    assert not list((root / ".pingstore/runs").glob("exp025-*-analyse"))
