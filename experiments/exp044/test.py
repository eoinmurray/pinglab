from __future__ import annotations

import json
from pathlib import Path

import pytest
from experiments.exp044 import evidence
from experiments.exp044 import recipe as exp044

"""Temporary synthetic banks and mocked inference; never run scientific experiments."""

import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from experiments.exp044 import (
    analyse,
    collection,
    compute,
    inputs,
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
        stages, "memberships", lambda _: {"exp022": "demo", "exp044": "demo"}
    )
    monkeypatch.setattr(
        stages, "_capture_code", lambda *args: {"git_commit": "fixture", "dirty": False}
    )
    monkeypatch.setattr(recipe, "RASTER_N_E_PLOT", 2)
    monkeypatch.setattr(recipe, "RASTER_N_I_PLOT", 1)
    monkeypatch.setenv("PINGLAB_SMOKE", "1")
    with stages.stage_run(tmp_path, "exp022", "compute") as bank:
        for dt in recipe.DT_SWEEP_MS:
            for seed in recipe.SEEDS:
                name = recipe.cell_name(dt, seed)
                cell = bank.export / name
                cell.mkdir(parents=True)
                cfg = {
                    **_common_config(),
                    "dt": dt,
                    "seed": seed,
                    "n_hidden": 4,
                    "n_inh": 2,
                    "hidden_sizes": [4],
                }
                write_json_atomic(cell / "config.json", cfg)
                checkpoints = {}
                for role, filename in (
                    ("final_epoch", "weights_final.pth"),
                    ("best_validation", "weights.pth"),
                ):
                    path = cell / filename
                    path.write_bytes(role.encode())
                    checkpoints[role] = {
                        "filename": filename,
                        "epoch": recipe.EPOCHS,
                        "sha256": file_sha256(path),
                    }
                write_json_atomic(
                    cell / "metrics.json",
                    {
                        "config": cfg,
                        "training_cell_name": name,
                        "best_epoch": recipe.EPOCHS,
                        "checkpoints": checkpoints,
                        "epochs": [
                            {
                                "ep": ep,
                                "acc": 80.0 + ep / 10,
                                "test_rate_e": ep / 2 + dt,
                            }
                            for ep in range(1, recipe.EPOCHS + 1)
                        ],
                    },
                )
    calls = []

    def simulate(args, **kwargs):
        calls.append(args)

        def arg(key):
            return args[args.index(key) + 1]

        out = Path(arg("--out-dir"))
        out.mkdir(parents=True)
        cfg = load_json(Path(arg("--load-config")))
        assert arg("--device") == "auto"
        assert Path(arg("--load-weights")).name == "weights_final.pth"
        if "--sample-index" in args:
            assert "--max-samples" not in args
            steps = round(cfg["t_ms"] / cfg["dt"])
            e, i = np.zeros((steps, 4), dtype=bool), np.zeros((steps, 2), dtype=bool)
            e[::20, :] = True
            i[::40, :] = True
            np.savez_compressed(out / "recording.npz", spk_e=e, spk_i=i, dt=cfg["dt"])
        else:
            samples = int(arg("--max-samples"))
            write_json_atomic(
                out / "metrics.json",
                {
                    "config": {
                        **cfg,
                        "evaluation_partition": "official_mnist_test",
                        "evaluation_samples": samples,
                    },
                    "best_acc": 90.0,
                    "n_correct": 9 * samples // 10,
                    "n_total": samples,
                    "rates_hz": {
                        "hid": 10.0 + cfg["dt"] + cfg["seed"] - 42,
                        "inh": 20.0,
                    },
                },
            )
        write_json_atomic(out / "config.json", cfg)
        (out / "run.sh").write_text("fixture\n")

    monkeypatch.setattr(compute, "run_cli", simulate)
    return tmp_path, bank.run_id, calls


def resign(directory):
    record = load_json(directory / "run.json")
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)


def test_independent_stages_preserve_science_and_never_publish(lab, monkeypatch):
    root, bank_id, calls = lab
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    before = bank.reference
    compute_id = compute.compute(bank_id)
    assert len(calls) == 20
    assert sum("--sample-index" in args for args in calls) == 5
    output = inputs.source(root, compute_id, "compute")
    assert output.record["inputs"] == {"bank": before}
    assert len(list(output.export.glob("snapshot--*--recording.npz"))) == 5
    assert not list(output.export.rglob("run.sh"))
    monkeypatch.setenv("PINGLAB_SMOKE", "0")
    monkeypatch.setattr(
        compute, "run_cli", lambda *a, **k: pytest.fail("downstream inference")
    )
    analysis_id = analyse.analyse(compute_id)
    analysis_run = inputs.source(root, analysis_id, "analyse")
    results = load_json(analysis_run.export / "results.json")
    assert results["recipe"]["profile"] == "smoke"
    assert results["config"]["evaluation_samples"] == 100
    assert len(results["results"]) == 15
    assert results["aggregate"][0]["e_rate_hz"]["mean"] == pytest.approx(11.05)
    assert results["aggregate"][0]["e_rate_hz"]["sem"] == pytest.approx(1 / np.sqrt(3))
    assert results["measurement"]["gamma_period_estimator"] is None
    assert results["measurement"]["history_partition"] == "validation"
    assert results["rasters"][0]["e_rate_hz"] == 1000
    raw = evidence.snapshot(
        output.file("snapshot", recipe.cell_name(0.05, 42), "recording.npz"),
        0.05,
        results["config"]["training_contract"]["common"],
    )
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(4, 2, replace=False))
    i_idx = np.sort(rng.choice(2, 1, replace=False))
    assert results["rasters"][0]["e_indices"] == e_idx.tolist()
    with np.load(analysis_run.export / "rasters.npz") as data:
        np.testing.assert_array_equal(
            data[recipe.cell_name(0.05, 42) + "__e"], raw["spk_e"][:, e_idx]
        )
        np.testing.assert_array_equal(
            data[recipe.cell_name(0.05, 42) + "__i"], raw["spk_i"][:, i_idx]
        )
    monkeypatch.setattr(
        analyse, "summarize", lambda *a: pytest.fail("presentation remeasurement")
    )
    monkeypatch.setattr(
        evidence, "histories", lambda *a: pytest.fail("presentation reads histories")
    )
    presentation_id = present.present(analysis_id)
    presentation = inputs.source(root, presentation_id, "present")
    assert presentation.record["inputs"] == {"analysis": analysis_run.reference}
    assert all((presentation.export / name).is_file() for name in recipe.FIGURES)
    assert all(p.is_file() for p in presentation.export.iterdir())
    assert (
        load_json(presentation.export / "numbers.json")["results"] == results["results"]
    )
    assert not (root / ".artifacts").exists()
    assert (
        inputs.source(root, bank_id, "compute", experiment="exp022").reference == before
    )
    assert presentation_id not in (presentation.export / "dt_sweep.svg").read_text()


def test_selected_bank_is_new_source_without_requiring_its_import_history(lab):
    root, bank_id, calls = lab
    directory = root / ".pingstore/runs" / bank_id
    record = load_json(directory / "run.json")
    record["inputs"] = {
        "import": {
            "run_id": "exp022-r999-compute",
            "payload_digest": "sha256:" + "0" * 64,
        }
    }
    write_json_atomic(directory / "run.json", record)
    before = (directory / "run.json").read_bytes()
    identity = compute.compute(bank_id)
    result = inputs.source(root, identity, "compute")
    boundary = result.record["source_boundary"]
    assert boundary["policy"] == "selected-v4-training-bank"
    assert (
        boundary["banks"][bank_id]["historical_inputs_not_traversed"]
        == record["inputs"]
    )
    assert boundary["banks"][bank_id]["reference"] == result.record["inputs"]["bank"]
    analysis_id = analyse.analyse(identity)
    present_id = present.present(analysis_id)
    assert (
        inputs.source(root, present_id, "present").record["source_boundary"] == boundary
    )
    assert (directory / "run.json").read_bytes() == before
    assert len(calls) == 20


def test_missing_selected_bank_blocks_before_reserving_or_simulating(lab):
    root, _, calls = lab
    with pytest.raises(PingstoreError, match="complete v4 input lineage: missing"):
        compute.compute("exp022-r999-compute")
    assert calls == []
    assert not list((root / ".pingstore/runs").glob("*exp044*"))


def test_wrong_stage_and_experiment_fail_without_downstream_work(lab):
    root, bank_id, calls = lab
    with pytest.raises(PingstoreError, match="not a exp044 compute"):
        analyse.analyse(bank_id)
    identity = compute.compute(bank_id)
    with pytest.raises(PingstoreError, match="not a exp044 analyse"):
        present.present(identity)
    with pytest.raises(PingstoreError, match="not a exp022 compute"):
        compute.compute(identity)
    assert len(calls) == 20


def test_corrupt_payload_and_snapshot_geometry_are_rejected(lab):
    root, bank_id, calls = lab
    identity = compute.compute(bank_id)
    output = inputs.source(root, identity, "compute")
    path = output.file("snapshot", recipe.cell_name(0.05, 42), "recording.npz")
    np.savez_compressed(path, spk_e=np.zeros((2, 4)), spk_i=np.zeros((2, 2)), dt=0.05)
    with pytest.raises(PingstoreError, match="checksum"):
        analyse.analyse(identity)
    resign(output.directory)
    with pytest.raises(PingstoreError, match="snapshot shape"):
        analyse.analyse(identity)
    assert len(calls) == 20


def test_ancestor_metadata_amendment_during_stage_is_allowed(lab):
    root, bank_id, _ = lab
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    compute_id = compute.compute(bank_id)
    source = inputs.source(root, compute_id, "compute")
    with inputs.execution(root, "analyse", sources={"compute": source}) as run:
        record = load_json(bank.directory / "run.json")
        record["execution"]["changed"] = True
        write_json_atomic(bank.directory / "run.json", record)
        write_json_atomic(run.export / "fixture.json", {})
    assert (root / ".pingstore/runs" / run.run_id).is_dir()


def test_imports_do_not_resolve_environment_training_roots(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments import exp044; "
            "assert not hasattr(exp044, 'RUN_PATHS'); assert not hasattr(exp044, 'cell_dir'); "
            "assert exp044.cell_name(0.05, 42) == 'ping__dt0p05__seed42'",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_production_and_snapshot_caps_are_explicit():
    assert recipe.configuration()["evaluation_samples"] == 1000
    assert recipe.configuration(smoke=True)["evaluation_samples"] == 100
    args = recipe.inference_args(
        Path("cell"),
        Path("weights_final.pth"),
        Path("out"),
        samples=100,
        sample_index=50,
    )
    assert "--max-samples" not in args
    assert args[args.index("--sample-index") + 1] == "50"


def test_v2_is_rejected_even_with_an_incomplete_payload(lab):
    root, bank_id, calls = lab
    path = root / ".pingstore/runs" / bank_id / "run.json"
    r = load_json(path)
    r["schema"] = "pingstore.run/v2"
    write_json_atomic(path, r)
    with pytest.raises(PingstoreError, match="requires v4"):
        compute.compute(bank_id)
    assert calls == []


def test_changed_authoritative_ancestor_metadata_is_allowed(lab):
    root, bank_id, _ = lab
    compute_id = compute.compute(bank_id)
    path = root / ".pingstore/runs" / bank_id / "run.json"
    r = load_json(path)
    r["execution"]["note"] = "manifest-only change"
    write_json_atomic(path, r)
    assert analyse.analyse(compute_id).endswith("-analyse")


def test_failure_stays_hidden_and_reservations_cannot_be_reused(lab, monkeypatch):
    root, bank_id, _ = lab
    identity = stages.reserve_stage(root / ".pingstore", recipe.SLUG, "compute")

    def fail(*a, **k):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(compute, "run_cli", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        compute.compute(bank_id, run_id=identity)
    assert (root / ".pingstore/runs" / f".{identity}.tmp").is_dir()
    assert not (root / ".pingstore/runs" / identity).exists()
    with pytest.raises(PingstoreError, match="interrupted execution"):
        compute.compute(bank_id, run_id=identity)


@pytest.mark.parametrize(
    "damage", ["missing_rate", "zero_samples", "nan_rate", "missing_history"]
)
def test_missing_measurements_never_become_zeros(lab, damage):
    root, bank_id, calls = lab
    if damage == "missing_history":
        bank = inputs.source(root, bank_id, "compute", experiment="exp022")
        path = bank.export / recipe.cell_name(0.05, 42) / "metrics.json"
        r = load_json(path)
        r["epochs"].pop()
        write_json_atomic(path, r)
        resign(bank.directory)
        with pytest.raises(PingstoreError, match="incomplete training history"):
            compute.compute(bank_id)
        assert not calls
        return
    compute_id = compute.compute(bank_id)
    output = inputs.source(root, compute_id, "compute")
    path = output.file("infer", recipe.cell_name(0.05, 42), "metrics.json")
    r = load_json(path)
    if damage == "missing_rate":
        del r["rates_hz"]["hid"]
    elif damage == "zero_samples":
        r["n_total"] = 0
    else:
        r["rates_hz"]["hid"] = float("nan")
    write_json_atomic(path, r)
    resign(output.directory)
    with pytest.raises(PingstoreError):
        analyse.analyse(compute_id)
    assert len(calls) == 20


@pytest.mark.parametrize("flag", [[], ["--plot-only"], ["--skip-training"]])
def test_combined_launchers_are_retired(flag):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "experiments.exp044", *flag],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "independent stages" in result.stderr


def test_collection_reserves_and_dispatches_explicit_stages(lab, monkeypatch):
    from experiments.collections.gamma_gated_sparsity.plan import build_plan

    root, bank_id, _ = lab
    plan = build_plan(root / "campaign", "fixture", smoke=True)
    plan["profile"] = "smoke"
    plan["exp022_manifest"] = str(root / "bank-manifest.json")
    write_json_atomic(Path(plan["exp022_manifest"]), {"pingstore_run_id": bank_id})
    row = next(
        r for s in plan["stages"] for r in s["experiments"] if r["slug"] == "exp044"
    )
    assert row["command"] == []
    assert row["execution"]["stages"] == ["compute", "analyse", "present"]
    ids = collection.reserve(root, row, origin="slurm-wilkes")
    for stage, identity in ids.items():
        assert identity.endswith("-" + stage)
        reservation = load_json(
            root
            / ".pingstore/runs"
            / f".{identity}.tmp"
            / ".reservation.json"
        )
        assert reservation["origin"] == "slurm-wilkes"
    assert collection.reserve(root, row) == ids
    with pytest.raises(PingstoreError, match="legacy exp044"):
        collection.require_staged({"execution": {"mode": "monolithic"}})
    commands = []

    def dispatch(command, **kwargs):
        commands.append(command)
        stage = command[2].rsplit(".", 1)[-1]
        module = {"compute": compute, "analyse": analyse, "present": present}[stage]
        getattr(module, stage)(
            command[command.index("--source") + 1],
            run_id=command[command.index("--run-id") + 1],
        )
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(collection.subprocess, "run", dispatch)
    refs = collection.execute(root, plan, row)
    assert len(commands) == 3
    assert collection.execute(root, plan, row) == refs
    assert len(commands) == 3
    assert (
        collection.completed(root, plan, row).record["run_id"]
        == refs["present"]["run_id"]
    )
    plan["profile"] = "production"
    with pytest.raises(PingstoreError, match="profile"):
        collection.execute(root, plan, row)


def test_article_renders_selected_analysis(lab):
    from demolab_cli import _paths

    root, bank_id, _ = lab
    compute_id = compute.compute(bank_id)
    analysis_id = analyse.analyse(compute_id)
    present_id = present.present(analysis_id)
    output = inputs.source(root, present_id, "present")
    source_root = Path(__file__).resolve().parents[2]
    shutil.copytree(source_root / "writings", root / "writings")
    (root / ".demolab").mkdir()
    shutil.copy2(_paths.TYP / "lib.typ", root / ".demolab/lib.typ")
    write_json_atomic(
        root / "preview.json",
        {"exp044": {"exp044": "/" + str(output.export.relative_to(root))}},
    )
    document = root / "document.typ"
    document.write_text(
        '#set page(paper: "a4", margin: 18mm)\n#set text(size: 10pt)\n#import "writings/exp044.typ": body\n#body\n'
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
    text = (root / "writings/exp044.typ").read_text()
    headings = ["== Abstract", "== Results", "== Methods", "#reference-list"]
    assert [text.index(h) for h in headings] == sorted(text.index(h) for h in headings)
    assert "default: 256" not in text
    assert "9–14 Hz" not in text
    (output.export / "numbers.json").write_text("corrupt")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0


def _common_config() -> dict:
    return {
        "model": "ping",
        "dataset": "mnist",
        "max_samples": 7000,
        "epochs": 50,
        "t_ms": 200.0,
        "tau_ampa_ms": 2.0,
        "tau_gaba_ms": 6.0,
        "input_rate": 25.0,
        "input_rate_sampling": "fixed",
        "hidden_sizes": [1024],
        "n_in": 784,
        "n_hidden": 1024,
        "n_inh": 256,
        "n_out": 10,
        "ei_strength": 1.0,
        "w_in": [0.9, 0.09],
        "w_in_initial_zero_fraction": 0.95,
        "readout_mode": "mem-mean",
        "readout_w_init_mean": 1.12060546875,
        "readout_w_init_std": 0.8349609375,
        "surrogate_slope": 1.0,
        "lr": 0.0004,
        "batch_size": 256,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "v_grad_dampen": 1000.0,
        "dales_law": True,
        "trainable_w_ei": False,
        "trainable_w_ie": False,
        "dataset_split": {
            "optimizer_train_samples": 6300,
            "validation_samples": 700,
            "official_test_samples": 10000,
            "checkpoint_selection_partition": "validation",
            "official_test_used_during_training": False,
            "source_train_partition": "official_mnist_train",
            "source_test_partition": "official_mnist_test",
            "split_seed": 42,
            "validation_fraction": 0.1,
        },
        "validation_encoder_draws": {
            "count": 3,
            "encoder_seeds": [1, 2, 3],
            "input_rate_seeds": [4, 5, 6],
        },
        "fr_reg_upper_strength": 0.0,
        "fr_reg_upper_target_hz": 0.0,
        "recurrent_initial_zero_fraction": 0.0,
        "adaptive_threshold": False,
        "train_leak": False,
        "signed_readout": False,
        "readout_bias": False,
        "trainable_w_ee": False,
        "trainable_w_ii": False,
        "state_clamp": False,
        "ei_ratio": 0.25,
        "w_ee": 0.0,
        "readout_reduction": "mean",
        "readout_reference": "absolute",
        "readout_units": "mV",
        "readout_w_out_scale": 1.0,
        "tau_m_e_bounds_ms": [10.0, 30.0],
        "tau_m_i_bounds_ms": [5.0, 15.0],
        "readout_tau_bounds_ms": [10.0, 30.0],
        "adapt_tau_bounds_ms": [20.0, 100.0],
        "adapt_strength_init_mv": 0.0,
        "adapt_strength_max_mv": 10.0,
    }


def _write_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[tuple[float, int], Path]:
    directories: dict[tuple[float, int], Path] = {}
    for dt_ms in exp044.DT_SWEEP_MS:
        for seed in exp044.SEEDS:
            directory = tmp_path / f"ping__{exp044.dt_label(dt_ms)}__seed{seed}"
            directory.mkdir()
            config = {**_common_config(), "dt": dt_ms, "seed": seed}
            (directory / "config.json").write_text(json.dumps(config))
            directories[(dt_ms, seed)] = directory
    return directories


def test_training_contract_verifies_all_15_cells(tmp_path: Path, monkeypatch) -> None:
    _write_cells(tmp_path, monkeypatch)
    contract = evidence.training_contract(tmp_path)
    assert contract["common"]["batch_size"] == 256
    assert len(contract["cells"]) == 15
    assert {cell["dt_ms"] for cell in contract["cells"]} == set(exp044.DT_SWEEP_MS)
    assert {cell["seed"] for cell in contract["cells"]} == set(exp044.SEEDS)


@pytest.mark.parametrize("field", exp044.TRAINING_COMMON_FIELDS)
def test_training_contract_rejects_each_unregistered_difference(
    field: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    directories = _write_cells(tmp_path, monkeypatch)
    target = directories[(exp044.DT_SWEEP_MS[-1], exp044.SEEDS[-1])]
    config = json.loads((target / "config.json").read_text())
    value = config[field]
    if isinstance(value, bool):
        config[field] = not value
    elif isinstance(value, list):
        config[field] = [*value, 999]
    elif isinstance(value, (int, float)):
        config[field] = value + 1
    else:
        config[field] = f"{value}-mismatch"
    (target / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=f"config {field}="):
        evidence.training_contract(tmp_path)


@pytest.mark.parametrize(("field", "value"), [("dt", 0.2), ("seed", 99)])
def test_training_contract_rejects_unregistered_identity(
    field: str,
    value: object,
    tmp_path: Path,
    monkeypatch,
) -> None:
    directories = _write_cells(tmp_path, monkeypatch)
    target = directories[(exp044.DT_SWEEP_MS[0], exp044.SEEDS[0])]
    config = json.loads((target / "config.json").read_text())
    config[field] = value
    (target / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=f"config {field}"):
        evidence.training_contract(tmp_path)
