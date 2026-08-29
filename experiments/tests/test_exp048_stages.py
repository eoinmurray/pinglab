"""Synthetic exp048 contract probes; no training or production inference."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from experiments.exp048 import (
    analyse,
    compute,
    evidence,
    inputs,
    measurements,
    plots,
    present,
    recipe,
    stimuli,
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
        stages, "memberships", lambda _: {"exp022": "demo", "exp048": "demo"}
    )
    monkeypatch.setattr(
        stages, "_capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    # Test-only short streams. There is no production CLI recipe override.
    for name, value in {
        "TRAINED_T_MS": 0.4,
        "TAU_HEADLINE_MS": 0.2,
        "TAU_SWEEP_MS": [0.2, 0.4],
        "TAU_GRID_MS": [0.2, 0.4],
        "RATE_GRID_HZ": [5.0, 25.0],
        "LOW_RATE_HZ": [0.1, 0.5, 2.0],
        "N_STREAMS": 1,
        "N_GRID_STREAMS": 1,
        "LOW_RATE_STREAMS": 1,
        "VARYING_HEADLINE": [
            (0.2, 10.0),
            (0.4, 100.0),
            (0.2, 25.0),
            (0.4, 200.0),
            (0.2, 15.0),
        ],
    }.items():
        monkeypatch.setattr(recipe, name, value)
    monkeypatch.setattr(measurements, "TRAINED_T_MS", recipe.TRAINED_T_MS)
    with stages.stage_run(
        tmp_path, "exp022", "compute", export_root="export/cells"
    ) as run:
        for seed in recipe.SEEDS:
            name = recipe.cell_name(seed)
            cell = run.export / "cells" / name
            cell.mkdir(parents=True)
            cfg = {
                "model": "ping",
                "dataset": "mnist",
                "seed": seed,
                "dt": recipe.DT,
                "t_ms": recipe.TRAINED_T_MS,
                "n_hidden": recipe.N_E,
                "n_inh": recipe.N_I,
                "n_in": recipe.N_IN,
                "n_out": recipe.N_CLASSES,
                "hidden_sizes": [recipe.N_E],
                "readout_mode": "mem-mean",
                "input_rate": recipe.INPUT_RATE_HZ,
                "ei_strength": 1.0,
                "fr_reg_upper_strength": 0.0,
                "epochs": 50,
                "max_samples": 7000,
                "dataset_split": {
                    "checkpoint_selection_partition": "validation",
                    "official_test_used_during_training": False,
                },
            }
            write_json_atomic(cell / "config.json", cfg)
            (cell / "weights.pth").write_bytes(f"best-{seed}".encode())
            (cell / "weights_final.pth").write_bytes(f"final-{seed}".encode())
            write_json_atomic(
                cell / "metrics.json",
                {
                    "best_epoch": 3,
                    "training_cell_name": name,
                    "config": cfg,
                    "checkpoints": {
                        "best_validation": {
                            "filename": "weights.pth",
                            "epoch": 3,
                            "sha256": file_sha256(cell / "weights.pth"),
                        }
                    },
                },
            )
    x = np.random.default_rng(5).random((100, 784), dtype=np.float32)
    y = np.tile(np.arange(10), 10)
    monkeypatch.setattr(compute, "load_mnist_split", lambda **_: (x, x, y, y))
    calls = []

    def simulate(args, **kwargs):
        assert kwargs == {"no_sync": True}
        calls.append(args)

        def value(key):
            return args[args.index(key) + 1]

        assert Path(value("--load-weights")).name == "weights.pth"
        out = Path(value("--out-dir"))
        out.mkdir()
        if args[0] == "dump-weights":
            w = (
                np.random.default_rng(9)
                .normal(size=(recipe.N_E, recipe.N_CLASSES))
                .astype(np.float32)
            )
            np.savez(
                out / "weights_dump.npz",
                W_ff_0_trained=np.ones((2, 2)),
                W_ff_1_trained=w,
            )
        else:
            with np.load(value("--input-file")) as raw:
                spike_input = raw["input_spikes"]
            n = len(spike_input)
            d = {
                "T": np.int32(n),
                "n_e": np.int32(recipe.N_E),
                "n_i": np.int32(recipe.N_I),
                "dt": np.float32(recipe.DT),
                "n_trials": np.int32(1),
            }
            for prefix in ("e", "i", "out"):
                d.update(
                    {
                        f"{prefix}_trial": np.zeros(n, dtype=np.int32),
                        f"{prefix}_t": np.arange(n, dtype=np.int32),
                        f"{prefix}_cell": np.zeros(n, dtype=np.int32),
                    }
                )
            np.savez(out / "rasters.npz", **d)
        cfg = load_json(Path(value("--load-config")))
        cfg.update(
            load_config=value("--load-config"), load_weights=value("--load-weights")
        )
        if args[0] == "sim":
            cfg["input_file"] = value("--input-file")
        write_json_atomic(out / "config.json", cfg)
        (out / "run.sh").write_text("synthetic test backend\n")

    monkeypatch.setattr(compute, "run_cli", simulate)
    return tmp_path, run.run_id, calls


def resign(directory):
    record = load_json(directory / "run.json")
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)


def test_rate_inset_matches_article_range_and_uncertainty(lab, monkeypatch, tmp_path):
    root, bank, _ = lab
    compute_id = compute.compute(bank)
    analysis_id = analyse.analyse(compute_id)
    numbers = load_json(
        inputs.source(root, analysis_id, "analyse").export / "results.json"
    )
    inspected = []

    def inspect(fig, *a, **kw):
        curve = fig.axes[1]
        zoom = curve.child_axes[0]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend = curve.get_legend()
        assert not legend.get_window_extent(renderer).overlaps(
            zoom.xaxis.label.get_window_extent(renderer)
        )
        assert legend.get_frame().get_facecolor() == (1, 1, 1, 1)
        assert legend.get_zorder() > curve.lines[1].get_zorder()
        inspected.append((zoom.get_xlim(), len(curve.collections)))

    monkeypatch.setattr(plots, "save_figure", inspect)
    plots.plot_grid_and_rate(
        numbers["grid_sweep_agg"],
        numbers["encoding_rate_psychometric"]["curve"],
        tmp_path / "grid",
        "test",
    )
    assert inspected[0][0] == (0, 10)
    assert inspected[0][1] >= 1
    article = (Path(__file__).resolve().parents[2] / "writings/exp048.typ").read_text()
    assert "0–#p10.input_rate_hz" in article
    assert "shaded band" in article
    assert "failure floor below" not in article


def test_production_recipe_and_rng_groups():
    jobs = recipe.jobs()
    assert len(jobs) == 197
    assert sum(j["streams"] for j in jobs) == 6512
    assert sum(j["streams"] * len(j["segments"]) for j in jobs) == 65110
    assert recipe.SEEDS == [42, 43, 44]
    assert recipe.CHECKPOINT_ROLE == "best_validation"
    assert len([j for j in jobs if j["kind"] == "grid"]) == 144
    assert len([j for j in jobs if j["kind"] == "low"]) == 27
    tau = [j for j in jobs if j["kind"] == "tau" and j["seed"] == 42]
    assert tau[0]["sample_seed"] == tau[4]["sample_seed"] == 184
    assert tau[0]["sample_group"] != tau[4]["sample_group"]
    assert tau[0]["poisson_seed"] == tau[4]["poisson_seed"] == 5242
    assert tau[4]["segments"][0] == [25.0, 200.0]


def test_encoding_and_sequential_sampling_match_legacy():
    x = np.random.default_rng(0).random((100, 784), dtype=np.float32)
    y = np.arange(100) % 10
    pixels = x[:10]
    a = stimuli.encode_stream(pixels, 1.0, 25.0, torch.Generator().manual_seed(17))
    b = stimuli.encode_varying_stream(
        pixels, [(1.0, 25.0)] * 10, torch.Generator().manual_seed(17)
    )
    assert torch.equal(a, b)
    jobs = [j for j in recipe.jobs() if j["kind"] == "grid" and j["seed"] == 42]
    old_rng = np.random.default_rng(42 + 555 + 42)
    new_rng = np.random.default_rng(jobs[0]["sample_seed"])
    for job in jobs:
        for _ in range(job["streams"]):
            idx = old_rng.choice(len(y), 10, replace=False)
            p, labels = compute.select(job, x, y, new_rng)
            np.testing.assert_array_equal(p, x[idx])
            np.testing.assert_array_equal(labels, y[idx])


def test_decoder_delay_and_matched_window():
    spikes = np.zeros((5, recipe.N_E), dtype=np.int8)
    spikes[0, 0] = 1
    w = np.zeros((recipe.N_E, 10), dtype=np.float32)
    w[0, 2] = 2
    v = measurements._v_out_series(spikes, w, 2.0)
    assert not v[0].any()
    assert v[1, 2] == pytest.approx((1 - np.exp(-0.1 / 2)) / 0.1 * 2)
    got = measurements.sliding_readout(spikes, w, 2.0, 0.2)
    np.testing.assert_allclose(
        got,
        np.array([v[max(0, t - 1) : t + 1].mean(axis=0) for t in range(5)]),
        rtol=1e-6,
    )


def test_stages_are_independent_and_repeatable(lab, monkeypatch):
    root, bank, calls = lab
    before = (root / ".pingstore/runs" / bank / "run.json").read_bytes()
    cid = compute.compute(bank)
    assert cid == "exp048-r001-compute"
    assert len(calls) == 3 + sum(j["streams"] for j in recipe.jobs())
    assert not (root / ".artifacts").exists()
    assert len(list((root / ".pingstore/runs").glob("exp048-*"))) == 1

    def forbidden(*a, **kw):
        raise AssertionError("upstream execution is forbidden")

    monkeypatch.setattr(compute, "compute", forbidden)
    monkeypatch.setattr(compute, "run_cli", forbidden)
    monkeypatch.setattr(compute, "load_mnist_split", forbidden)
    aid = analyse.analyse(cid)
    a = inputs.source(root, aid, "analyse")
    result = load_json(a.export / "results.json")
    assert len(result["grid_sweep_per_seed"]) == 12
    assert all(r["n_total"] == 30 for r in result["grid_sweep_agg"])
    assert len(result["encoding_rate_psychometric"]["curve"]) == 5
    monkeypatch.setattr(analyse, "analyse", forbidden)
    monkeypatch.setattr(analyse, "decode", forbidden)
    monkeypatch.setattr(measurements, "aggregate_grid_rows", forbidden)
    pid = present.present(aid)
    p = inputs.source(root, pid, "present")
    assert set(recipe.FIGURES) <= {f.name for f in p.export.iterdir()}
    assert all(f.is_file() for f in p.export.iterdir())
    numbers = load_json(p.export / "numbers.json")
    assert {k: numbers[k] for k in result} == result
    assert p.record["inputs"] == {"analysis": a.reference}
    assert (root / ".pingstore/runs" / bank / "run.json").read_bytes() == before
    assert not (root / ".artifacts").exists()


@pytest.mark.parametrize(
    "mutation", ["missing", "coordinate", "dt", "label", "seed", "input_hash", "recipe"]
)
def test_rejects_malformed_compute_even_if_resigned(lab, mutation):
    root, bank, _ = lab
    cid = compute.compute(bank)
    run = inputs.source(root, cid, "compute")
    first = run.export / "job-000/stream-000"
    if mutation == "missing":
        (first / "rasters.npz").unlink()
    elif mutation in ("coordinate", "dt"):
        p = first / "rasters.npz"
        d = evidence.load_arrays(p)
        d["e_t" if mutation == "coordinate" else "dt"] = (
            np.array([-1]) if mutation == "coordinate" else np.array(1.0)
        )
        np.savez_compressed(p, **d)
    elif mutation == "label":
        p = first / "stimulus.npz"
        d = evidence.load_arrays(p)
        d["labels"][0] = 10
        np.savez_compressed(p, **d)
    elif mutation in ("seed", "input_hash"):
        p = first / "stream.json"
        d = load_json(p)
        d["poisson_seed" if mutation == "seed" else "input_sha256"] = 0
        write_json_atomic(p, d)
    else:
        p = run.export / "evidence.json"
        d = load_json(p)
        d["jobs"] = d["jobs"][:-1]
        write_json_atomic(p, d)
    resign(run.directory)
    with pytest.raises((PingstoreError, OSError)):
        analyse.analyse(cid)
    assert not list((root / ".pingstore/runs").glob("exp048-*-analyse"))


def test_corrupt_bank_and_wrong_stage_fail_before_execution(lab):
    root, bank, calls = lab
    with pytest.raises(PingstoreError):
        analyse.analyse(bank)
    path = (
        root / ".pingstore/runs" / bank / "export/cells/ping__off__seed42/weights.pth"
    )
    path.write_bytes(b"corrupt")
    with pytest.raises(PingstoreError):
        compute.compute(bank)
    assert calls == []


def test_failure_remains_hidden_and_cannot_resume(lab, monkeypatch):
    root, bank, _ = lab

    def fail(*a, **kw):
        raise RuntimeError("simulator failure")

    monkeypatch.setattr(compute, "run_cli", fail)
    with pytest.raises(RuntimeError, match="simulator failure"):
        compute.compute(bank)
    runs = root / ".pingstore/runs"
    hidden = list(runs.glob(".exp048-*.tmp"))
    assert len(hidden) == 1
    assert not list(runs.glob("exp048-*"))
    with pytest.raises(PingstoreError, match="interrupted"):
        compute.compute(bank, run_id=hidden[0].name[1:-4])


def test_full_ancestry_rechecked_after_work(lab, monkeypatch):
    root, bank, _ = lab
    cid = compute.compute(bank)
    aid = analyse.analyse(cid)
    original = plots.plot_grid_and_rate

    def mutate(*args):
        original(*args)
        path = root / ".pingstore/runs" / bank / "run.json"
        path.write_text(path.read_text() + "\n")

    monkeypatch.setattr(plots, "plot_grid_and_rate", mutate)
    with pytest.raises(PingstoreError):
        present.present(aid)
    assert not list((root / ".pingstore/runs").glob("exp048-*-present"))


def test_v2_rejected(lab):
    root, bank, calls = lab
    p = root / ".pingstore/runs" / bank / "run.json"
    d = load_json(p)
    d["schema"] = "pingstore.run/v2"
    write_json_atomic(p, d)
    with pytest.raises(PingstoreError):
        compute.compute(bank)
    assert not calls


def test_cli_requires_source_and_combined_runner_is_retired():
    root = Path(__file__).resolve().parents[2]
    for stage in ("compute", "analyse", "present"):
        result = subprocess.run(
            [sys.executable, "-m", f"experiments.exp048.{stage}"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "--source" in result.stderr
    result = subprocess.run(
        [sys.executable, "-m", "experiments.exp048"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "explicit compute, analyse or present" in result.stderr


@pytest.mark.parametrize("mutation", ["grid", "curve", "figure", "bank_pin"])
def test_present_rejects_incomplete_or_mismatched_analysis(lab, mutation):
    root, bank, _ = lab
    aid = analyse.analyse(compute.compute(bank))
    run = inputs.source(root, aid, "analyse")
    if mutation in ("grid", "curve"):
        p = run.export / "results.json"
        data = load_json(p)
        if mutation == "grid":
            data["grid_sweep_agg"].pop()
        else:
            data["encoding_rate_psychometric"]["curve"].pop()
        write_json_atomic(p, data)
    elif mutation == "figure":
        p = run.export / "varying.npz"
        data = evidence.load_arrays(p)
        data["segment_steps"][0] += 1
        np.savez_compressed(p, **data)
    else:
        p = run.directory / "run.json"
        data = load_json(p)
        data["inputs"]["bank"]["run_json_sha256"] = "0" * 64
        write_json_atomic(p, data)
    resign(run.directory)
    with pytest.raises(PingstoreError):
        present.present(aid)
    assert not list((root / ".pingstore/runs").glob("exp048-*-present"))


def test_checkpoint_role_mismatch_rejected_even_with_valid_run_checksum(lab):
    root, bank, calls = lab
    directory = root / ".pingstore/runs" / bank
    p = directory / "export/cells/ping__off__seed42/metrics.json"
    d = load_json(p)
    d["checkpoints"]["best_validation"]["filename"] = "weights_final.pth"
    write_json_atomic(p, d)
    resign(directory)
    with pytest.raises(PingstoreError, match="best_validation"):
        compute.compute(bank)
    assert not calls


def test_preallocated_source_neutral_identity_and_retry(lab, monkeypatch):
    root, bank, _ = lab
    rid = stages.reserve_stage(root / ".pingstore", "exp048", "compute")
    assert compute.compute(bank, run_id=rid) == rid == "exp048-r001-compute"
    with pytest.raises(PingstoreError, match="unused reserved"):
        compute.compute(bank, run_id=rid)
    aid = analyse.analyse(rid)

    def fail(*a, **kw):
        raise RuntimeError("render failure")

    monkeypatch.setattr(plots, "plot_headline_stream", fail)
    with pytest.raises(RuntimeError, match="render failure"):
        present.present(aid)
    assert not list((root / ".pingstore/runs").glob("exp048-*-present"))
    assert len(list((root / ".pingstore/runs").glob(".exp048-*-present.tmp"))) == 1


def test_real_simulator_parser_preserves_explicit_stream_contract(lab):
    # Parse the exact emitted arguments; do not execute a forward pass.
    root, bank, calls = lab
    compute.compute(bank)
    for args in (calls[0], next(a for a in calls if a[0] == "sim")):
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json, sys; from tools.snnsim.tool import parse_args; "
                "print(json.dumps(vars(parse_args(sys.argv[1:]))))",
                *args,
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        )
        parsed = json.loads(process.stdout.splitlines()[-1])
        assert parsed["readout_mode"] == "mem-mean"
        assert parsed["n_hidden"] == [1024]
        assert parsed["dt"] == 0.1
        assert not parsed.get("infer", False)
        if args[0] == "sim":
            evidence.simulation_configuration(
                parsed,
                load_json(
                    root
                    / ".pingstore/runs"
                    / bank
                    / "export/cells/ping__off__seed42/config.json"
                ),
                args,
            )


def test_sample_sem_precision_is_preserved():
    rows = [
        {
            "tau_ms": 200.0,
            "input_rate_hz": 25.0,
            "acc": a,
            "n_total": 400,
            "train_seed": s,
        }
        for a, s in zip([90.25, 91.75, 89.5], [42, 43, 44])
    ]
    result = measurements.aggregate_grid_rows(rows)[0]
    a = np.array([90.25, 91.75, 89.5], dtype=np.float32)
    assert result["acc"] == float(a.mean())
    assert result["acc_sem"] == float(a.std(ddof=1) / np.sqrt(3))
    assert result["n_total"] == 1200


def test_varying_decoder_uses_each_segments_own_window():
    job = recipe.jobs()[1]
    n = sum(round(t / recipe.DT) for t, _ in job["segments"])
    tt = np.arange(0, n, 17, dtype=np.int32)
    raw = {
        "T": np.array(n),
        "e_t": tt,
        "e_cell": tt % recipe.N_E,
        "i_t": tt,
        "i_cell": tt % recipe.N_I,
    }
    stimulus = {"pixels": np.zeros((5, 784)), "labels": np.arange(5)}
    w = np.random.default_rng(8).normal(size=(recipe.N_E, 10)).astype(np.float32)
    _, _, summary, figure = analyse.decode(job, raw, stimulus, w, 2.0)
    v = measurements._v_out_series(evidence.dense(raw, "e", recipe.N_E), w, 2.0)
    csum = np.concatenate([np.zeros((1, 10), dtype=np.float32), np.cumsum(v, axis=0)])
    cur, predictions = 0, []
    for tau, _ in job["segments"]:
        length = round(tau / recipe.DT)
        end = cur + length
        logits = (csum[end] - csum[cur]) / length
        predictions.append(int(np.argmax(logits)))
        np.testing.assert_array_equal(
            figure["probs"][end - 1], measurements.softmax_rowwise(logits)
        )
        cur = end
    assert summary["seg_preds"] == predictions
