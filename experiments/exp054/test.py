"""Exp054 stage contracts with synthetic recordings, never production simulation."""

import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from experiments.exp054 import (
    analyse,
    compute,
    evidence,
    inputs,
    measurements,
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
from pingstore.discovery import discover_runs


def recording(cfg):
    result = {
        "dt": np.float32(cfg["dt_ms"]),
        "T": np.int32(cfg["sim_ms"] / cfg["dt_ms"]),
        "n_trials": np.int32(1),
        "n_e": np.int32(cfg["n_e"]),
        "n_i": np.int32(cfg["n_i"]),
    }
    rng = np.random.default_rng(42)
    for prefix, width in (("e", cfg["n_e"]), ("i", cfg["n_i"]), ("out", 10)):
        slots = np.sort(rng.choice(int(result["T"]) * width, size=1500, replace=False))
        result.update(
            {
                f"{prefix}_trial": np.zeros(len(slots), np.int32),
                f"{prefix}_t": (slots // width).astype(np.int32),
                f"{prefix}_cell": (slots % width).astype(np.int32),
            }
        )
    return result


def config_record(cfg, item):
    return {
        **{
            key: cfg[key]
            for key in ("refractory_e_ms", "refractory_i_ms", "refractory_policy")
            if key in cfg
        },
        "mode": "sim",
        "model": "ping",
        "input": "synthetic-spikes",
        "n_hidden": [cfg["n_e"]],
        "n_inh": cfg["n_i"],
        "n_batch": 1,
        "n_in": cfg["n_e"] if item["private"] else cfg["shared_n_in"],
        "t_ms": cfg["sim_ms"],
        "dt": cfg["dt_ms"],
        "tau_gaba": cfg.get("tau_gaba_ms"),
        "seed": cfg["seed"],
        "spike_rate": item["rate_hz"],
        "w_ei_mean": item["wei"],
        "w_ie_mean": item["wie"],
        "private_w_in": item["private"],
        "w_in": [cfg["private_w_in"] if item["private"] else cfg["shared_w_in"]],
        "w_in_initial_zero_fraction": cfg["shared_zero_fraction"],
        "dales_law": True,
        "recurrent_initial_zero_fraction": 0.0,
        "scale_w_in": 1.0,
        "scale_w_ei": 1.0,
        "scale_w_ie": 1.0,
    }


@pytest.fixture
def lab(tmp_path, monkeypatch):
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(
        stages,
        "memberships",
        lambda _: {"exp054": "test"},
    )
    monkeypatch.setattr(
        stages, "_capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    monkeypatch.setenv("PINGLAB_SMOKE", "1")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    calls = []
    cfg = recipe.configuration(smoke=True)
    data = recording(cfg)

    def simulate(args, **kwargs):
        calls.append(args)
        output = Path(args[args.index("--out-dir") + 1])
        item = next(j for j in recipe.jobs(cfg) if j["id"] == output.name)
        assert args == recipe.simulation_args(cfg, item, output)
        write_json_atomic(output / "config.json", config_record(cfg, item))
        write_json_atomic(output / "metrics.json", {"fixture": True})
        compact = {k: v for k, v in data.items() if not k.startswith("out_")}
        burn = int(args[args.index("--recording-start-step") + 1])
        for population in ("e", "i"):
            keep = compact[f"{population}_t"] >= burn
            for field in ("trial", "t", "cell"):
                compact[f"{population}_{field}"] = compact[f"{population}_{field}"][
                    keep
                ]
        compact["recording_start_step"] = np.int32(burn)
        np.savez_compressed(output / "rasters.npz", **compact)
        (output / "run.sh").write_text("# synthetic fixture, never executed\n")

    monkeypatch.setattr(compute, "run_cli", simulate)
    return tmp_path, calls


def resign(path):
    manifest = load_json(path / "run.json")
    manifest["payload_digest"] = payload_digest(path)
    write_json_atomic(path / "run.json", manifest)


@pytest.mark.parametrize("smoke", [False, True])
def test_population_recipe_and_input_channels(smoke):
    current = recipe.configuration(smoke=smoke)
    assert current["n_e"] == 1024 and current["n_i"] == 256
    assert current["dt_ms"] == 0.1
    assert current["tau_gaba_ms"] == 6.0
    assert recipe.validate(current) == current
    assert "mean_field" not in current
    for cfg in (current,):
        assert recipe.validate(cfg) == cfg
        for private, channels in ((True, cfg["n_e"]), (False, 200)):
            args = recipe.simulation_args(
                cfg, recipe.job(cfg, 0, 0, 100, private), Path("probe")
            )
            assert args[args.index("--n-in") + 1] == str(channels)
            assert args[args.index("--dt") + 1] == str(cfg["dt_ms"])
            if "tau_gaba_ms" in cfg:
                assert args[args.index("--tau-gaba") + 1] == "6.0"
                item = recipe.job(cfg, 0, 0, 100, private)
                record = config_record(cfg, item)
                evidence.simulation_config(record, cfg, item)
                record["tau_gaba"] = 9.0
                with pytest.raises(PingstoreError, match="configuration"):
                    evidence.simulation_config(record, cfg, item)
            assert args[args.index("--recording-start-step") + 1] == str(
                round(cfg["burn_ms"] / cfg["dt_ms"])
            )
        assert len(recipe.jobs(cfg)) == (51 if smoke else 136)
    for invalid in (
        {**current, "n_e": 256},
        {**current, "n_i": 1024},
        {**current, "schema": "exp054.recipe/v999"},
        {**current, "tau_gaba_ms": 9.0},
        {**current, "dt_ms": 0.25},
    ):
        with pytest.raises(PingstoreError, match="recipe"):
            recipe.validate(invalid)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_raster_timestep_matches_recording_precision(tmp_path, dtype):
    cfg = recipe.configuration()
    data = recording(cfg)
    data["dt"] = dtype(0.1)
    path = tmp_path / "rasters.npz"
    np.savez_compressed(path, **data)
    evidence.raster(path, cfg)
    data["dt"] = np.nextafter(dtype(0.1), dtype(1.0))
    np.savez_compressed(path, **data)
    with pytest.raises(PingstoreError, match="dimensions"):
        evidence.raster(path, cfg)


def test_independent_stages_and_all_figures(lab, monkeypatch):
    root, calls = lab
    identity = compute.compute()
    source = inputs.source(root, identity, "compute")
    assert len(calls) == 51 and source.record["inputs"] == {}
    assert len(list(source.export.glob("probe--*--rasters.npz"))) == 51
    assert not list((source.directory / "export/evidence").rglob("rasters.npz"))
    original = source.reference
    monkeypatch.setenv("PINGLAB_SMOKE", "0")
    monkeypatch.setattr(
        compute, "run_cli", lambda *a, **k: pytest.fail("analysis simulated")
    )
    analysis_id = analyse.analyse(identity)
    analysis = inputs.source(root, analysis_id, "analyse")
    assert inputs.configuration(analysis)["profile"] == "smoke"
    numbers = load_json(analysis.export / "results.json")
    assert "mean_field" not in numbers
    assert discover_runs(root / ".pingstore/runs") == []
    monkeypatch.setattr(
        measurements,
        "recordings",
        lambda *a: pytest.fail("presentation measured rasters"),
    )
    output = inputs.source(root, present.present(analysis_id), "present")
    assert output.record["inputs"] == {"analysis": analysis.reference}
    assert all((output.export / name).stat().st_size > 1000 for name in recipe.FIGURES)
    assert all(p.is_file() for p in output.export.iterdir())
    document = load_json(output.export / "numbers.json")
    assert all(document[k] == v for k, v in numbers.items())
    assert inputs.source(root, identity, "compute").reference == original
    assert not (root / ".artifacts").exists()
    assert [r["id"] for r in discover_runs(root / ".pingstore/runs")] == [
        output.record["run_id"]
    ]


@pytest.mark.parametrize("smoke,count,points", [(False, 136, 11), (True, 51, 6)])
def test_preserved_recipe_and_shared_origin(smoke, count, points):
    cfg = recipe.configuration(smoke=smoke)
    assert len(recipe.jobs(cfg)) == count
    assert len(cfg["wei_mean"]) == len(cfg["wie_mean"]) == points
    assert recipe.turnon_points(cfg)[-1] == ("C", points - 1, points - 1)
    assert "mean_field" not in cfg
    assert len({j["id"] for j in recipe.jobs(cfg)}) == count


def test_lossless_repacking_and_nan_codec(tmp_path):
    cfg = recipe.configuration(smoke=True)
    original, packed = tmp_path / "original.npz", tmp_path / "packed.npz"
    np.savez(original, **recording(cfg))
    evidence.repack(original, packed)
    with zipfile.ZipFile(original) as a, zipfile.ZipFile(packed) as b:
        assert a.namelist() == b.namelist()
        assert all(a.read(n) == b.read(n) for n in a.namelist())
    document = {
        "ac": np.array([np.nan, 1.0, 2.0]),
        "missing": np.nan,
        "spikes": np.array([0, 1], np.int8),
    }
    evidence.write(tmp_path, document)
    restored = evidence.read(tmp_path)
    np.testing.assert_array_equal(restored["ac"], document["ac"])
    assert np.isnan(restored["missing"]) and restored["spikes"].dtype == np.int8


@pytest.mark.parametrize(
    "damage", ["dt", "trial", "range", "duplicate", "fields", "object"]
)
def test_bad_recordings_are_rejected(tmp_path, damage):
    cfg = recipe.configuration(smoke=True)
    data = recording(cfg)
    if damage == "dt":
        data["dt"] = np.float32(0.5)
    elif damage == "trial":
        data["e_trial"][0] = 1
    elif damage == "range":
        data["e_cell"][0] = cfg["n_e"]
    elif damage == "duplicate":
        for field in ("trial", "t", "cell"):
            data[f"e_{field}"][1] = data[f"e_{field}"][0]
    elif damage == "fields":
        del data["out_cell"]
    else:
        data["e_t"] = data["e_t"].astype(object)
    np.savez(tmp_path / "rasters.npz", **data)
    with pytest.raises((PingstoreError, ValueError)):
        evidence.raster(tmp_path / "rasters.npz", cfg)


@pytest.mark.parametrize("stage", ["compute", "analyse", "present"])
def test_failures_remain_hidden_and_cannot_resume(lab, monkeypatch, stage):
    root, _ = lab

    def fail(*a, **k):
        raise RuntimeError("fixture failure")

    if stage == "compute":
        monkeypatch.setattr(compute, "run_cli", fail)
        operation = compute.compute
    else:
        identity = compute.compute()
        if stage == "analyse":
            monkeypatch.setattr(measurements, "recordings", fail)
            operation = partial(analyse.analyse, identity)
        else:
            analysis = analyse.analyse(identity)
            monkeypatch.setattr(plots, "fig_turnon_maps_compound", fail)
            operation = partial(present.present, analysis)
    before = {p.name for p in (root / ".pingstore/runs").glob("exp054-*")}
    with pytest.raises(RuntimeError, match="fixture failure"):
        operation()
    assert {p.name for p in (root / ".pingstore/runs").glob("exp054-*")} == before
    (hidden,) = (root / ".pingstore/runs").glob(".exp054-*.tmp")
    assert (hidden / ".writer.lock").exists()
    with pytest.raises(PingstoreError, match="interrupted"):
        operation(run_id=hidden.name[1:-4])


@pytest.mark.parametrize("damage", ["payload", "layout", "v2", "recipe", "inventory"])
def test_corrupt_inputs_rejected_before_reservation(lab, damage):
    root, _ = lab
    identity = compute.compute()
    path = root / ".pingstore/runs" / identity
    record = load_json(path / "run.json")
    if damage == "payload":
        next(path.glob("export/probe--*--rasters.npz")).write_bytes(b"corrupt")
    elif damage == "manifest":
        record["stage"] = "present"
    elif damage == "layout":
        (path / "illegal").write_text("bad root")
    elif damage == "v2":
        record["schema"] = "pingstore.run/v2"
    elif damage == "recipe":
        record["execution"]["configuration"]["seed"] = 99
    elif damage == "inventory":
        index = load_json(path / "export/recordings.json")
        index["jobs"].pop()
        write_json_atomic(path / "export/recordings.json", index)
        record["payload_digest"] = payload_digest(path)
    write_json_atomic(path / "run.json", record)
    with pytest.raises((PingstoreError, ValueError)):
        analyse.analyse(identity)
    assert not list((root / ".pingstore/runs").glob(".exp054-*.tmp"))


def test_ancestor_metadata_change_does_not_change_payload_identity(lab):
    root, _ = lab
    identity = compute.compute()
    analysis = analyse.analyse(identity)
    path = root / ".pingstore/runs" / identity / "run.json"
    record = load_json(path)
    record["execution"]["note"] = "changed authoritative provenance"
    write_json_atomic(path, record)
    assert present.present(analysis).endswith("-present")


def test_source_change_during_stage_prevents_completion(lab, monkeypatch):
    root, _ = lab
    identity = compute.compute()
    original = measurements.recordings

    def changing(source, cfg):
        result = original(source, cfg)
        path = root / ".pingstore/runs" / identity / "export/recordings.json"
        document = load_json(path)
        document["note"] = "changed during analysis"
        write_json_atomic(path, document)
        return result

    monkeypatch.setattr(measurements, "recordings", changing)
    with pytest.raises(PingstoreError):
        analyse.analyse(identity)
    assert len(list((root / ".pingstore/runs").glob("exp054-*"))) == 1


def test_hpc_requires_prior_reservation(lab, monkeypatch):
    root, calls = lab
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    with pytest.raises(PingstoreError, match="reserved before"):
        compute.compute()
    assert not calls
    identity = stages.reserve_stage(
        root / ".pingstore", "exp054", "compute", origin="slurm"
    )
    assert compute.compute(run_id=identity) == identity
    assert inputs.source(root, identity, "compute").record["origin"] == "slurm"


def test_measured_rates_use_post_burn_full_populations(tmp_path):
    cfg = recipe.configuration(smoke=True)
    data = recording(cfg)
    e, i = measurements.dense(data, cfg)
    assert e.shape == (3000, 1024) and i.shape == (3000, 256)
    count = np.count_nonzero(data["e_t"] >= 1000)
    assert measurements.score(e, cfg)["rate"] == count / (1024 * 0.3)


def test_null_lag_label_uses_renderable_mathtext(tmp_path):
    cfg = recipe.configuration(smoke=True)
    data = measurements.score(measurements.dense(recording(cfg), cfg)[0], cfg)
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph.*missing from font")
        with plots.configured(cfg):
            plots.fig_null_autocorr([data], [data], tmp_path / "null.png")
    assert (tmp_path / "null.png").is_file()


def test_contrast_can_reach_one_and_silent_data_remain_undefined():
    from experiments.helpers.rhythmicity import rhythmicity_scalars

    lags = np.arange(8, dtype=float)
    result = rhythmicity_scalars(lags, [np.nan, 4, 4, 0, 0, 0, 1, 1], [0.5], [1])
    assert result["contrast"] == 1.0
    cfg = recipe.configuration()
    steps = round((cfg["sim_ms"] - cfg["burn_ms"]) / cfg["dt_ms"])
    result = measurements.score(np.zeros((steps, cfg["n_e"]), np.int8), cfg)
    assert np.isnan(result["contrast"])


def test_postburn_sparse_evidence_preserves_all_analysis_inputs(tmp_path):
    cfg = recipe.configuration(smoke=True)
    full = recording(cfg)
    burn = int(cfg["burn_ms"] / cfg["dt_ms"])
    lean = {k: v for k, v in full.items() if not k.startswith("out_")}
    for population in ("e", "i"):
        keep = lean[f"{population}_t"] >= burn
        for field in ("trial", "t", "cell"):
            key = f"{population}_{field}"
            lean[key] = lean[key][keep]
    lean["recording_start_step"] = np.int32(burn)
    path = tmp_path / "rasters.npz"
    np.savez_compressed(path, **lean)
    validated = evidence.raster(path, cfg)
    for original, selected in zip(
        measurements.dense(full, cfg), measurements.dense(validated, cfg), strict=True
    ):
        np.testing.assert_array_equal(original, selected)
    lean["recording_start_step"] = np.int32(burn + 1)
    np.savez_compressed(path, **lean)
    with pytest.raises(PingstoreError, match="dimensions differ"):
        evidence.raster(path, cfg)
