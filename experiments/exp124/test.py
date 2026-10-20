"""Check calibrated semantics and the native compute/analyse/present contract.

The integration fixture uses synthetic cochlear events in an isolated temporary
Pingstore, so tests neither download SHD nor claim new benchmark evidence. Native
training, replay, snapshot retention and downstream validation still execute.
"""

import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from experiments.exp124 import analyse, compute, dataset, present, recipe
from experiments.exp124.custom.extensions import class_balanced_cross_entropy
from pingstore import stages
from pingstore.contracts import load_json
from snnlab.analysis import spike_autocorrelogram
from snnlab.sim import NetworkSnapshot, load_training_checkpoint


def test_configuration_is_fresh_and_validated():
    first = recipe.configuration()
    first["initial_zero_fractions"][0] = 0
    first["learning_rates"]["input"] = 1
    first["models"].clear()
    first["recording"]["ping"].clear()
    second = recipe.configuration()
    assert second["models"] == ["coba", "ping"]
    assert second["recording"]["ping"] == ["e1", "e2", "i1", "i2"]
    assert second["initial_zero_fractions"] == [0.95, 0.90]
    assert second["learning_rates"]["input"] == 5e-6
    assert second["full_dataset"] and second["epochs"] == 50
    assert recipe.configuration("local-check")["epochs"] == 1
    counts = [2] * 20
    resolved = recipe.configuration(
        train_class_counts=counts, validation_class_counts=[1] * 20
    )
    counts[0] = 99
    assert resolved["train_examples"] == 40
    assert resolved["validation_examples"] == 20
    assert resolved["train_class_counts"][0] == 2
    for override in (
        {"protocol": "typo"},
        {"epochs": 0},
        {"batch_size": 1.5},
        {"seed": -1},
        {"train_class_counts": [2] * 20},
        {"train_class_counts": [0] * 20, "validation_class_counts": [1] * 20},
    ):
        with pytest.raises(ValueError):
            recipe.configuration(**override)


def test_split_is_disjoint_reproducible_and_stratified():
    labels = np.repeat(np.arange(20), 13)
    cfg = recipe.configuration()
    training, validation = dataset.select_indices(labels, cfg)
    assert not set(training) & set(validation)
    assert len(training) + len(validation) == len(labels)
    assert (training, validation) == dataset.select_indices(labels, cfg)
    training, validation = dataset.select_indices(
        labels, recipe.configuration("local-check")
    )
    np.testing.assert_array_equal(np.bincount(labels[training]), np.ones(20))
    np.testing.assert_array_equal(np.bincount(labels[validation]), np.ones(20))


def test_equal_class_objective_does_not_weight_frequent_classes_more():
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0]])
    target = torch.tensor([0, 1, 0])
    losses = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    actual = class_balanced_cross_entropy(logits, target, {"classes": 2})
    torch.testing.assert_close(actual, ((losses[0] + losses[2]) / 2 + losses[1]) / 2)


def test_native_zero_decay_adamw_preserves_adam_updates():
    value = torch.tensor([0.1, -0.2, 2.0])
    adam, native = torch.nn.Parameter(value.clone()), torch.nn.Parameter(value.clone())
    groups = recipe.configuration()["optimizer"]
    optimizers = (
        torch.optim.Adam([adam], **groups),
        torch.optim.AdamW([native], **groups),
    )
    for step in range(3):
        for parameter, optimizer in zip((adam, native), optimizers):
            parameter.grad = torch.tensor([0.5, -0.2, 1.0]) * (step + 1)
            optimizer.step()
        torch.testing.assert_close(adam, native, rtol=0, atol=0)


def test_custom_encoding_boundaries_collisions_and_corruption():
    arrays = dict(
        event_offsets=np.array([0, 4]),
        event_time_ms=np.array([0.0, 0.9, 1.0, 2.0]),
        event_channel=np.array([0, 0, 1, 0]),
    )
    cfg = dict(
        duration_ms=2.0,
        sample_offset=0,
        input_noise_hz=0.0,
        spike_deletion_probability=0.0,
    )
    spikes = dataset.encode_shd(arrays, [0], dt_ms=1.0, channels=2, config=cfg, seed=1)
    torch.testing.assert_close(spikes[:, 0], torch.eye(2))
    deleted = dataset.encode_shd(
        arrays,
        [0],
        dt_ms=1.0,
        channels=2,
        config=dict(cfg, spike_deletion_probability=1.0),
        seed=1,
    )
    assert not deleted.any()
    inserted = dataset.encode_shd(
        arrays,
        [0],
        dt_ms=1.0,
        channels=2,
        config=dict(cfg, input_noise_hz=1000.0),
        seed=1,
    )
    assert inserted.all()


def test_autocorrelogram_matches_native_and_serializes_silence():
    spikes = np.zeros((500, 2))
    spikes[::10] = 1
    expected_lags, expected = spike_autocorrelogram(
        spikes, 1.0, max_lag_ms=200.0, bin_ms=1.0
    )
    actual = analyse.spike_autocorrelation(spikes, 1.0)
    np.testing.assert_array_equal(actual["lag_ms"], expected_lags)
    np.testing.assert_allclose(
        [np.nan if v is None else v for v in actual["correlation"]],
        expected,
        equal_nan=True,
    )
    assert all(
        v is None
        for v in analyse.spike_autocorrelation(np.zeros((500, 2)), 1.0)["correlation"]
    )


@pytest.mark.parametrize("resolved", [False, True])
def test_historical_inventories_come_from_executed_definitions(
    tmp_path, monkeypatch, resolved
):
    source = SimpleNamespace(
        record={"execution": {"configuration": {"models": ["coba"]}}},
        file=lambda *parts: tmp_path.joinpath(*parts),
    )
    graph = dict(
        populations=[
            dict(id="E9", spiking=True),
            dict(id="I9", spiking=True),
            dict(id="readout", spiking=False),
        ]
    )
    training = dict(
        parameter_groups=[
            dict(parameters=["executed_learned.weight"], frozen=False),
            dict(parameters=["executed_fixed.weight"], frozen=True),
        ]
    )
    if resolved:
        # Resolved native membership takes precedence over authoring groups.
        training["resolved_parameters"] = dict(trainable=["resolved_learned.weight"])
    monkeypatch.setattr(
        analyse,
        "network_definition",
        lambda source, model, role: graph if role == "graph.json" else training,
    )
    assert analyse.recorded_populations(source) == {"coba": ["e9", "i9"]}
    expected = "resolved_learned.weight" if resolved else "executed_learned.weight"
    assert analyse.trainable_parameters(source, "coba") == {expected}


@pytest.mark.integration
@pytest.mark.parametrize("parallel", [False, True])
def test_native_pipeline_retains_evidence_and_validates_sources(
    tmp_path, monkeypatch, parallel
):
    # Exercise both topologies and all stages without real data or permanent runs.
    cfg = recipe.configuration("local-check")
    cfg["duration_ms"] = 100.0
    cfg["autocorrelation_max_lag_ms"] = 40.0
    cfg["models"].reverse()
    cfg["performance"]["compile_mode"] = None
    configuration = recipe.configuration

    def fixture_configuration(protocol="full", **kwargs):
        resolved = configuration(protocol, **kwargs)
        resolved.update(copy.deepcopy(cfg))
        return resolved

    monkeypatch.setattr(recipe, "configuration", fixture_configuration)
    monkeypatch.setattr(stages, "memberships", lambda repo: {"exp124": "datasets"})
    monkeypatch.setattr(
        stages, "_capture_code", lambda *args: {"git_commit": "synthetic-fixture"}
    )
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setenv("SNNLAB_DEVICE", "cpu")
    monkeypatch.setenv("PINGLAB_CPU_THREADS", "1")
    labels = np.tile(np.arange(20), 2)
    event_times = np.repeat(np.arange(0.0, 100.0, 5.0), 700)
    event_channels = np.tile(np.arange(700), 20)
    arrays = dict(
        event_offsets=np.arange(41) * len(event_times),
        event_time_ms=np.tile(event_times, 40),
        event_channel=np.tile(event_channels, 40),
        labels=labels,
        official_train_indices=np.arange(40),
        speakers=np.zeros(40, dtype=int),
        split=np.repeat([0, 1], 20),
    )
    counts = dict(
        train_class_counts=[1] * 20,
        validation_class_counts=[1] * 20,
    )
    monkeypatch.setattr(
        dataset,
        "load_inputs",
        lambda *args: (
            labels,
            arrays,
            {"official_train_count": 40, "source": "synthetic fixture"},
            counts,
        ),
    )
    selected_parameters = {}
    native_train = compute.train

    def capture_selected(**kwargs):
        result = native_train(**kwargs)
        name = kwargs["save_selected_checkpoint"].name
        selected_parameters[name] = load_training_checkpoint(
            kwargs["save_selected_checkpoint"]
        ).parameters
        return result

    monkeypatch.setattr(compute, "train", capture_selected)
    if parallel:
        # Exercise actual spawned workers on CPU without paid/GPU resources.
        monkeypatch.setattr(
            compute,
            "execution_items",
            lambda cfg, *_: (
                [{"id": model, "device": "cpu"} for model in cfg["models"]],
                1,
            ),
        )
    entrypoint = compute.compute_parallel if parallel else compute.compute
    identity = entrypoint(protocol="local-check")
    source = stages.source_run(tmp_path / ".pingstore", identity)
    execution = source.record["execution"]
    assert execution["concurrency"]["mode"] == (
        "spawned_processes" if parallel else "sequential"
    )
    if parallel:
        assert len({worker["worker_pid"] for worker in execution["workers"]}) == 2
    assert set(execution["models"]) == set(cfg["models"])
    assert set(execution["recording_info"]) == set(cfg["models"])
    assert len(list((tmp_path / ".pingstore/runs").glob("exp124-*-compute"))) == 1
    for model in cfg["models"]:
        assert source.file(model, "circuit.svg").exists()
        for stage in ("start", "end"):
            assert source.file(model, "train", stage, "snapshot.json").exists()
        assert source.file(f"{model}--inference", "start", "snapshot.json").exists()
        membership = analyse.trainable_parameters(source, model)
        assert len(membership) == 3
        assert "readout_projection.weight" in membership
        assert "score_gain.weight" not in membership
        trained = analyse.parameters(source, model, "end")
        replayed = NetworkSnapshot.read(
            source.file(f"{model}--inference", "start", "snapshot.json")
        ).load_parameters()
        assert trained.keys() == replayed.keys()
        for name in trained:
            np.testing.assert_array_equal(trained[name], replayed[name].numpy())
        best = NetworkSnapshot.read(
            source.file(model, "best-validation", "start", "snapshot.json")
        ).load_parameters()
        # With one completed epoch, the best checkpoint must be the final one.
        selected = (
            {name: torch.from_numpy(value) for name, value in trained.items()}
            if parallel
            else selected_parameters[f"{model}.best-checkpoint"]
        )
        assert best.keys() == selected.keys()
        for name in best:
            torch.testing.assert_close(best[name], selected[name], rtol=0, atol=0)
        metrics = load_json(source.file(model, "metrics.json"))
        assert metrics["performance"]
        assert metrics["epochs"][-1]["epoch"] == 1
    # Only the experiment identity may come from today's recipe. Downstream
    # settings and inventories must still come from the completed source.
    monkeypatch.setattr(
        recipe, "configuration", lambda: dict(experiment=cfg["experiment"])
    )
    analysis_id = analyse.analyse(identity)
    presentation_id = present.present(analysis_id)
    output = stages.source_run(tmp_path / ".pingstore", presentation_id)
    numbers = load_json(output.file("numbers.json"))
    assert numbers["full_training"]["configuration"]["protocol"] == "local-check"
    assert "parameter_table" not in numbers
    analysis = stages.source_run(tmp_path / ".pingstore", analysis_id)
    assert "parameter_table" not in load_json(analysis.file("results.json"))
    sample = load_json(analysis.file("raster_measurements.json"))
    for model in cfg["models"]:
        for acf in sample["models"][model]["autocorrelations"].values():
            assert any(value is not None for value in acf["correlation"])
    # Render the actual report component from the original retained snapshots.
    # The renderer gets no current recipe values or preformatted parameter rows.
    site = Path(__file__).resolve().parents[2] / "site"
    script = r"""
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { experimental_AstroContainer } from 'astro/container';
import { readSnapshotEvidence } from './src/lib/snapshot-data.ts';
const [root, id, temporary] = process.argv.slice(2);
const evidence = await readSnapshotEvidence(id, {
  experiment: 'exp124', runsRoot: root,
  snapshots: { coba: 'coba/train/end/snapshot.json', ping: 'ping/train/end/snapshot.json' },
});
const require = createRequire(import.meta.resolve('astro'));
const { transform } = require('@astrojs/compiler-rs');
const compile = async (sourcePath, output, imports = {}) => {
  const source = await readFile(sourcePath, 'utf8');
  const compiled = transform(source, {
    filename: sourcePath, internalURL: import.meta.resolve('astro/compiler-runtime'),
    resultScopedSlot: true, resolvePath: specifier => specifier,
  });
  assert.deepEqual(compiled.diagnostics.filter(item => item.severity === 'error'), []);
  const code = compiled.code
    .replace(/^import "[^"\n]+\?astro&type=style[^"\n]*";\n/gm, '')
    .replace(/from "([^"\n]+)"/g, (_, name) => `from "${imports[name] ?? import.meta.resolve(name)}"`);
  await writeFile(output, code);
  return (await import(pathToFileURL(output).href)).default;
};
const tablePath = join(temporary, 'Table.mjs');
await compile(resolve('src/components/Table.astro'), tablePath);
const parameters = await compile(resolve('../experiments/exp124/report/ParameterTable.astro'), join(temporary, 'Parameters.mjs'), {
  '../../../site/src/components/Table.astro': pathToFileURL(tablePath).href,
});
const render = async data => (await experimental_AstroContainer.create()).renderToString(parameters, { props: { evidence: data } });
const html = await render(evidence);
assert.match(html, /100 ms/);  // The fixture's recorded duration, not the recipe's 1,000 ms.
assert.match(html, /0.05 \/ 0.025 µS/);
assert.match(html, /0.000005 \/ 0.00002 \/ 0.01/);
assert.match(html, /20 \/ — ms/);
assert.match(html, /equal-class cross-entropy/);
assert.match(html, /exp124-r001-compute/);
assert.match(await render(undefined), /Unavailable/);
const missing = structuredClone(evidence);
missing.snapshots.ping = undefined;
assert.match(await render(missing), /Unavailable/);
await writeFile(join(temporary, 'snapshot-parameters.html'), html);
"""
    checked = subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            "--input-type=module",
            "-",
            str(tmp_path / ".pingstore/runs"),
            presentation_id,
            str(tmp_path),
        ],
        input=script,
        cwd=site,
        text=True,
        capture_output=True,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    for model in cfg["models"]:
        assert output.file(f"{model}_sample_raster.svg").exists()
        assert (
            output.file(f"{model}_network.svg").read_bytes()
            == source.file(model, "circuit.svg").read_bytes()
        )
    assert {path.name for path in output.export.iterdir()} == {
        "numbers.json",
        "full_coba_training_curves.svg",
        "full_ping_training_curves.svg",
        "coba_sample_raster.svg",
        "ping_sample_raster.svg",
        "coba_network.svg",
        "ping_network.svg",
    }
    source.check_unchanged()


@pytest.mark.parametrize(
    "device,gpus,threads", [("cpu", 0, 4), ("cuda", 1, 4), ("cuda", 2, 1)]
)
def test_parallel_rejects_insufficient_resources_before_loading_data(
    monkeypatch, device, gpus, threads
):
    monkeypatch.setattr(compute, "resolve_device", lambda: device)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: gpus)
    monkeypatch.setenv("PINGLAB_CPU_THREADS", str(threads))

    def unexpected_load(*args):
        pytest.fail("invalid resources must fail before dataset acquisition")

    monkeypatch.setattr(dataset, "load_inputs", unexpected_load)
    with pytest.raises(ValueError, match="parallel compute requires"):
        compute.compute_parallel(protocol="local-check")


def test_execution_allocation_preserves_model_order_and_cpu_budget(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    cfg = dict(models=["ping", "coba"])
    items, threads = compute.execution_items(cfg, "cuda", 8, True)
    assert items == [
        {"id": "ping", "device": "cuda:0"},
        {"id": "coba", "device": "cuda:1"},
    ]
    assert threads == 4
    assert compute.execution_items(cfg, "cpu", 3, False) == (
        [{"id": "ping", "device": "cpu"}, {"id": "coba", "device": "cpu"}],
        3,
    )


def test_cli_selects_parallel_entrypoint_and_preserves_run_id(monkeypatch):
    calls = []
    monkeypatch.setattr(
        compute, "compute", lambda **kw: calls.append(("sequential", kw))
    )
    monkeypatch.setattr(
        compute, "compute_parallel", lambda **kw: calls.append(("parallel", kw))
    )
    compute.main([])
    compute.main(
        ["--parallel", "--run-id", "exp124-r001-compute", "--protocol", "local-check"]
    )
    assert calls == [
        ("sequential", dict(run_id=None, protocol="full")),
        ("parallel", dict(run_id="exp124-r001-compute", protocol="local-check")),
    ]


def failing_condition_worker(directory, item, index, *arguments):
    """Fail in a real child to exercise cancellation and the parent stage boundary."""
    raise RuntimeError("synthetic model failure")


def test_parallel_worker_failure_does_not_publish_partial_run(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "memberships", lambda repo: {"exp124": "datasets"})
    monkeypatch.setattr(
        stages, "_capture_code", lambda *args: {"git_commit": "synthetic-fixture"}
    )
    monkeypatch.setattr(compute, "condition_worker", failing_condition_worker)
    cfg = recipe.configuration("local-check")
    items = [{"id": model, "device": "cpu"} for model in cfg["models"]]
    with pytest.raises(RuntimeError, match="concurrent worker .* failed"):
        with stages.stage_run(tmp_path, "exp124", "compute", configuration=cfg) as run:
            snapshot = run.scratch / "dataset.npz"
            np.savez(snapshot, labels=np.array([0]))
            compute.run_conditions(run, cfg, snapshot, None, items, 1, True)
    runs = tmp_path / ".pingstore/runs"
    assert not list(runs.glob("exp124-*-compute"))
    assert not list(runs.glob(".*/export/recording.npz"))


def test_modal_dispatcher_accepts_parallel_cli():
    from modal_dispatcher.request import execution_environment, validate_request

    entrypoint = validate_request(
        Path(__file__).resolve().parents[2],
        "exp124",
        ["--parallel"],
        execution_environment("H100:2", 8),
    )
    assert entrypoint.name == "compute.py"
