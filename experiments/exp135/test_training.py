"""Subset/encoding contracts and a bounded native-training pipeline integration."""

import io
import json
import subprocess
import tarfile
import threading

import numpy as np
import pytest
from experiments.exp135 import analyse, compute, dataset, present, recipe
from experiments.helpers.network_artifacts import snapshot_parameters
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.layout import initialize_layout
from pingstore.stages import source_run
from snnlab import sim


def test_ten_percent_selection_is_balanced_reproducible_and_subject_disjoint():
    rows = [
        dict(sample_id=f"sample-{i:04d}", subject=i // 47 + 1, label=i % 11)
        for i in range(1077)
    ]
    selected, held_out = dataset.select(rows)
    assert selected == dataset.select(list(reversed(rows)))[0]
    assert len(selected["train"]) == 108
    counts = np.bincount([r["label"] for r in selected["train"]], minlength=11)
    assert sorted(counts) == [9] * 2 + [10] * 9
    assert len(selected["validation"]) == 11
    assert {r["subject"] for r in selected["train"]}.isdisjoint(held_out)
    assert {r["subject"] for r in selected["validation"]} <= set(held_out)
    assert not (
        {r["sample_id"] for r in selected["train"]}
        & {r["sample_id"] for r in selected["validation"]}
    )


def test_encoding_preserves_polarity_full_extent_and_counts_collisions():
    events = np.array(
        [[4, 8, 0, 100], [7, 11, 0, 100], [4, 8, 1, 100], [127, 127, 1, 6100]]
    )
    result = dataset.encode(events)
    assert result["duration_ms"] == 6001
    assert result["timestamp_origin_ms"] == 100
    assert result["collisions"] == 1
    assert result["binary_events"] == 3
    assert result["event_time_ms"].tolist() == [0, 0, 6000]
    assert result["event_channel"].tolist() == [65, 1089, 2047]
    with pytest.raises(ValueError, match="out of bounds"):
        dataset.encode(np.array([[128, 0, 0, 0]]))


@pytest.fixture
def fixture_training(tmp_path, monkeypatch):
    """Synthetic archive fixture exercises real APIs, never the public dataset."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    report = repo / "experiments/exp135/report/report.mdx"
    report.parent.mkdir(parents=True)
    report.write_text(
        "---\nid: exp135\ncollection: uncategorized\ninputs: [exp135]\n---\n"
    )
    monkeypatch.setattr(recipe, "CROP_MS", 8.0)
    monkeypatch.setattr(recipe, "COMPILE_MODE", None)
    monkeypatch.setattr(recipe, "N_CLASSES", 2)
    monkeypatch.setattr(recipe, "EXPECTED_TRAIN_SAMPLES", 8)
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", repo)
    archive = tmp_path / "fixture.tar.gz"
    step, polarity, y, x = np.indices((8, 2, 32, 32))
    events = np.stack(
        [x.ravel() * 4, y.ravel() * 4, polarity.ravel(), step.ravel()], axis=1
    )
    with tarfile.open(archive, "w:gz") as handle:
        for subject in range(1, 5):
            for label in range(2):
                buffer = io.BytesIO()
                np.save(buffer, events)
                member = tarfile.TarInfo(
                    f"ibmGestureTrain/user{subject:02d}_fixture/{label}.npy"
                )
                member.size = buffer.tell()
                buffer.seek(0)
                handle.addfile(member, buffer)
    monkeypatch.setattr(recipe, "TRAIN_ARCHIVE_MD5", dataset.checksum(archive, "md5"))
    (repo / ".gitignore").write_text(".pingstore/\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture source",
        ],
        check=True,
    )
    return repo, archive


def test_native_training_and_all_stages_with_fixture_archive(
    fixture_training, tmp_path
):
    repo, archive = fixture_training
    identity = compute.compute(archive=archive, epochs=2, batch_size=1)
    trained = source_run(repo / ".pingstore", identity, stage="compute")
    starts = {}
    for model in recipe.MODELS:
        start = snapshot_parameters(trained, model, "training", "start")
        end = snapshot_parameters(trained, model, "training", "end")
        starts[model] = start
        for key in recipe.COMMON_WEIGHTS:
            assert np.isfinite(end[key]).all()
            assert not np.array_equal(start[key], end[key]), key
        restored = trained.restore_bundle(
            f"{model}/final.checkpoint", tmp_path / f"{model}-checkpoint"
        )
        checkpoint = sim.load_training_checkpoint(restored)
        assert checkpoint.completed_updates == 4
        assert checkpoint.best_checkpoint is not None
        sample = np.load(trained.file(model, "spikes.npz"), allow_pickle=False)
        assert set(recipe.POPULATIONS[model]) <= set(sample.files)
    for name in recipe.COMMON_WEIGHTS:
        assert np.array_equal(starts["coba"][name], starts["ping"][name])
    analysis_id = analyse.analyse(identity)
    presentation_id = present.present(analysis_id)
    presented = source_run(repo / ".pingstore", presentation_id, stage="present")
    numbers = json.loads(presented.file("numbers.json").read_text())
    assert numbers["paired_initialization"] is True
    table = {str(row[0]): row[1:] for row in numbers["parameter_table"]}
    assert table["Readout"][0].startswith("2 non-spiking")
    assert table["Training / validation examples"][0] == "2 / 2"
    assert table["Simulation timestep / presentation"][0] == "1 ms / 8 ms"
    for model in recipe.MODELS:
        assert presented.file(f"{model}-training-curves.svg").stat().st_size > 100
        assert presented.file(f"{model}-sample-raster.svg").stat().st_size > 100
        assert presented.file(f"{model}-network-circuit.svg").stat().st_size > 100
        assert numbers["models"][model]["n_train"] == 2
        assert numbers["models"][model]["official_test_used"] is False
        assert numbers["models"][model]["any_weights_changed"] is True
        assert len(numbers["models"][model]["epochs"]) == 3


def test_two_shards_complete_one_run_and_reject_changed_payload(
    fixture_training, tmp_path
):
    from experiments.helpers import modal_jobs
    from experiments.helpers.test_modal_jobs import Volume, Worker, no_call

    repo, archive = fixture_training
    local_root = tmp_path / "download/.pingstore"
    receipt = modal_jobs.submit(
        local_root,
        recipe.SLUG,
        Worker(local_root),
        options={"gpu": "A100-80GB:2"},
        arguments={},
        app_id="fixture",
    )
    identity = receipt["run_id"]
    reservation = load_json(local_root / "runs" / f".{identity}.tmp/.reservation.json")
    remote = repo / ".pingstore/runs" / f".{identity}.tmp"
    initialize_layout(remote, recipe.SLUG)
    write_json_atomic(remote / ".reservation.json", reservation)
    compute.prepare_parallel(
        run_id=identity, archive=archive, cfg=recipe.configuration(epochs=2)
    )
    directory = compute.working(identity)
    with compute.concurrent.compute_lock(directory, exclusive=False):
        with pytest.raises(PingstoreError, match="busy"):
            compute.collect(identity)
    first = compute.run_shard(identity, 0, device="cpu")
    assert compute.run_shard(identity, 0, device="cpu") == first
    with pytest.raises((PingstoreError, FileNotFoundError)):
        compute.collect(identity)
    assert not (directory / "run.json").exists()
    history = directory / "export/coba/training-history.json"
    original = history.read_bytes()
    history.write_text("corrupted")
    with pytest.raises(PingstoreError, match="payload"):
        compute.run_shard(identity, 0, device="cpu")
    history.write_bytes(original)
    compute.run_shard(identity, 1, device="cpu")
    compute.collect(identity)
    completed = source_run(repo / ".pingstore", identity, stage="compute")
    assert len(completed.record["execution"]["shards"]) == 2
    assert [s["work_items"] for s in completed.record["execution"]["shards"]] == [
        ["coba"],
        ["ping"],
    ]
    assert not (completed.directory / ".scratch").exists()
    assert completed.file("coba", "spikes.npz").exists()
    assert completed.file("ping", "spikes.npz").exists()
    files = {
        f"{receipt['remote_path']}/{p.relative_to(completed.directory)}": p.read_bytes()
        for p in completed.directory.rglob("*")
        if p.is_file()
    }
    names = [
        p.relative_to(completed.directory).as_posix()
        for p in completed.directory.rglob("*")
        if p.is_file()
    ]
    files[f"{receipt['dispatch']}/completion.json"] = json.dumps(
        dict(
            run_id=identity,
            dispatch=receipt["dispatch"],
            files=names,
        )
    ).encode()
    downloaded = modal_jobs.check(
        local_root, identity, collect=True, volume=Volume(files), call_factory=no_call
    )
    assert downloaded["status"] == "collected"
    local = source_run(local_root, identity)
    assert local.reference == completed.reference
    assert {
        p.name for p in (local_root / "runs").iterdir() if not p.name.startswith(".")
    } == {identity}
    for model in recipe.MODELS:
        assert (
            local.file(model, "spikes.npz").read_bytes()
            == completed.file(model, "spikes.npz").read_bytes()
        )


def test_plan_rejects_changed_shared_data_and_dirty_source(fixture_training):
    repo, archive = fixture_training
    identity = compute.prepare_parallel(
        archive=archive, cfg=recipe.configuration(epochs=2)
    )
    directory = compute.working(identity)
    snapshot = directory / "export/train-events.npz"
    snapshot.write_bytes(b"changed")
    with pytest.raises(PingstoreError, match="dataset"):
        compute.run_shard(identity, 0, device="cpu")
    (repo / "changed-source.py").write_text("changed = True\n")
    with pytest.raises(PingstoreError, match="clean"):
        compute.prepare_parallel(archive=archive)


def test_gpu_workers_start_before_wait_and_get_distinct_devices(monkeypatch):
    monkeypatch.setattr(compute.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-A,GPU-B")
    spawned = []

    class Child:
        returncode = None

        def wait(self, timeout=None):
            assert len(spawned) == 2
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    def spawn(command, **kwargs):
        child = Child()
        spawned.append((command, kwargs, child))
        return child

    compute.launch_workers("exp135-r001-compute", popen=spawn)
    assert [r[1]["env"]["CUDA_VISIBLE_DEVICES"] for r in spawned] == ["GPU-A", "GPU-B"]
    assert [r[0][r[0].index("--shard-index") + 1] for r in spawned] == ["0", "1"]


def test_failed_gpu_worker_terminates_sibling(monkeypatch):
    monkeypatch.setattr(compute.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    stopped = threading.Event()
    children = []

    class Child:
        returncode = None

        def __init__(self, index):
            self.index = index

        def wait(self, timeout=None):
            if self.index == 0:
                self.returncode = 1
                return 1
            assert stopped.wait(2), "sibling was left running after failure"
            return -15

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15
            stopped.set()

    def spawn(*args, **kwargs):
        child = Child(len(children))
        children.append(child)
        return child

    with pytest.raises(RuntimeError, match="incomplete"):
        compute.launch_workers("exp135-r001-compute", popen=spawn)
    assert children[1].returncode == -15


def test_frozen_dispatch_recipe_mismatch_fails_before_dataset_preparation(
    fixture_training,
):
    repo, archive = fixture_training
    identity = compute.reserve_stage(repo / ".pingstore", recipe.SLUG, "compute")
    directory = compute.working(identity)
    write_json_atomic(
        directory / ".scratch/dispatch-plan.json",
        dict(
            schema="exp135.modal-plan/v1",
            source=compute.clean_source(repo),
            configuration={"wrong": "recipe"},
            resources={"gpu": "A100-80GB:2"},
        ),
    )
    with pytest.raises(PingstoreError, match="frozen Modal"):
        compute.prepare_parallel(run_id=identity, archive=archive)
    assert not list((directory / "export").iterdir())


def test_missing_archive_and_bad_checksum_fail_without_downloading(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset.verified_archive(tmp_path / "absent.tar.gz")
    bad = tmp_path / "bad.tar.gz"
    bad.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum"):
        dataset.verified_archive(bad)


def test_plan_has_no_dataset_or_run_side_effects(capsys):
    compute.main(["--plan"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["epochs"] == 20
    assert plan["train_fraction"] == 0.10


def test_training_plots_use_paired_scales_percentages_and_class_chance(
    tmp_path, monkeypatch
):
    import matplotlib.pyplot as plt

    report = {
        "configuration": {"epochs": 2, "n_classes": 11},
        "models": {
            model: {
                "n_train": 54,
                "n_validation": 11,
                "epochs": [
                    {"epoch": 0, "validation_loss": 3.0, "validation_accuracy": 0.0},
                    {
                        "epoch": 1,
                        "train_loss": 2.0,
                        "validation_loss": 2.5,
                        "train_accuracy": 0.25,
                        "validation_accuracy": 0.2,
                    },
                    {
                        "epoch": 2,
                        "train_loss": final_loss,
                        "validation_loss": 2.2,
                        "train_accuracy": 0.5,
                        "validation_accuracy": 0.3,
                    },
                ],
            }
            for model, final_loss in (("coba", 1.0), ("ping", 0.5))
        },
    }
    figures = []
    monkeypatch.setattr(present, "save", lambda figure, *_: figures.append(figure))
    try:
        for model in recipe.MODELS:
            present.training_curves(report, model, tmp_path)
        for figure in figures:
            loss, accuracy = figure.axes
            assert loss.get_title().startswith("A ")
            assert accuracy.get_title().startswith("B ")
            assert accuracy.get_ylabel() == "Accuracy (%)"
            assert accuracy.lines[0].get_ydata().tolist() == [25, 50]
            assert accuracy.lines[0].get_xdata().tolist() == [1, 2]
            assert accuracy.lines[1].get_xdata().tolist() == [0, 1, 2]
            assert list(accuracy.lines[2].get_ydata()) == pytest.approx([100 / 11] * 2)
        assert figures[0].axes[0].get_ylim() == figures[1].axes[0].get_ylim()
        assert figures[0].axes[1].get_ylim() == figures[1].axes[1].get_ylim()
    finally:
        for figure in figures:
            plt.close(figure)


@pytest.mark.parametrize(
    "failure",
    ["empty202", "html", "range", "empty200", "nongzip", "truncated", "checksum"],
)
def test_download_rejects_invalid_responses_then_verifies_retry(
    tmp_path, monkeypatch, failure
):
    import gzip
    import hashlib

    payload = gzip.compress(b"verified fixture archive")

    class Response(io.BytesIO):
        def __init__(self, body, *, status=200, headers=None):
            super().__init__(body)
            self.status = status
            self.headers = (
                headers
                if headers is not None
                else {
                    "Content-Type": "application/gzip",
                    "Content-Length": str(len(body)),
                }
            )

    responses = {
        "empty202": lambda: Response(b"", status=202),
        "html": lambda: Response(
            b"<html>challenge</html>", headers={"Content-Type": "text/html"}
        ),
        "range": lambda: Response(
            payload[:8], status=206, headers={"Content-Range": "bytes 0-7/200"}
        ),
        "empty200": lambda: Response(b""),
        "nongzip": lambda: Response(b"not an archive"),
        "truncated": lambda: Response(
            payload, headers={"Content-Length": str(len(payload) + 1)}
        ),
        "checksum": lambda: Response(gzip.compress(b"wrong dataset")),
    }
    opened = []

    def fetch(request, *, timeout):
        opened.append(request)
        assert timeout == 120
        return responses[failure]() if len(opened) == 1 else Response(payload)

    monkeypatch.setattr(dataset.urllib.request, "urlopen", fetch)
    monkeypatch.setattr(dataset.time, "sleep", lambda _: None)
    monkeypatch.setattr(recipe, "TRAIN_ARCHIVE_MD5", hashlib.md5(payload).hexdigest())
    target = tmp_path / "archive.tar.gz"
    assert dataset.verified_archive(target, download=True) == target
    assert target.read_bytes() == payload
    assert not target.with_suffix(".gz.part").exists()
    assert len(opened) == 2


def test_download_fails_closed_after_bounded_retries(tmp_path, monkeypatch):
    calls = []

    class Response(io.BytesIO):
        status = 202
        headers = {"Content-Type": "text/html", "Content-Length": "0"}

    def fetch(*args, **kwargs):
        calls.append(args)
        return Response(b"")

    monkeypatch.setattr(dataset.urllib.request, "urlopen", fetch)
    monkeypatch.setattr(dataset.time, "sleep", lambda _: None)
    target = tmp_path / "archive.tar.gz"
    with pytest.raises(RuntimeError, match="three attempts") as caught:
        dataset.verified_archive(target, download=True)
    assert "HTTP 200" in str(caught.value.__cause__)
    assert len(calls) == 3
    assert not target.exists()
    assert not target.with_suffix(".gz.part").exists()


def test_fractional_millisecond_timestamps_are_preserved_before_binning():
    events = np.array(
        [
            [4, 8, 0, 100.9],
            [7, 11, 0, 101.0],
            [4, 8, 1, 101.9],
            [127, 127, 1, 103.15],
        ]
    )
    encoded = dataset.encode(events)
    assert encoded["timestamp_origin_ms"] == 100.9
    assert encoded["event_time_ms"].tolist() == [0, 1, 2]
    assert encoded["event_channel"].tolist() == [65, 1089, 2047]
    assert encoded["collisions"] == 1
    assert encoded["duration_ms"] == 3
    with pytest.raises(ValueError, match="coordinates/polarities must be integers"):
        dataset.encode(np.array([[4.5, 8, 0, 0.002]]))
    with pytest.raises(ValueError, match="finite"):
        dataset.encode(np.array([[4, 8, 0, np.nan]]))


def test_local_check_preparation_is_explicit_and_preserves_source_extent(
    fixture_training,
):
    repo, archive = fixture_training
    check = recipe.local_check_configuration()
    check["local_check"]["presentation_ms"] = 3.0
    encoded = dataset.prepare(
        archive, repo / "prepared", local_check=check["local_check"]
    )
    assert len(encoded["splits"]["train"]) == 2
    assert len(encoded["splits"]["validation"]) == 2
    assert encoded["presentation_ms"] == 3
    assert encoded["splits"]["train"][0]["source_event_extent_ms"] == 7
    assert encoded["splits"]["train"][0]["prefix_limit_ms"] == 3
    with np.load(repo / "prepared/train-events.npz", allow_pickle=False) as arrays:
        assert arrays["event_time_ms"].max() == 2


def test_full_gesture_local_plan_does_not_request_a_prefix(capsys):
    compute.main(["--phase", "local-check-full", "--plan"])
    cfg = json.loads(capsys.readouterr().out)
    assert cfg["epochs"] == 1
    assert cfg["local_check"]["presentation_ms"] is None


def test_full_gesture_local_preparation_keeps_complete_extent(fixture_training):
    repo, archive = fixture_training
    check = recipe.local_check_configuration(full_gestures=True)
    encoded = dataset.prepare(
        archive, repo / "full-prepared", local_check=check["local_check"]
    )
    assert len(encoded["splits"]["train"]) == 2
    assert encoded["presentation_ms"] == 8
    with np.load(repo / "full-prepared/train-events.npz", allow_pickle=False) as arrays:
        assert arrays["event_time_ms"].max() == 7


def test_random_crops_replay_and_validation_is_centred():
    # A unique channel per source timestep makes crop origins observable.
    times = np.arange(1000, dtype=np.float64)
    arrays = dict(
        event_time_ms=times,
        event_channel=np.arange(1000),
        event_offsets=np.array([0, 1000]),
        durations_ms=np.array([1000]),
    )
    config = dict(duration_ms=500.0, random=True)
    first = dataset.encode_crops(
        arrays, [0], dt_ms=1.0, channels=2048, config=config, seed=42
    )
    replay = dataset.encode_crops(
        arrays, [0], dt_ms=1.0, channels=2048, config=config, seed=42
    )
    other = dataset.encode_crops(
        arrays, [0], dt_ms=1.0, channels=2048, config=config, seed=43
    )
    assert first.equal(replay)
    assert not first.equal(other)
    assert first.shape == (500, 1, 2048)
    assert first.sum() == 500
    fixed = dataset.encode_crops(
        arrays,
        [0],
        dt_ms=1.0,
        channels=2048,
        config=dict(duration_ms=500.0, random=False),
        seed=1,
    )
    assert fixed[0, 0, 250] == 1
    assert fixed[-1, 0, 749] == 1


def test_native_provider_refreshes_crops_between_epochs(tmp_path):
    from snnlab.sim.inputs.provider import DatasetProvider

    path = tmp_path / "events.npz"
    np.savez_compressed(
        path,
        labels=np.array([0]),
        event_time_ms=np.arange(1000, dtype=float),
        event_channel=np.arange(1000),
        event_offsets=np.array([0, 1000]),
        durations_ms=np.array([1000]),
    )
    bundle = recipe.author_network("coba", presentation_ms=500.0)
    binding = sim.DatasetSnapshotBinding(
        path,
        "events",
        "fixture",
        "train",
        sim.DatasetEncoder(
            "custom",
            definition="exp135.random_crop/v1",
            seed=42,
            config=dict(duration_ms=500.0, random=True),
        ),
        target_id="gesture",
    )
    a = DatasetProvider(bundle.graph, binding, execution_seed=42)
    ping = recipe.author_network("ping", presentation_ms=500.0)
    b = DatasetProvider(ping.graph, binding, execution_seed=42)
    one = a.batch([0], phase="train", epoch=0).tensors["events"]
    two = a.batch([0], phase="train", epoch=1).tensors["events"]
    paired = b.batch([0], phase="train", epoch=0).tensors["events"]
    assert one.equal(paired)
    assert not one.equal(two)
