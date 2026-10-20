"""Generic dispatch and a local subprocess round trip; no provider calls."""

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.modal_dispatcher import __main__ as cli
from tools.modal_dispatcher import jobs, request
from tools.modal_dispatcher.tests.test_jobs import Worker

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def experiment(tmp_path):
    repo = tmp_path / "repo"
    path = repo / "experiments" / "exp999" / "compute.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "import argparse\nparser = argparse.ArgumentParser()\n"
        'parser.add_argument("--run-id")\n'
        'parser.add_argument("--protocol")\n'
        'parser.add_argument("--source")\n'
    )
    return repo


def test_unlisted_experiment_freezes_command_resources_and_device(experiment):
    root = experiment / ".pingstore"
    worker = Worker(root)
    environment = request.execution_environment("none", 3)
    receipt = request.dispatch(
        experiment,
        "exp999",
        worker,
        object(),
        app_id="ap-fixture",
        arguments=["--protocol", "fixture"],
        inputs={},
        environment=environment,
        gpu="none",
        cpu=3,
        memory=2048,
        timeout=60,
    )
    assert worker.calls == 1
    assert receipt["plan"] == {
        "entrypoint": "experiments/exp999/compute.py",
        "arguments": ["--protocol", "fixture"],
        "environment": environment,
        "resources": dict(gpu=None, cpu=3, memory=2048, timeout=60),
    }
    assert environment["SNNLAB_DEVICE"] == "cpu"
    assert environment["OMP_NUM_THREADS"] == "3"
    assert request.execution_environment("A100-80GB:2", 4)["SNNLAB_DEVICE"] == "cuda"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--run-id=exp999-r010-compute"],
        ["--run", "other"],
        ["--r=other"],
        ["--help"],
        ["--hel"],
        ["--"],
        ["--epochs", "1"],
        [3],
    ],
)
def test_invalid_arguments_never_reserve_or_spawn(experiment, arguments):
    root = experiment / ".pingstore"
    worker = Worker(root)
    with pytest.raises(ValueError):
        request.dispatch(
            experiment,
            "exp999",
            worker,
            object(),
            app_id="ap-fixture",
            arguments=arguments,
            inputs={},
            environment={},
        )
    assert worker.calls == 0
    assert not root.exists()


@pytest.mark.parametrize("experiment_id", ["../exp999", "exp999/..", "exp998"])
def test_invalid_or_missing_entrypoint(experiment, experiment_id):
    with pytest.raises(ValueError):
        request.validate_request(experiment, experiment_id, [], {})


@pytest.mark.parametrize("options", [dict(cpu=0), dict(memory=0), dict(timeout=86401)])
def test_invalid_resources_fail_before_submission(experiment, options):
    with pytest.raises(ValueError):
        request.dispatch(
            experiment,
            "exp999",
            object(),
            object(),
            app_id="ap-fixture",
            arguments=[],
            inputs={},
            environment={},
            **options,
        )
    assert not (experiment / ".pingstore").exists()


def test_unpinned_source_argument_fails_before_provider(experiment):
    with pytest.raises(ValueError, match="explicit --input"):
        request.dispatch(
            experiment,
            "exp999",
            object(),
            object(),
            app_id="ap-fixture",
            arguments=["--source=exp998-r001-compute"],
            inputs={},
            environment={},
        )


def test_smoke_cannot_silently_launch_an_experiments_full_recipe(experiment):
    with pytest.raises(ValueError, match="SMOKE profile"):
        request.validate_request(
            experiment,
            "exp999",
            [],
            request.execution_environment("none", 1, smoke=True),
        )


def test_cli_forwards_arguments_as_json_without_shell(experiment, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "REPO", experiment)
    monkeypatch.setattr(
        cli.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )
    value = "fixture with spaces; $(no-shell)"
    cli.main(
        [
            "dispatch",
            "--experiment",
            "exp999",
            "--gpu",
            "none",
            "--",
            "--protocol",
            value,
        ]
    )
    command, options = calls[0]
    assert command[:5] == [sys.executable, "-m", "modal", "run", "--detach"]
    assert json.loads(command[command.index("--arguments") + 1]) == [
        "--protocol",
        value,
    ]
    assert options == {"cwd": experiment, "check": True}


def test_cli_check_does_not_import_training_app(monkeypatch, capsys):
    monkeypatch.setattr(jobs, "check", lambda *args, **kwargs: {"status": "fixture"})
    cli.main(["check", "--run-id", "exp999-r001-compute"])
    assert json.loads(capsys.readouterr().out)["status"] == "fixture"
    assert "tools.modal_dispatcher.app" not in sys.modules


def test_every_current_compute_entrypoint_supports_dispatch():
    for path in (REPO / "experiments").glob("exp[0-9][0-9][0-9]/compute.py"):
        request.validate_request(REPO, path.parent.name, [], {})


def test_remote_worker_subprocess_and_collector_round_trip(
    experiment, tmp_path, monkeypatch
):
    # The production image runs Python 3.12; local repo tests also support 3.10.
    if sys.version_info < (3, 11):
        import tomli

        monkeypatch.setitem(sys.modules, "tomllib", tomli)
    monkeypatch.setattr(sys, "path", list(sys.path))
    source = tmp_path / "source"
    remote = tmp_path / "remote"
    output = tmp_path / "output"
    for base in (source, remote):
        (base / "experiments").mkdir(parents=True)
        (base / "tools").mkdir()
    shutil.copytree(
        experiment / "experiments", source / "experiments", dirs_exist_ok=True
    )
    (source / "pyproject.toml").write_text("")
    (source / "uv.lock").write_text('[[package]]\nname="snnlab"\nversion="fixture"\n')
    # This subprocess only writes a tiny valid fixture, never imports SNNLab.
    path = source / "experiments" / "exp999" / "compute.py"
    path.write_text(
        path.read_text()
        + f"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, {str(REPO / "tools")!r})
from pingstore.contracts import payload_digest, write_json_atomic
args = parser.parse_args()
directory = Path(".pingstore/runs") / ("." + args.run_id + ".tmp")
reservation = json.loads((directory / ".reservation.json").read_text())
record = dict(reservation, scope="full", collection="fixture", inputs={{}},
              execution={{}}, provenance={{}}, created_at="2026-10-20T12:00:00+00:00")
(directory / "export" / "result.json").write_text(json.dumps(
    dict(protocol=args.protocol, device=os.environ["SNNLAB_DEVICE"])))
(directory / ".reservation.json").unlink()
record["payload_digest"] = payload_digest(directory)
write_json_atomic(directory / "run.json", record)
directory.rename(directory.with_name(args.run_id))
"""
    )

    class Volume:
        def reload(self):
            pass

        def commit(self):
            self.committed = True

        def read_file(self, name):
            yield (output / name).read_bytes()

    volume = Volume()
    environment = request.execution_environment("none", 1)
    arguments = ["--protocol", "fixture value"]
    receipt = request.dispatch(
        experiment,
        "exp999",
        Worker(experiment / ".pingstore"),
        volume,
        app_id="ap-fixture",
        arguments=arguments,
        inputs={},
        environment=environment,
        gpu="none",
        cpu=1,
    )
    _, reservation, _ = jobs.pending(experiment / ".pingstore", receipt["run_id"])
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(app_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute"
    )
    function.decorator_list = []
    namespace = dict(
        Path=Path,
        REMOTE=remote,
        SOURCE=source,
        OUTPUT=output,
        SOURCE_DIRS=("experiments", "tools"),
        shutil=shutil,
        os=os,
        sys=sys,
        subprocess=subprocess,
        results=volume,
        modal=SimpleNamespace(current_function_call_id=lambda: "fc-fixture"),
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(app_path), "exec"),
        namespace,
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "fixture")
    namespace["compute"](
        "exp999",
        reservation,
        receipt["dispatch"],
        arguments,
        environment,
        submission=receipt,
    )
    assert volume.committed
    result = jobs.check(
        experiment / ".pingstore",
        receipt["run_id"],
        collect=True,
        volume=volume,
        call_factory=lambda _: pytest.fail("must use volume marker"),
    )
    assert result["status"] == "collected"
    directory = experiment / ".pingstore" / "runs" / receipt["run_id"]
    assert json.loads((directory / "export" / "result.json").read_text()) == {
        "protocol": "fixture value",
        "device": "cpu",
    }
    assert (
        json.loads((directory / "run.json").read_text())["execution"]["modal"]
        == receipt
    )
