"""Exercise deployment with real Git bundles and a local SSH stand-in."""

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from tools.hpc import __main__ as hpc


@pytest.fixture
def deployment(tmp_path):
    repo = tmp_path / "source"
    (repo / "experiments/exp999").mkdir(parents=True)
    (repo / "experiments/exp999/compute.py").write_text("print('fixture')\n")
    (repo / ".gitignore").write_text(".pingstore/\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    args = argparse.Namespace(
        experiment="exp999",
        stage="compute",
        host="csd3",
        remote_dir=str(tmp_path / "remote space's" / "deployment"),
        service="SL3",
        walltime="01:00:00",
        cpus=4,
        memory_gb=32,
        gpus=1,
        input=[],
        arguments=["--", "--label", "a value; $(false)"],
        live=False,
        receipt=tmp_path / "receipt.json",
    )
    return repo, args


def local_ssh(monkeypatch, args, *, ambiguous=False):
    actual_run = hpc.run
    observed = []

    def execute(command, **kwargs):
        if command[0] != "ssh":
            return actual_run(command, **kwargs)
        observed.append(command)
        receipt = json.loads(args.receipt.read_text())
        if command[2].startswith("sbatch "):
            assert receipt["status"] == "submitting"
            assert receipt["job_id"] is None
            return subprocess.CompletedProcess(
                command, 0, "unknown\n" if ambiguous else "12345;cluster\n", ""
            )
        assert receipt["status"] == "uploading"
        return actual_run(["bash", "-c", command[2]], **kwargs)

    monkeypatch.setattr(hpc, "run", execute)
    return observed


def test_preview_is_local_and_preserves_arguments(deployment, monkeypatch):
    repo, args = deployment
    actual_run = hpc.run

    def no_remote(command, **kwargs):
        assert command[0] == "git"
        return actual_run(command, **kwargs)

    monkeypatch.setattr(hpc, "run", no_remote)
    result = hpc.deploy(args, repo)
    assert result["arguments"] == args.arguments[1:]
    assert "--qos=gpu2" in result["command"]
    assert not args.receipt.exists()
    assert not Path(args.remote_dir).exists()


def test_live_bundle_preserves_clean_commit_and_receipt(deployment, monkeypatch):
    repo, args = deployment
    args.live = True
    calls = local_ssh(monkeypatch, args)
    result = hpc.deploy(args, repo)
    checkout = Path(args.remote_dir) / "repo"
    assert result["job_id"] == "12345"
    assert result["status"] == "submitted"
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        == result["commit"]
    )
    assert (
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=checkout, text=True
        )
        == ""
    )
    assert (checkout / "experiments/exp999/compute.py").read_bytes() == (
        repo / "experiments/exp999/compute.py"
    ).read_bytes()
    script = Path(args.remote_dir, "job.sbatch").read_text()
    assert "worker ../deployment.json" in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    import shlex

    command = shlex.split(calls[-1][2])
    assert f"--chdir={checkout}" in command
    assert command[-1] == str(Path(args.remote_dir, "job.sbatch"))
    count = len(calls)
    with pytest.raises(ValueError, match="already attempted"):
        hpc.deploy(args, repo)
    assert len(calls) == count
    monkeypatch.chdir(checkout)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    hpc.worker(Path(args.remote_dir) / "deployment.json")
    (checkout / "experiments/exp999/compute.py").write_text("changed")
    with pytest.raises(ValueError, match="uncommitted"):
        hpc.worker(Path(args.remote_dir) / "deployment.json")


def test_ambiguous_submission_keeps_receipt_and_blocks_retry(deployment, monkeypatch):
    repo, args = deployment
    args.live = True
    local_ssh(monkeypatch, args, ambiguous=True)
    with pytest.raises(ValueError, match="ambiguous"):
        hpc.deploy(args, repo)
    assert json.loads(args.receipt.read_text())["status"] == "submitting"
    with pytest.raises(ValueError, match="already attempted"):
        hpc.deploy(args, repo)


def test_existing_remote_directory_is_never_overwritten(deployment, monkeypatch):
    repo, args = deployment
    args.live = True
    remote = Path(args.remote_dir)
    remote.mkdir(parents=True)
    (remote / "keep").write_text("original")
    local_ssh(monkeypatch, args)
    with pytest.raises(subprocess.CalledProcessError):
        hpc.deploy(args, repo)
    assert (remote / "keep").read_text() == "original"
    assert json.loads(args.receipt.read_text())["status"] == "uploading"


def test_uncommitted_sources_block_live_before_receipt(deployment):
    repo, args = deployment
    args.live = True
    (repo / "new.py").write_text("uncommitted")
    with pytest.raises(ValueError, match="uncommitted"):
        hpc.deploy(args, repo)
    assert not args.receipt.exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("experiment", "../exp999"),
        ("remote_dir", "relative"),
        ("host", "-oProxyCommand=bad"),
        ("cpus", 0),
        ("gpus", -1),
        ("walltime", "01:99:00"),
    ],
)
def test_invalid_requests_fail_locally(deployment, key, value):
    repo, args = deployment
    setattr(args, key, value)
    with pytest.raises(ValueError):
        hpc.plan(args, repo)


def test_cpu_plan_uses_sl2_without_gpu_request(deployment):
    repo, args = deployment
    args.gpus = 0
    args.service = "SL2"
    result = hpc.plan(args, repo)
    assert "--qos=gpu1" in result["command"]
    assert not any(arg.startswith("--gres=") for arg in result["command"])
    assert "export SNNLAB_DEVICE=cpu" in hpc.job_script(result)


def test_worker_refuses_login_node(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    path.write_text("{}")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="allocation"):
        hpc.worker(path)


def test_live_transfers_validated_input_ancestry(deployment, monkeypatch):
    from pingstore import core

    repo, args = deployment
    monkeypatch.setattr(core, "memberships", lambda _: {"exp999": "fixture"})
    with core.stage_run(repo, "exp999", "compute") as compute:
        (compute.export / "result.json").write_text('{"value": 1}\n')
    source = core.source_run(repo / ".pingstore", compute.run_id)
    with core.stage_run(
        repo, "exp999", "analyse", inputs={"compute": source}
    ) as analyse:
        (analyse.export / "numbers.json").write_text('{"value": 2}\n')
    args.input = [analyse.run_id]
    args.live = True
    local_ssh(monkeypatch, args)
    hpc.deploy(args, repo)
    remote_store = Path(args.remote_dir) / "repo/.pingstore"
    for identity in (compute.run_id, analyse.run_id):
        original = core.source_run(repo / ".pingstore", identity)
        restored = core.source_run(remote_store, identity, reference=original.reference)
        assert restored.reference == original.reference
        assert (restored.directory / "run.json").read_bytes() == (
            original.directory / "run.json"
        ).read_bytes()


def test_corrupt_input_blocks_upload_before_receipt(deployment, monkeypatch):
    from pingstore import core
    from pingstore.library import PingstoreError

    repo, args = deployment
    monkeypatch.setattr(core, "memberships", lambda _: {"exp999": "fixture"})
    with core.stage_run(repo, "exp999", "compute") as compute:
        (compute.export / "result.json").write_text('{"value": 1}\n')
    (compute.directory / "export/result.json").write_text('{"value": 999}\n')
    args.input = [compute.run_id]
    args.live = True
    with pytest.raises(PingstoreError):
        hpc.deploy(args, repo)
    assert not args.receipt.exists()
    assert not Path(args.remote_dir).exists()
