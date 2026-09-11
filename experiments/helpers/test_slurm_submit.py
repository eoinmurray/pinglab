import subprocess

import pytest

from experiments.helpers.slurm_submit import submit_pipeline
from pingstore.contracts import PingstoreError, load_json


def test_receipt_precedes_submission_and_blocks_repeat(tmp_path):
    receipt = tmp_path / "submitted.json"
    observed = []

    def runner(command, **_kwargs):
        observed.append(load_json(receipt)["status"])
        return subprocess.CompletedProcess(command, 0, "12345\n", "")

    jobs = submit_pipeline(
        steps=("compute",),
        command_for=lambda _step, _dependency: ["sbatch", "worker.sbatch"],
        receipt=receipt,
        live=True,
        test_only=False,
        context={"experiment": "fixture"},
        runner=runner,
    )
    assert observed == ["submitting"]
    assert jobs == {"compute": "12345"}
    assert load_json(receipt)["status"] == "submitted"
    with pytest.raises(PingstoreError, match="already attempted"):
        submit_pipeline(
            steps=("compute",),
            command_for=lambda _step, _dependency: ["sbatch", "worker.sbatch"],
            receipt=receipt,
            live=True,
            test_only=False,
            runner=runner,
        )


def test_dry_run_does_not_write_or_contact_scheduler(tmp_path):
    receipt = tmp_path / "submitted.json"

    def fail(*_args, **_kwargs):
        raise AssertionError("dry run contacted scheduler")

    assert (
        submit_pipeline(
            steps=("compute", "collect"),
            command_for=lambda step, dependency: ["sbatch", step, str(dependency)],
            receipt=receipt,
            live=False,
            test_only=False,
            runner=fail,
        )
        == {}
    )
    assert not receipt.exists()
