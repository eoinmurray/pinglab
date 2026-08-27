"""Relocation checks that never train, submit jobs or touch retained evidence."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO / "experiments" / "exp022"
SLURM = EXPERIMENT / "slurm"


@pytest.mark.parametrize("imports", [
    "from experiments.exp022 import campaign, compute",
    "from experiments.exp022 import compute, campaign",
    "from experiments.exp022 import tr06_diagnostic, fr_strength_pilot; "
    "from experiments.exp022 import campaign, compute",
    "import sys; sys.path.insert(0, 'experiments'); import exp022; "
    "from experiments.exp022 import campaign, compute; "
    "assert exp022.campaign is campaign; "
    "assert exp022.run_tr06_diagnostic is compute.run_tr06_diagnostic",
])
def test_relocated_imports_preserve_campaign_and_scheduler_hooks(imports):
    subprocess.run(
        [sys.executable, "-c", imports + "; assert compute.campaign is campaign"],
        cwd=REPO, check=True, capture_output=True, text=True,
    )


@pytest.mark.parametrize("entrypoint", [
    "compute.py", "analyse.py", "present.py", "tr06_diagnostic.py",
    "slurm/wilkes_diagnostic.py",
])
def test_file_entrypoints_resolve_from_an_external_directory(entrypoint, tmp_path):
    completed = subprocess.run(
        [sys.executable, str(EXPERIMENT / entrypoint), "--help"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    assert "usage:" in completed.stdout


def test_pilot_module_entrypoint():
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.exp022.fr_strength_pilot", "--help"],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    assert "--strength" in completed.stdout


def test_slurm_scripts_and_collection_references_resolve():
    scripts = sorted(SLURM.glob("*.sh")) + sorted(SLURM.glob("*.sbatch"))
    scripts.append(
        REPO / "experiments/collections/gamma_gated_sparsity/collection-job.sbatch"
    )
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
        for reference in re.findall(r"experiments/exp022/[\w./-]+", script.read_text()):
            target = REPO / reference
            assert target.is_file(), (script, reference)
    # This helper is executed directly; module initialization is only sourced.
    assert os.access(SLURM / "ensure-mnist-link.sh", os.X_OK)


def test_submit_wrapper_finds_repository_before_validation(tmp_path):
    cache = tmp_path / "cache"
    (cache / "MNIST").mkdir(parents=True)
    manifest = tmp_path / "campaign.json"
    manifest.write_text("{}")
    uv = tmp_path / "uv"
    uv.write_text(
        '#!/bin/bash\n'
        'printf "cwd=%s\\n" "$PWD"\n'
        'printf "arg=%s\\n" "$@"\n'
        'exit 23\n'
    )
    uv.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(SLURM / "submit-tier.sh"), str(manifest), "standard", "--dry-run"],
        cwd=tmp_path, capture_output=True, text=True,
        env={
            **os.environ,
            "EXP022_SLURM_ACCOUNT": "test-account",
            "EXP022_WALLTIME": "00:01:00",
            "EXP022_CONCURRENCY": "1",
            "EXP022_MNIST_CACHE": str(cache),
            "EXP022_UV": str(uv),
        },
    )
    # Stop at mocked validation: no scheduler or campaign mutation is involved.
    assert completed.returncode == 23, completed.stderr
    assert f"cwd={REPO}" in completed.stdout
    assert "arg=experiments/exp022/compute.py" in completed.stdout
    assert "arg=--campaign-validate" in completed.stdout
