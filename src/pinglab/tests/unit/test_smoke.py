"""End-to-end smoke tests — spawn oscilloscope.py subprocesses in each mode.

Marked `slow` because each test launches a fresh `uv run` process.
Run with: `uv run pytest -m slow`
"""
from __future__ import annotations

import subprocess

import pytest

pytestmark = pytest.mark.slow

OSC = "uv run python src/pinglab/oscilloscope.py"


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("smoke")


def _run(cmd, timeout=120):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, (
        f"cmd failed (exit {result.returncode}):\n  {cmd}\n"
        f"stderr: {result.stderr[-500:]}"
    )


# ── Oscilloscope modes ───────────────────────────────────────────────────

def test_sim_no_output(out_dir):
    _run(f"{OSC} sim --out-dir {out_dir}")


def test_image_output(out_dir):
    _run(f"{OSC} image --out-dir {out_dir}")


def test_video_output(out_dir):
    _run(f"{OSC} video --frames 3 --frame-rate 10 --out-dir {out_dir}")


def test_dataset_input_scikit(out_dir):
    _run(f"{OSC} image --input dataset --dataset scikit --out-dir {out_dir}")


@pytest.mark.parametrize("scan_var,lo,hi", [
    ("ei_strength", 0, 0.5),
    ("spike_rate", 5, 50),
    ("dt", 0.05, 0.5),
])
def test_scan(out_dir, scan_var, lo, hi):
    _run(f"{OSC} video --scan-var {scan_var} --scan-min {lo} --scan-max {hi} "
         f"--frames 3 --frame-rate 10 --out-dir {out_dir}")


def test_sim_snntorch_canonical(out_dir):
    _run(f"{OSC} sim --model snntorch-clone --out-dir {out_dir}")


# ── Training modes ───────────────────────────────────────────────────────

@pytest.mark.parametrize("dataset", ["scikit", "mnist", "smnist"])
def test_train_one_epoch(out_dir, dataset):
    _run(f"{OSC} train --epochs 1 --dataset {dataset} --n-hidden 64 "
         f"--max-samples 50 --out-dir {out_dir}/train-{dataset}")


def test_train_epochs_zero_probe(out_dir):
    _run(f"{OSC} train --epochs 0 --n-hidden 64 --out-dir {out_dir}/train-init")


def test_train_observe_video(out_dir):
    _run(f"{OSC} train --observe video --epochs 2 --n-hidden 64 "
         f"--max-samples 50 --out-dir {out_dir}/train-obs")


def test_train_ping(out_dir):
    _run(f"{OSC} train --epochs 2 --ei-strength 0.5 --n-hidden 64 "
         f"--max-samples 50 --cm-back-scale 1000 --out-dir {out_dir}/train-ping")
