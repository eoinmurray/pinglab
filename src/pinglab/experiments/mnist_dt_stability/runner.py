"""Goal 1: dt-stability on MNIST — typer subcommand app.

Subcommands:
    train          Train one model at one dt
    calibrate      Train the full 7-model ladder × 2 dts at a size preset
    verify         Fresh test-set eval; halt if any model < threshold
    sweep          dt-sweep on all trained checkpoints (parallel Modal T4)
    paper          Regenerate figures + compile paper.typ
    experiment     Full pipeline: calibrate → verify → sweep → paper
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer

from pinglab.experiments.mnist_dt_stability.config import (
    MODELS,
    SIZES,
    HEADLINE_LADDER,
    TRAINING_DTS,
    SWEEP_DTS,
    COMMON,
    CALIBS_ROOT,
    FIGURES_ROOT,
    Size,
    suffix,
    calib_dir,
)

app = typer.Typer(no_args_is_help=True, help="Goal 1: dt-stability on MNIST")


# ── Helpers ──────────────────────────────────────────────────────────────

OSC = "src/pinglab/oscilloscope.py"


def _run(args: list[str], check: bool = True) -> int:
    """Run a subprocess streaming to stdout. Returns exit code."""
    print(f"$ {' '.join(str(a) for a in args)}", flush=True)
    result = subprocess.run([str(a) for a in args])
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(str(a) for a in args)}")
    return result.returncode


def _resolve_size(size: str | None, samples: int | None, epochs: int | None) -> Size:
    """Resolve a size preset name or explicit (samples, epochs) pair."""
    if samples is not None and epochs is not None:
        return Size(samples=samples, epochs=epochs)
    if size is None:
        size = "standard"
    if size not in SIZES:
        raise typer.BadParameter(
            f"unknown size {size!r}; choose from {list(SIZES)} or pass --samples and --epochs"
        )
    return SIZES[size]


# ── Training ─────────────────────────────────────────────────────────────

@app.command()
def train(
    model: str = typer.Argument(..., help="Model name from the ladder"),
    dt: float = typer.Option(0.1, help="Training Δt in ms"),
    samples: int = typer.Option(1000, help="Training set size"),
    epochs: int = typer.Option(40, help="Number of epochs"),
    modal: bool = typer.Option(True, help="Dispatch to Modal"),
    gpu: str = typer.Option("H100", help="Modal GPU type"),
):
    """Train one model at one dt."""
    if model not in MODELS:
        raise typer.BadParameter(f"unknown model {model!r}; choose from {list(MODELS)}")
    cfg = MODELS[model]
    size = Size(samples=samples, epochs=epochs)
    out_dir = calib_dir(model, dt, size)

    args = ["uv", "run", "python", OSC, "train", *cfg.args,
            "--dataset", COMMON["dataset"],
            "--dt", dt,
            "--t-ms", COMMON["t_ms"],
            "--input-rate", COMMON["input_rate"],
            "--max-samples", samples,
            "--epochs", epochs,
            "--adaptive-lr",
            "--early-stopping", COMMON["early_stopping"],
            "--observe", "video",
            "--out-dir", str(out_dir), "--wipe-dir"]
    if modal:
        args += ["--modal", "--modal-gpu", gpu]
    _run(args)


@app.command()
def calibrate(
    size: str = typer.Argument("standard", help="Size preset: smoke|quick|standard|large|full"),
    samples: int = typer.Option(None, help="Override samples (with --epochs)"),
    epochs: int = typer.Option(None, help="Override epochs (with --samples)"),
    modal: bool = typer.Option(True, help="Dispatch to Modal"),
    gpu: str = typer.Option("H100", help="Modal GPU type"),
    parallel: bool = typer.Option(True, help="Launch all 14 runs in parallel (Modal)"),
):
    """Train the full ladder (7 models × 2 training dts = 14 runs)."""
    sz = _resolve_size(size, samples, epochs)
    print(f"calibrate: {len(MODELS)} models × {len(TRAINING_DTS)} dts "
          f"= {len(MODELS) * len(TRAINING_DTS)} runs @ {sz.samples} samples × {sz.epochs} epochs")

    if modal and parallel:
        # Launch all N runs as background processes; each dispatches its own
        # Modal app with detach=True so they survive the parent exit.
        procs = []
        for m in MODELS:
            for dt in TRAINING_DTS:
                cfg = MODELS[m]
                out_dir = calib_dir(m, dt, sz)
                args = ["uv", "run", "python", OSC, "train", *cfg.args,
                        "--dataset", COMMON["dataset"],
                        "--dt", str(dt),
                        "--t-ms", str(COMMON["t_ms"]),
                        "--input-rate", str(COMMON["input_rate"]),
                        "--max-samples", str(sz.samples),
                        "--epochs", str(sz.epochs),
                        "--adaptive-lr",
                        "--early-stopping", str(COMMON["early_stopping"]),
                        "--observe", "video",
                        "--out-dir", str(out_dir), "--wipe-dir",
                        "--modal", "--modal-gpu", gpu]
                log_path = Path(f"/tmp/pinglab-{m}-dt{dt}.log")
                log = open(log_path, "w")
                print(f"  launching {m} dt={dt} → {log_path}")
                p = subprocess.Popen([str(a) for a in args], stdout=log, stderr=log)
                procs.append((p, log))
        print(f"  waiting for {len(procs)} parallel runs to complete...")
        failed = []
        for p, log in procs:
            rc = p.wait()
            log.close()
            if rc != 0:
                failed.append(p.args)
        if failed:
            raise RuntimeError(f"{len(failed)} training runs failed; see /tmp/pinglab-*.log")
    else:
        for m in MODELS:
            for dt in TRAINING_DTS:
                train(model=m, dt=dt, samples=sz.samples, epochs=sz.epochs,
                      modal=modal, gpu=gpu)


# ── Verification ─────────────────────────────────────────────────────────

@app.command()
def verify(
    size: str = typer.Option("standard"),
    samples: int = typer.Option(None),
    epochs: int = typer.Option(None),
    threshold: float = typer.Option(80.0, help="Minimum best_acc (%) to pass"),
):
    """Fresh test-set eval on all 14 calibrations; halt if any below threshold."""
    from pinglab.experiments.mnist_dt_stability.analysis import verify as verify_mod
    sz = _resolve_size(size, samples, epochs)
    verify_mod.run(threshold=threshold, size=sz)


# ── dt-sweep ─────────────────────────────────────────────────────────────

@app.command()
def sweep(
    size: str = typer.Option("standard"),
    samples: int = typer.Option(None),
    epochs: int = typer.Option(None),
    modal: bool = typer.Option(True, help="Run sweeps on Modal T4 in parallel"),
):
    """dt-sweep on all trained checkpoints (frozen inputs, 14-point Δt in [0.05, 2.0] ms)."""
    sz = _resolve_size(size, samples, epochs)
    dts = " ".join(str(d) for d in SWEEP_DTS)

    if modal:
        procs = []
        for dt in TRAINING_DTS:
            for m in MODELS:
                D = calib_dir(m, dt, sz)
                args = ["uv", "run", "python", OSC, "infer",
                        "--from-dir", str(D),
                        "--dt-sweep", *[str(x) for x in SWEEP_DTS],
                        "--frozen-inputs", "--observe", "video",
                        "--out-dir", str(D / "infer-frozen"),
                        "--modal", "--modal-gpu", "T4"]
                log_path = Path(f"/tmp/pinglab-sweep-{m}-dt{dt}.log")
                log = open(log_path, "w")
                print(f"  launching sweep {m} dt={dt} → {log_path}")
                p = subprocess.Popen([str(a) for a in args], stdout=log, stderr=log)
                procs.append((p, log))
        print(f"  waiting for {len(procs)} sweeps to complete...")
        failed = []
        for p, log in procs:
            rc = p.wait()
            log.close()
            if rc != 0:
                failed.append(p.args)
        if failed:
            raise RuntimeError(f"{len(failed)} sweeps failed; see /tmp/pinglab-sweep-*.log")
    else:
        # Sequential, local (MPS/CPU)
        for dt in TRAINING_DTS:
            for m in MODELS:
                D = calib_dir(m, dt, sz)
                _run(["uv", "run", "python", OSC, "infer",
                      "--from-dir", str(D),
                      "--dt-sweep", *SWEEP_DTS,
                      "--frozen-inputs", "--observe", "video",
                      "--out-dir", str(D / "infer-frozen")])


# ── Figures ──────────────────────────────────────────────────────────────

@app.command()
def figures(
    size: str = typer.Option("standard"),
    samples: int = typer.Option(None),
    epochs: int = typer.Option(None),
):
    """Regenerate the 4 paper figures as PNGs in the figures/ artifact dir."""
    from pinglab.experiments.mnist_dt_stability.analysis import figures as figs
    sz = _resolve_size(size, samples, epochs)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    print("fig 1 — calibration accuracy")
    figs.calibration_accuracy(size=sz)
    print("fig 2 — dt-sweep combined")
    figs.dt_sweep(size=sz)
    print("fig 3 — training curves")
    figs.training_curves(size=sz)
    print("fig 4 — ablation attribution")
    figs.ablation_attribution(size=sz)
    print("fig 5 — snntorch-library parity (if data present)")
    try:
        figs.parity_sweep(size=sz, train_dt=1.0)
    except Exception as e:
        print(f"  skipped: {e}")


# ── Full pipeline ────────────────────────────────────────────────────────

@app.command()
def experiment(
    size: str = typer.Argument("standard"),
    samples: int = typer.Option(None),
    epochs: int = typer.Option(None),
    gpu: str = typer.Option("H100"),
):
    """Full data pipeline: calibrate → verify → sweep → regenerate figures."""
    sz = _resolve_size(size, samples, epochs)
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║ Goal 1 experiment @ size={size}")
    print(f"║ {sz.samples} samples × {sz.epochs} epochs, GPU={gpu}")
    print(f"╚══════════════════════════════════════════════╝")
    calibrate(size=size, samples=samples, epochs=epochs, modal=True, gpu=gpu, parallel=True)
    verify(size=size, samples=samples, epochs=epochs, threshold=80.0)
    sweep(size=size, samples=samples, epochs=epochs, modal=True)
    figures(size=size, samples=samples, epochs=epochs)
    print("")
    print("✓ data pipeline complete — figures at src/artifacts/mnist-dt-stability/figures/")
