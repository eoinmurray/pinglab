"""Goal 2: PING's advantage on sequential MNIST — typer subcommand app.

Subcommands:
    train      Train one model at one configuration
    calibrate  Train all ladder models at a size preset
    ladder     Staged calibration: init-probe → learn-probe → short → medium → full
    verify     Fresh test-set sanity check
    paper      (placeholder) regenerate figures + compile paper

The distinctive thing here vs Goal 1 is the ladder subcommand — it runs each
stage with pass/fail criteria and halts on failure, so we can tune hyperparameters
at a stage without wasting compute on later stages.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer

from pinglab.experiments.ping_smnist_advantage.config import (
    COMMON,
    CALIBS_ROOT,
    FIGURES_ROOT,
    LADDER,
    MODELS,
    SIZES,
    STAGES,
    Size,
    Stage,
    calib_dir,
    suffix,
)

app = typer.Typer(no_args_is_help=True,
                  help="Goal 2: does PING help on sequential MNIST?")

OSC = "src/pinglab/oscilloscope.py"


def _run(args: list, check: bool = True) -> int:
    print(f"$ {' '.join(str(a) for a in args)}", flush=True)
    result = subprocess.run([str(a) for a in args])
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode})")
    return result.returncode


def _resolve_size(size: str | None, samples: int | None, epochs: int | None) -> Size:
    if samples is not None and epochs is not None:
        return Size(samples=samples, epochs=epochs)
    if size is None:
        size = "quick"
    if size not in SIZES:
        raise typer.BadParameter(
            f"unknown size {size!r}; choose from {list(SIZES)} or pass --samples and --epochs"
        )
    return SIZES[size]


def _eval_pass(metrics: dict, criteria: dict) -> tuple[bool, list[str]]:
    """Apply pass criteria dict to a metrics dict. Returns (passed, reasons)."""
    reasons = []
    # Final-state end metrics are nested under "end" (rate_e, act, cv, f0)
    # plus top-level best_acc. For epochs=0 probes, "end" may be None — fall
    # back to "init" state which also has the dynamics keys.
    end = metrics.get("end") or metrics.get("init") or {}
    flat = {**end, "best_acc": metrics.get("best_acc", 0)}
    for key, bounds in criteria.items():
        val = flat.get(key)
        if val is None:
            reasons.append(f"{key}: missing")
            continue
        if "min" in bounds and val < bounds["min"]:
            reasons.append(f"{key}={val:.2f} < {bounds['min']} (min)")
        if "max" in bounds and val > bounds["max"]:
            reasons.append(f"{key}={val:.2f} > {bounds['max']} (max)")
    return (not reasons, reasons)


# ── Subcommands ─────────────────────────────────────────────────────────

@app.command()
def train(
    model: str = typer.Argument(..., help="Model name from the ladder"),
    samples: int = typer.Option(500, help="Training set size"),
    epochs: int = typer.Option(15, help="Number of epochs"),
    dt: float = typer.Option(1.0, help="Training Δt in ms"),
    modal: bool = typer.Option(False, help="Dispatch to Modal"),
    gpu: str = typer.Option("H100", help="Modal GPU type"),
    observe: bool = typer.Option(False, help="Record video during training"),
):
    """Train one sMNIST model."""
    if model not in MODELS:
        raise typer.BadParameter(f"unknown model {model!r}")
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
            "--out-dir", str(out_dir), "--wipe-dir"]
    if observe:
        args += ["--observe", "video"]
    if modal:
        args += ["--modal", "--modal-gpu", gpu]
    _run(args)


@app.command()
def calibrate(
    size: str = typer.Argument("quick"),
    samples: int = typer.Option(None),
    epochs: int = typer.Option(None),
    dt: float = typer.Option(1.0),
    modal: bool = typer.Option(False),
    gpu: str = typer.Option("H100"),
):
    """Train all ladder models at the given size (single dt)."""
    sz = _resolve_size(size, samples, epochs)
    print(f"calibrate: {len(LADDER)} models @ {sz.samples} × {sz.epochs}, dt={dt}")
    for m in LADDER:
        train(m, samples=sz.samples, epochs=sz.epochs, dt=dt,
              modal=modal, gpu=gpu, observe=False)


@app.command()
def ladder(
    only: str = typer.Option(None, help="Run only this model (debug)"),
    start_from: str = typer.Option(None, help="Skip stages before this name"),
    modal: bool = typer.Option(False, help="Dispatch to Modal"),
    gpu: str = typer.Option("H100"),
):
    """Staged calibration ladder: init-probe → learn-probe → short → medium → full.

    Each stage has pass/fail criteria. If any model fails a stage, the ladder
    halts with a report of what failed — fix hyperparameters in config.py
    and re-run (resume from the failing stage via --start-from).
    """
    models = [only] if only else LADDER
    skip = bool(start_from)

    report = {}
    for stage in STAGES:
        if skip and stage.name != start_from:
            continue
        skip = False
        print()
        print(f"══ stage: {stage.name} ({stage.size.samples} × {stage.size.epochs})")
        stage_report = {}
        for m in models:
            print(f"\n── {m}")
            train(m, samples=stage.size.samples, epochs=stage.size.epochs,
                  dt=1.0, modal=modal, gpu=gpu, observe=False)
            metrics_path = calib_dir(m, 1.0, stage.size) / "metrics.json"
            if not metrics_path.exists():
                print(f"  ✗ no metrics.json at {metrics_path}")
                stage_report[m] = (False, ["no metrics"])
                continue
            metrics = json.loads(metrics_path.read_text())
            passed, reasons = _eval_pass(metrics, stage.pass_criteria)
            if passed:
                acc = metrics.get("best_acc", 0)
                print(f"  ✓ PASSED (best_acc={acc:.1f}%)")
            else:
                print(f"  ✗ FAILED: {'; '.join(reasons)}")
            stage_report[m] = (passed, reasons)

        report[stage.name] = stage_report
        failures = [m for m, (ok, _) in stage_report.items() if not ok]
        if failures:
            print()
            print("══ ladder halted")
            print(f"  stage {stage.name} failures: {failures}")
            print(f"  fix hyperparameters in config.py, rerun with --start-from {stage.name}")
            raise typer.Exit(code=1)
        print(f"  ✓ all models passed stage {stage.name}")

    print()
    print("══ ladder complete")
    for stage_name, stage_report in report.items():
        for m, (ok, reasons) in stage_report.items():
            mark = "✓" if ok else "✗"
            print(f"  {mark} {stage_name:<14} {m}")


@app.command()
def figures():
    """(Placeholder) Regenerate figures into the figures/ artifact dir."""
    print("Goal 2 figures not yet implemented. Once ladder results exist, populate")
    print("pinglab/experiments/ping_smnist_advantage/analysis/figures.py with")
    print("generator functions and wire them here.")


@app.command()
def experiment(modal: bool = True, gpu: str = "H100"):
    """Full data pipeline for Goal 2: ladder → figures (when ready)."""
    ladder(only=None, start_from=None, modal=modal, gpu=gpu)
    figures()
