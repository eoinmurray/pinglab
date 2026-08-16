"""Bounded comparison for the TR-06 variable-rate spike-count readout.

This is a diagnostic, not a production training entry.  It compares the
registered TR-06 recipe with a fan-in-normalized spike-count initializer and a
matched variable-rate ``mem-mean`` control.  Every artifact stays below the
required ``--out-dir``.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments import exp022
from experiments.helpers import modal_backend

VARIANTS = (
    "registered-spike-count",
    "fanin-spike-count",
    "mean-005-spike-count",
    "mean-010-spike-count",
    "mem-mean-control",
)


def _replace_value(args: list[str], flag: str, value: str) -> None:
    args[args.index(flag) + 1] = value


def _remove_pair(args: list[str], flag: str) -> None:
    index = args.index(flag)
    del args[index : index + 2]


def diagnostic_args(
    variant: str,
    *,
    output: Path,
    max_samples: int,
    epochs: int,
    seed: int,
    device: str,
    n_hidden: int | None = None,
    t_ms: float | None = None,
    dt_ms: float | None = None,
) -> list[str]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown TR-06 diagnostic variant: {variant}")
    cell = copy.deepcopy(exp022.training_run_cell("TR-06", seed=seed))
    # The production registry deliberately pins 7,000 samples.  This bounded
    # diagnostic declares and applies its smaller scale explicitly.
    cell.pop("max_samples", None)
    args = exp022.build_train_args(cell, output, max_samples, epochs)
    args += ["--device", device]
    if n_hidden is not None:
        _replace_value(args, "--n-hidden", str(n_hidden))
    if t_ms is not None:
        _replace_value(args, "--t-ms", str(t_ms))
    if dt_ms is not None:
        _replace_value(args, "--dt", str(dt_ms))
    if variant == "fanin-spike-count":
        _remove_pair(args, "--readout-w-init-mean")
        _remove_pair(args, "--readout-w-init-std")
    elif variant in {"mean-005-spike-count", "mean-010-spike-count"}:
        mean = 0.05 if variant == "mean-005-spike-count" else 0.10
        std = 0.04 if variant == "mean-005-spike-count" else 0.08
        _replace_value(args, "--readout-w-init-mean", str(mean))
        _replace_value(args, "--readout-w-init-std", str(std))
    elif variant == "mem-mean-control":
        _replace_value(args, "--readout", "mem-mean")
    return args


def run_variant(
    variant: str,
    *,
    root: Path,
    max_samples: int,
    epochs: int,
    seed: int,
    device: str,
    n_hidden: int | None = None,
    t_ms: float | None = None,
    dt_ms: float | None = None,
) -> dict[str, object]:
    directory = root / variant
    command = diagnostic_args(
        variant,
        output=directory,
        max_samples=max_samples,
        epochs=epochs,
        seed=seed,
        device=device,
        n_hidden=n_hidden,
        t_ms=t_ms,
        dt_ms=dt_ms,
    )
    subprocess.run([sys.executable, str(exp022.SNN_TOOL), *command], check=True)
    result = summarize(directory, variant)
    (directory / "diagnostic_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def summarize(directory: Path, variant: str) -> dict[str, object]:
    metrics = json.loads((directory / "metrics.json").read_text())
    epochs = metrics["epochs"]
    return {
        "variant": variant,
        "directory": str(directory),
        "best_validation_accuracy_pct": metrics["best_acc"],
        "best_validation_loss": metrics["best_validation_loss"],
        "best_epoch": metrics["best_epoch"],
        "epochs": [
            {
                key: row.get(key)
                for key in (
                    "ep",
                    "acc",
                    "test_loss",
                    "test_rate_e",
                    "test_rate_i",
                    "test_margin",
                    "test_logit_scale",
                    "test_output_spikes_per_sample",
                    "test_output_silent_fraction",
                    "test_output_class_spike_fraction",
                    "test_output_by_input_rate",
                    "gnorm__W_ff.0",
                    "gnorm__W_ff.1",
                )
            }
            for row in epochs
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=700)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-hidden", type=int)
    parser.add_argument("--t-ms", type=float)
    parser.add_argument("--dt-ms", type=float)
    parser.add_argument("--only-variant", action="append", choices=VARIANTS)
    parser.add_argument("--modal", action="store_true")
    parser.add_argument("--live", action="store_true")
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    root = options.out_dir.resolve()
    variants = options.only_variant or list(VARIANTS)
    if options.modal:
        modal_backend.dispatch(
            slug="exp022-tr06-diagnostic",
            runner="exp022",
            job_ids=variants,
            live=options.live,
            local_collect_dir=root,
            ledger_path=root / "compute_ledger.json",
            timeout_s=14_400,
            extra_env={
                "EXP022_TR06_DIAGNOSTIC_MAX_SAMPLES": str(options.max_samples),
                "EXP022_TR06_DIAGNOSTIC_EPOCHS": str(options.epochs),
                "EXP022_TR06_DIAGNOSTIC_SEED": str(options.seed),
                **(
                    {"EXP022_TR06_DIAGNOSTIC_N_HIDDEN": str(options.n_hidden)}
                    if options.n_hidden
                    else {}
                ),
                **(
                    {"EXP022_TR06_DIAGNOSTIC_T_MS": str(options.t_ms)}
                    if options.t_ms
                    else {}
                ),
                **(
                    {"EXP022_TR06_DIAGNOSTIC_DT_MS": str(options.dt_ms)}
                    if options.dt_ms
                    else {}
                ),
            },
            is_done_name="tr06_diagnostic_done",
            run_job_name="run_tr06_diagnostic",
        )
        return
    if options.live:
        raise SystemExit("--live requires --modal")
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for variant in variants:
        results.append(
            run_variant(
                variant,
                root=root,
                max_samples=options.max_samples,
                epochs=options.epochs,
                seed=options.seed,
                device=options.device,
                n_hidden=options.n_hidden,
                t_ms=options.t_ms,
                dt_ms=options.dt_ms,
            )
        )
    payload = {
        "purpose": "TR-06 bounded readout diagnosis; not production evidence",
        "seed": options.seed,
        "max_samples": options.max_samples,
        "epochs": options.epochs,
        "n_hidden": options.n_hidden or exp022.N_EXCITATORY,
        "t_ms": options.t_ms or exp022.T_MS,
        "dt_ms": options.dt_ms or exp022.DT_MS,
        "variants": results,
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(root / "summary.json")


if __name__ == "__main__":
    main()
