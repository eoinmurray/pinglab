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

from experiments import exp022

VARIANTS = (
    "registered-spike-count",
    "fanin-spike-count",
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
) -> list[str]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown TR-06 diagnostic variant: {variant}")
    cell = copy.deepcopy(exp022.training_run_cell("TR-06", seed=seed))
    # The production registry deliberately pins 7,000 samples.  This bounded
    # diagnostic declares and applies its smaller scale explicitly.
    cell.pop("max_samples", None)
    args = exp022.build_train_args(cell, output, max_samples, epochs)
    args += ["--device", device]
    if variant == "fanin-spike-count":
        _remove_pair(args, "--readout-w-init-mean")
        _remove_pair(args, "--readout-w-init-std")
    elif variant == "mem-mean-control":
        _replace_value(args, "--readout", "mem-mean")
    return args


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
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    root = options.out_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for variant in VARIANTS:
        directory = root / variant
        command = diagnostic_args(
            variant,
            output=directory,
            max_samples=options.max_samples,
            epochs=options.epochs,
            seed=options.seed,
            device=options.device,
        )
        subprocess.run([sys.executable, str(exp022.SNN_TOOL), *command], check=True)
        results.append(summarize(directory, variant))
    payload = {
        "purpose": "TR-06 bounded readout diagnosis; not production evidence",
        "seed": options.seed,
        "max_samples": options.max_samples,
        "epochs": options.epochs,
        "variants": results,
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(root / "summary.json")


if __name__ == "__main__":
    main()
