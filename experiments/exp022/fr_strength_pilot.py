"""Bounded one-seed calibration for the normalized firing-rate penalty.

This is deliberately separate from the registered TR-02 bank: pilot outputs
must never satisfy, replace, or overwrite a production campaign cell.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

from experiments import exp022

STRENGTHS = (4e-3, 1.6e-2, 4.1e-2, 1e-1)
TARGET_HZ = 1.0
SEED = 42


def pilot_spec(model: str, strength: float) -> dict:
    if strength not in STRENGTHS:
        raise ValueError(f"unregistered pilot strength: {strength:g}")
    spec = copy.deepcopy(
        exp022.training_run_cell(
            "TR-02", model=model, rate_target_hz=TARGET_HZ, seed=SEED
        )
    )
    flag = spec["extra"].index("--fr-reg-upper-strength")
    spec["extra"][flag + 1] = str(strength)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=exp022.MODELS, required=True)
    parser.add_argument("--strength", type=float, choices=STRENGTHS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    spec = pilot_spec(args.model, args.strength)
    label = f"{args.strength:g}".replace(".", "p")
    out_dir = (args.output_root / f"{args.model}__lambda{label}__seed{SEED}").resolve()
    if out_dir.exists():
        raise SystemExit(f"pilot destination already exists: {out_dir}")
    samples, epochs = exp022.cell_samples_epochs(spec)
    train_args = exp022.build_train_args(spec, out_dir, samples, epochs)
    command = [sys.executable, str(exp022.SNN_TOOL), *train_args]
    print(json.dumps({
        "model": args.model,
        "strength": args.strength,
        "target_hz": TARGET_HZ,
        "seed": SEED,
        "max_samples": samples,
        "epochs": epochs,
        "output_directory": str(out_dir),
        "command": command,
    }, indent=2))
    subprocess.run(command, cwd=exp022.REPO, check=True)


if __name__ == "__main__":
    main()
