"""Compute the committed richer-input probe; never analyse or publish."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "tools")]

import shutil
import subprocess

import numpy as np
from experiments.exp099 import recipe
from pingstore.contracts import write_json_atomic
from pingstore.stages import stage_run


def simulate(root: Path, bundle_spec) -> list[str]:
    bundle = root / "network.bundle"
    run = root / "simulation"
    bundle_spec.write(bundle, visualise=True)
    command = [
        sys.executable,
        str(REPO / "tools/snnsim/tool.py"),
        "sim",
        "--bundle",
        str(bundle),
        "--t-ms",
        str(recipe.DURATION_MS),
        "--seed",
        str(recipe.SEED),
        "--out-dir",
        str(run),
        "--wipe-dir",
    ]
    subprocess.run(command, cwd=REPO, check=True)
    export_initialized_weights(bundle, run)
    shutil.copy2(bundle / "reports/expanded.svg", root / "network.svg")
    return command


def export_initialized_weights(bundle: Path, run: Path) -> None:
    """Recreate and retain the exact seeded matrices used by SNNSIM."""
    # The simulator uses flat module imports; match those only during compute.
    sys.path.insert(0, str(REPO / "tools/snnsim"))
    import infer
    import models as snnsim_models  # noqa: TID251
    from bundle import load_graph_bundle, translate_cobanet_v1

    _, graph = load_graph_bundle(bundle)
    spec = translate_cobanet_v1(graph)
    infer._pin_run(spec.dt, recipe.DURATION_MS, seed=recipe.SEED)
    snnsim_models.N_IN = spec.input_size
    snnsim_models.N_INH = spec.hidden_size // 4
    snnsim_models.EXACT_K_INITIALIZATION = spec.exact_k_initialization
    network = infer.build_net(
        "ping",
        w_in=spec.w_in,
        w_in_i=spec.w_in_i,
        w_ee=spec.w_ee,
        w_ei=spec.w_ei,
        w_ie=spec.w_ie,
        w_ii=spec.w_ii,
        w_in_initial_zero_fraction=0.0,
        recurrent_initial_zero_fraction=spec.recurrent_initial_zero_fraction,
        device=infer._auto_device(),
        randomize_init=True,
        dales_law=True,
        hidden_sizes=[spec.hidden_size],
        readout_mode=spec.readout_mode,
        signed_readout=False,
        readout_bias=False,
        n_inh_per_layer=None,
        train_leak=False,
        adaptive_threshold=False,
    )
    np.savez_compressed(
        run / "recurrent-weights.npz",
        w_in_e=network.W_ff[0].detach().cpu().numpy(),
        w_in_i=network.W_in_i.detach().cpu().numpy(),
        w_ee=network.W_ee["1"].detach().cpu().numpy(),
        w_ei=network.W_ei["1"].detach().cpu().numpy(),
        w_ie=network.W_ie["1"].detach().cpu().numpy(),
        w_ii=network.W_ii["1"].detach().cpu().numpy(),
    )


def compute(*, run_id: str | None = None) -> str:
    bundle = recipe.author_network()
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        run_id=run_id,
        configuration=recipe.configuration(bundle),
    ) as run:
        command = simulate(run.export, bundle)
        run.record["execution"]["simulation_command"] = command
        write_json_atomic(
            run.evidence / "simulation-command.json", {"command": command}
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="unused v3 identity reserved before dispatch")
    args = parser.parse_args()
    compute(run_id=args.run_id)


if __name__ == "__main__":
    main()
