"""Compute the committed richer-input probe; never analyse or publish."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "tools")]

import subprocess

import numpy as np
from experiments.exp099 import recipe
from pingstore.contracts import write_json_atomic
from pingstore.stages import stage_run


def simulate(
    root: Path,
    bundle: Path,
    *,
    duration_ms: float = recipe.DURATION_MS,
    seed: int = recipe.SEED,
) -> list[str]:
    run = root / "simulation"
    command = [
        sys.executable,
        str(REPO / "tools/snnsim/tool.py"),
        "sim",
        "--bundle",
        str(bundle),
        "--t-ms",
        str(duration_ms),
        "--seed",
        str(seed),
        "--out-dir",
        str(run),
        "--wipe-dir",
    ]
    subprocess.run(command, cwd=REPO, check=True)
    export_initialized_weights(bundle, run, duration_ms=duration_ms, seed=seed)
    return command


def export_initialized_weights(
    bundle: Path,
    run: Path,
    *,
    duration_ms: float = recipe.DURATION_MS,
    seed: int = recipe.SEED,
) -> None:
    """Recreate and retain the exact seeded matrices used by SNNSIM."""
    # The simulator uses flat module imports; match those only during compute.
    sys.path.insert(0, str(REPO / "tools/snnsim"))
    import infer
    import models as snnsim_models  # noqa: TID251
    from bundle import load_graph_bundle, translate_cobanet_v1

    _, graph = load_graph_bundle(bundle)
    spec = translate_cobanet_v1(graph)
    infer._pin_run(spec.dt, duration_ms, seed=seed)
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


def compute(
    *,
    run_id: str | None = None,
    condition: str = "richer-input",
    shared_peak_scale: float = 6.5,
    private_afferent_scale: float = 1.0,
    background_rate_scale: float = 1.0,
    ampa_background_scale: float = 1.0,
    gaba_background_scale: float = 1.0,
    w_ee_scale: float = 1.0,
    w_ei_scale: float = 1.0,
    w_ie_scale: float = 1.0,
    w_in_e_scale: float = 1.0,
    w_in_i_scale: float = 1.0,
    tau_gaba_ms: float = 9.0,
    duration_ms: float = recipe.DURATION_MS,
    seed: int = recipe.SEED,
    onset_ms: float = recipe.ONSET_MS,
    peak_ms: float = recipe.PEAK_MS,
    plateau_end_ms: float = recipe.PEAK_MS,
    offset_ms: float = recipe.OFFSET_MS,
    view_start_ms: float = recipe.VIEW_START_MS,
    view_end_ms: float = recipe.VIEW_END_MS,
) -> str:
    bundle = recipe.author_network(
        condition=condition,
        shared_peak_scale=shared_peak_scale,
        private_afferent_scale=private_afferent_scale,
        background_rate_scale=background_rate_scale,
        ampa_background_scale=ampa_background_scale,
        gaba_background_scale=gaba_background_scale,
        w_ee_scale=w_ee_scale,
        w_ei_scale=w_ei_scale,
        w_ie_scale=w_ie_scale,
        w_in_e_scale=w_in_e_scale,
        w_in_i_scale=w_in_i_scale,
        tau_gaba_ms=tau_gaba_ms,
        onset_ms=onset_ms,
        peak_ms=peak_ms,
        plateau_end_ms=plateau_end_ms,
        offset_ms=offset_ms,
    )
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        run_id=run_id,
        configuration=recipe.configuration(
            bundle,
            condition=condition,
            shared_peak_scale=shared_peak_scale,
            private_afferent_scale=private_afferent_scale,
            background_rate_scale=background_rate_scale,
            ampa_background_scale=ampa_background_scale,
            gaba_background_scale=gaba_background_scale,
            w_ee_scale=w_ee_scale,
            w_ei_scale=w_ei_scale,
            w_ie_scale=w_ie_scale,
            w_in_e_scale=w_in_e_scale,
            w_in_i_scale=w_in_i_scale,
            tau_gaba_ms=tau_gaba_ms,
            duration_ms=duration_ms,
            seed=seed,
            onset_ms=onset_ms,
            peak_ms=peak_ms,
            plateau_end_ms=plateau_end_ms,
            offset_ms=offset_ms,
            view_start_ms=view_start_ms,
            view_end_ms=view_end_ms,
        ),
    ) as run:
        bundle_path = bundle.write(run.export / "network.bundle")
        command = simulate(run.export, bundle_path, duration_ms=duration_ms, seed=seed)
        run.record["execution"]["simulation_command"] = command
        write_json_atomic(run.scratch / "simulation-command.json", {"command": command})
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    parser.add_argument(
        "--condition",
        choices=("richer-input", "shared-drive-isolation"),
        default="richer-input",
    )
    parser.add_argument("--shared-peak-scale", type=float, default=6.5)
    parser.add_argument("--private-afferent-scale", type=float, default=1.0)
    parser.add_argument("--background-rate-scale", type=float, default=1.0)
    parser.add_argument("--ampa-background-scale", type=float, default=1.0)
    parser.add_argument("--gaba-background-scale", type=float, default=1.0)
    parser.add_argument("--w-ee-scale", type=float, default=1.0)
    parser.add_argument("--w-ei-scale", type=float, default=1.0)
    parser.add_argument("--w-ie-scale", type=float, default=1.0)
    parser.add_argument("--w-in-e-scale", type=float, default=1.0)
    parser.add_argument("--w-in-i-scale", type=float, default=1.0)
    parser.add_argument("--tau-gaba-ms", type=float, default=9.0)
    parser.add_argument("--duration-ms", type=float, default=recipe.DURATION_MS)
    parser.add_argument("--seed", type=int, default=recipe.SEED)
    parser.add_argument("--onset-ms", type=float, default=recipe.ONSET_MS)
    parser.add_argument("--peak-ms", type=float, default=recipe.PEAK_MS)
    parser.add_argument("--plateau-end-ms", type=float, default=recipe.PEAK_MS)
    parser.add_argument("--offset-ms", type=float, default=recipe.OFFSET_MS)
    parser.add_argument("--view-start-ms", type=float, default=recipe.VIEW_START_MS)
    parser.add_argument("--view-end-ms", type=float, default=recipe.VIEW_END_MS)
    args = parser.parse_args()
    compute(
        run_id=args.run_id,
        condition=args.condition,
        shared_peak_scale=args.shared_peak_scale,
        private_afferent_scale=args.private_afferent_scale,
        background_rate_scale=args.background_rate_scale,
        ampa_background_scale=args.ampa_background_scale,
        gaba_background_scale=args.gaba_background_scale,
        w_ee_scale=args.w_ee_scale,
        w_ei_scale=args.w_ei_scale,
        w_ie_scale=args.w_ie_scale,
        w_in_e_scale=args.w_in_e_scale,
        w_in_i_scale=args.w_in_i_scale,
        tau_gaba_ms=args.tau_gaba_ms,
        duration_ms=args.duration_ms,
        seed=args.seed,
        onset_ms=args.onset_ms,
        peak_ms=args.peak_ms,
        plateau_end_ms=args.plateau_end_ms,
        offset_ms=args.offset_ms,
        view_start_ms=args.view_start_ms,
        view_end_ms=args.view_end_ms,
    )


if __name__ == "__main__":
    main()
