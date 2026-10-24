"""Execute the three-window AI-target to PING-target protocol in an independent Pingstore run.

Run compute.py with optional --seed, --device, --compile-mode, --n-e, --n-i
and --run-id. Every displayed exp148 condition belongs to this one run.
The native graph, start snapshot, sparse spike events, input checksum and stored-weight
verification are retained. Analysis and presentation run separately.
"""

import argparse
import hashlib
import os
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import torch
from experiments.exp148 import dataset, recipe
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares
from scipy.special import erfcx
from snnlab.sim import (
    DenseArrayBinding,
    Performance,
    RecordingSpec,
    SignalRecording,
    SnapshotSpec,
    simulate,
)


def execute(cfg, directory, *, diagram=False):
    """Run one condition with native snapshots and immutable-weight verification."""
    directory.mkdir(parents=True, exist_ok=True)
    bundle = recipe.author_network(cfg)
    bundle.write(directory / "network.bundle", visualise=False)
    if diagram:
        bundle.visualise(directory.parent / "circuit.svg", view="circuit")
    external = dataset.encode_drive(cfg).to(cfg["device"])
    events = {population: [] for population in ("E", "I")}
    traces = {}
    state_signals = (
        [f"{p}.voltage" for p in ("E", "I")]
        + [f"{p}.conductance" for p in cfg["initializer_parent_means_us"]]
        if cfg.get("record_state")
        else []
    )

    def collect_spikes(block):
        if block.kind == "dense":
            # Native per-cell conductance/voltage, reduced after recording; no feedback.
            traces.setdefault(block.signal, []).append(
                block.values.mean(dim=(1, 2)).cpu().numpy()
            )
        # Native sparse rows are absolute step, batch index, and cell index.
        if block.kind == "spike_events":
            events[block.signal.split(".")[0]].append(block.values.numpy())

    result = simulate(
        bundle=bundle,
        input_bindings=(DenseArrayBinding("X", external),),
        seed=cfg["seed"],
        device=cfg["device"],
        performance=Performance(**cfg["performance"]),
        diagnostics=False,
        recording=RecordingSpec(
            signals=tuple(
                SignalRecording(f"{p}.spikes", kind="spike_events") for p in ("E", "I")
            )
            + tuple(SignalRecording(signal) for signal in state_signals),
            sink=collect_spikes,
            sink_id="exp148.sparse-spikes/v1",
            block_steps=cfg["recording_block_steps"],
        ),
        snapshot=SnapshotSpec(path=directory / "start", stage="start", parameters=True),
        retain_outputs=False,
        retain_parameters=True,
        retain_final_state=False,
        retain_runtime_state=False,
    )
    arrays = {}
    for population, blocks in events.items():
        rows = np.concatenate(blocks)
        arrays[population + "_time_ms"] = rows[:, 0] * cfg["dt_ms"]
        arrays[population + "_cell"] = rows[:, 2]
    np.savez_compressed(directory / "events.npz", **arrays)
    if state_signals:
        np.savez_compressed(
            directory / "state-traces.npz",
            time_ms=np.arange(len(external)) * cfg["dt_ms"],
            input_spikes=external.sum(dim=(1, 2)).cpu().numpy(),
            **{name: np.concatenate(blocks) for name, blocks in traces.items()},
        )
    initial = result.snapshots["start"].parameters
    for name, value in result.parameters.items():
        torch.testing.assert_close(value.cpu(), initial[name], rtol=0, atol=0)
    audit = dict(
        native_modulation=True,
        stored_weights_unchanged=True,
        external_float32_sha256=hashlib.sha256(
            memoryview(external.cpu().numpy())
        ).hexdigest(),
        spike_time_convention="zero-based absolute native recording step multiplied by dt_ms",
    )
    write_json_atomic(directory / "verification.json", audit)
    result.write_metrics(directory / "metrics.json")


def compute(
    *,
    seed=42,
    device="cpu",
    compile_mode="off",
    n_e=4000,
    n_i=1000,
    run_id=None,
    record_state=False,
    mean_field_source=None,
):
    if mean_field_source is not None:
        return compute_mean_field(mean_field_source, run_id=run_id)
    cfg = recipe.configuration(
        seed=seed,
        device=os.environ.get("SNNLAB_DEVICE", device),
        compile_mode=compile_mode,
        n_e=n_e,
        n_i=n_i,
        record_state=record_state,
    )
    torch.set_num_threads(cfg["torch_threads"])
    with stage_run(REPO, "exp148", "compute", configuration=cfg, run_id=run_id) as run:
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={"simulation": ["start"]}
        )
        run.record["execution"]["snnlab_version"] = version("snnlab")
        print("Starting simulation", flush=True)
        execute(cfg, run.export / "simulation", diagram=True)
        print("Completed simulation", flush=True)
    print(run.run_id, flush=True)
    return run.run_id


def mean_field_current_map(cfg):
    """Map the recurrent conductances in µS to currents in nA."""
    net = cfg["spiking_configuration"]
    weights = cfg["summed_weights_us"]
    tau = net["tau_synapse_ms"]
    excitatory_force = net["excitatory_reversal_mv"] - net["resting_mv"]
    inhibitory_force = net["resting_mv"] - net["inhibitory_reversal_mv"]
    recurrent = (
        tau
        / 1000
        * np.array(
            [
                [
                    excitatory_force * weights["E_to_E"],
                    -inhibitory_force * weights["I_to_E"],
                ],
                [
                    excitatory_force * weights["E_to_I"],
                    -inhibitory_force * weights["I_to_I"],
                ],
            ]
        )
    )
    external = (
        tau
        / 1000
        * excitatory_force
        * net["external_rate_hz"]
        * np.array([weights["X_to_E"], weights["X_to_I"]])
    )
    return recurrent / np.asarray(cfg["conductance_per_hz"]), external


def mean_field_gain(current_na, cfg, quadrature):
    """Siegert noisy-LIF gain and its derivative, in Hz and Hz/nA.

    Fixed Gaussian voltage noise and fixed resting driving forces follow exp117.
    Gauss-Legendre quadrature integrates erfcx(-u), avoiding exp(u²) cancellation.
    """
    net = cfg["spiking_configuration"]
    sigma = cfg["voltage_noise_mv"]
    leak = net["leak_us"]
    tau_m = net["capacitance_nf"] / leak
    mean_voltage = net["resting_mv"] + np.asarray(current_na) / leak
    lower = (net["reset_mv"] - mean_voltage) / sigma
    upper = (net["threshold_mv"] - mean_voltage) / sigma
    nodes, weights = quadrature
    rates, derivatives = [], []
    for index, (a, b) in enumerate(zip(lower, upper)):
        # Beyond this bound both gain and derivative are negligible in float64.
        if b > 26:
            rates.append(0.0)
            derivatives.append(0.0)
            continue
        points = (a + b) / 2 + (b - a) / 2 * nodes
        integral = (b - a) / 2 * np.dot(weights, erfcx(-points))
        refractory = net["refractory_e_ms" if index == 0 else "refractory_i_ms"]
        rate_per_ms = 1 / (refractory + tau_m * np.sqrt(np.pi) * integral)
        derivative = (
            tau_m
            * np.sqrt(np.pi)
            * rate_per_ms**2
            * (erfcx(-b) - erfcx(-a))
            / (leak * sigma)
        )
        rates.append(1000 * rate_per_ms)
        derivatives.append(1000 * derivative)
    return np.array(rates), np.array(derivatives)


def mean_field_flow(state, alpha, cfg, quadrature):
    recurrent, external = mean_field_current_map(cfg)
    target, _ = mean_field_gain(
        recurrent @ state[2:] + external * [1, alpha], cfg, quadrature
    )
    return np.concatenate(
        [
            (target - state[:2]) / cfg["rate_relaxation_ms"],
            (state[:2] * cfg["conductance_per_hz"] - state[2:])
            / cfg["spiking_configuration"]["tau_synapse_ms"],
        ]
    )


def mean_field_jacobian(state, alpha, cfg, quadrature):
    recurrent, external = mean_field_current_map(cfg)
    _, slope = mean_field_gain(
        recurrent @ state[2:] + external * [1, alpha], cfg, quadrature
    )
    relaxation = np.asarray(cfg["rate_relaxation_ms"])
    tau = cfg["spiking_configuration"]["tau_synapse_ms"]
    return np.block(
        [
            [-np.diag(1 / relaxation), np.diag(slope / relaxation) @ recurrent],
            [np.diag(cfg["conductance_per_hz"]) / tau, -np.eye(2) / tau],
        ]
    )


def mean_field_equilibrium(alpha, guess, cfg, quadrature):
    recurrent, external = mean_field_current_map(cfg)

    def residual(rates):
        predicted, _ = mean_field_gain(
            recurrent @ (rates * cfg["conductance_per_hz"]) + external * [1, alpha],
            cfg,
            quadrature,
        )
        return rates - predicted

    def jacobian(rates):
        _, slope = mean_field_gain(
            recurrent @ (rates * cfg["conductance_per_hz"]) + external * [1, alpha],
            cfg,
            quadrature,
        )
        return np.eye(2) - np.diag(slope) @ recurrent @ np.diag(
            cfg["conductance_per_hz"]
        )

    net = cfg["spiking_configuration"]
    maximum = 1000 / np.array([net["refractory_e_ms"], net["refractory_i_ms"]])
    for initial in (guess, [2, 6], [20, 40], [100, 150]):
        solution = least_squares(
            residual,
            initial,
            jac=jacobian,
            bounds=(0, maximum),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=1000,
        )
        if np.max(np.abs(residual(solution.x))) < cfg["equilibrium_tolerance_hz"]:
            return np.concatenate([solution.x, solution.x * cfg["conductance_per_hz"]])
    raise RuntimeError(f"No validated mean-field equilibrium at alpha={alpha}")


def compute_mean_field(source, *, run_id=None):
    """Run the independent closure against explicit, unchanged spiking evidence."""
    spiking = source_run(
        REPO / ".pingstore", source, experiment="exp148", stage="compute"
    )
    net = spiking.record["execution"]["configuration"]
    snapshot = load_json(spiking.file("simulation/start/snapshot.json"))
    with np.load(spiking.file("simulation/start/parameters.npz")) as saved:
        sums = {
            name.removesuffix(".weight"): float(
                saved[info["key"]].sum(dtype=np.float64)
                / net["n_" + name.split("_to_")[1][0].lower()]
            )
            for name, info in snapshot["parameters"].items()
        }
    cfg = recipe.mean_field_configuration(net, sums)
    quadrature = np.polynomial.legendre.leggauss(cfg["gain_quadrature_nodes"])
    with stage_run(
        REPO,
        "exp148",
        "compute",
        inputs=dict(spiking=spiking),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        rows = []
        state = np.array([2.0, 6.0, 0.0, 0.0])
        for alpha in np.linspace(1, 0, cfg["alpha_points"]):
            state = mean_field_equilibrium(alpha, state[:2], cfg, quadrature)
            eigenvalues = np.linalg.eigvals(
                mean_field_jacobian(state, alpha, cfg, quadrature)
            )
            rows.append(
                dict(
                    alpha=float(alpha),
                    state=state.tolist(),
                    eigenvalues_real=eigenvalues.real.tolist(),
                    eigenvalues_imag=eigenvalues.imag.tolist(),
                    max_real_per_ms=float(eigenvalues.real.max()),
                )
            )
        crossings = []
        for left, right in zip(rows[:-1], rows[1:]):
            if left["max_real_per_ms"] * right["max_real_per_ms"] >= 0:
                continue

            def leading(alpha):
                equilibrium = mean_field_equilibrium(
                    alpha, left["state"][:2], cfg, quadrature
                )
                return np.linalg.eigvals(
                    mean_field_jacobian(equilibrium, alpha, cfg, quadrature)
                ).real.max()

            alpha = brentq(leading, right["alpha"], left["alpha"], xtol=1e-10)
            equilibrium = mean_field_equilibrium(
                alpha, left["state"][:2], cfg, quadrature
            )
            eigenvalues = np.linalg.eigvals(
                mean_field_jacobian(equilibrium, alpha, cfg, quadrature)
            )
            critical = eigenvalues[np.argmax(eigenvalues.real)]
            crossings.append(
                dict(
                    alpha=float(alpha),
                    frequency_hz=float(abs(critical.imag) * 1000 / (2 * np.pi)),
                    oscillatory=bool(
                        abs(critical.imag) > cfg["eigenvalue_tolerance_per_ms"]
                    ),
                    destabilizes_as_alpha_decreases=left["max_real_per_ms"] < 0,
                )
            )
        initial = np.asarray(rows[0]["state"]).copy()
        initial[0] += cfg["initial_e_perturbation_hz"]
        times = np.arange(
            0,
            net["duration_ms"] + cfg["sample_interval_ms"] / 2,
            cfg["sample_interval_ms"],
        )
        solution = solve_ivp(
            lambda time, value: mean_field_flow(
                value, recipe.drive_gain(time, net), cfg, quadrature
            ),
            (0, net["duration_ms"]),
            initial,
            t_eval=times,
            method=cfg["solver"]["method"],
            rtol=cfg["solver"]["rtol"],
            atol=cfg["solver"]["atol"],
            max_step=cfg["solver"]["max_step_ms"],
        )
        if not solution.success or not np.isfinite(solution.y).all():
            raise RuntimeError(f"Mean-field integration failed: {solution.message}")
        write_json_atomic(
            run.export / "mean-field.json",
            dict(configuration=cfg, continuation=rows, crossings=crossings),
        )
        np.savez_compressed(
            run.export / "mean-field.npz", time_ms=solution.t, state=solution.y.T
        )
        run.record["execution"]["backend"] = (
            "SciPy deterministic four-state ODE and analytical-Jacobian continuation"
        )
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile-mode", choices=("default", "off"), default="off")
    parser.add_argument("--n-e", type=int, default=4000)
    parser.add_argument("--n-i", type=int, default=1000)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--mean-field-source",
        help="Explicit spiking compute run supplying the matrices and protocol",
    )
    parser.add_argument(
        "--record-state",
        action="store_true",
        help="Record population mean voltage and pathway conductances in the three-window protocol",
    )
    compute(**vars(parser.parse_args()))
