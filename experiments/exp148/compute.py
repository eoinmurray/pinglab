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
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import numpy as np
import torch
from experiments.exp148 import dataset, recipe
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from snnlab import lang
from snnlab import meanfield as mf
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


def compute_mean_field(source, *, run_id=None):
    """Reduce explicit spiking evidence and execute SNNLab mean-field analysis."""
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
    with stage_run(
        REPO,
        "exp148",
        "compute",
        inputs=dict(spiking=spiking),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        bundle = lang.load_bundle(
            spiking.restore_bundle(
                "simulation/network.bundle", run.scratch / "spiking.bundle"
            )
        )
        model = recipe.mean_field_model(cfg, bundle)
        model.write(run.export / "mean-field-model.json")
        branch = mf.continuation(
            model,
            "X_to_I.gain",
            np.linspace(1, 0, cfg["alpha_points"]),
            guess=cfg["equilibrium_guess"],
        )
        rows = [
            dict(
                alpha=alpha,
                state=list(equilibrium.state.values()),
                eigenvalues_real=stability.eigenvalues.real.tolist(),
                eigenvalues_imag=stability.eigenvalues.imag.tolist(),
                max_real_per_ms=stability.max_real,
            )
            for alpha, equilibrium, stability in branch
        ]
        crossings = [
            dict(
                alpha=crossing.value,
                frequency_hz=crossing.frequency_hz,
                oscillatory=crossing.oscillatory,
                destabilizes_as_alpha_decreases=crossing.stable_before is True
                and crossing.stable_after is False,
            )
            for crossing in mf.find_stability_crossings(
                model,
                "X_to_I.gain",
                branch,
                tolerance=cfg["crossing_tolerance"],
                eigenvalue_tolerance=cfg["eigenvalue_tolerance_per_ms"],
            )
        ]
        initial = dict(branch[0][1].state)
        initial["E.rate"] += cfg["initial_e_perturbation_hz"]
        solution = mf.integrate(
            model,
            duration=net["duration_ms"],
            initial=initial,
            controls={"X_to_I.gain": recipe.mean_field_control(cfg)},
            sample_interval=cfg["sample_interval_ms"],
            method=cfg["solver"]["method"],
            rtol=cfg["solver"]["rtol"],
            atol=cfg["solver"]["atol"],
            max_step=cfg["solver"]["max_step_ms"],
        )
        write_json_atomic(
            run.export / "mean-field.json",
            dict(configuration=cfg, continuation=rows, crossings=crossings),
        )
        np.savez_compressed(
            run.export / "mean-field.npz",
            time_ms=solution.time_ms,
            state=np.array(list(solution.state.values())).T,
        )
        run.record["execution"]["backend"] = (
            "snnlab.meanfield four-state rate-conductance ODE"
        )
        run.record["execution"]["snnlab_version"] = version("snnlab")
        spiking.check_unchanged()
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
