"""Execute the complete signed-detuning × coupling grid as batched SNNLab requests."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
import torch
from experiments.exp133 import evidence, inputs, recipe, sender_receiver
from snnlab.sim import (
    DenseArrayBinding,
    ExecutionSpec,
    SnapshotSpec,
    build,
    save_runtime_state,
    simulate,
)


def execute(run, bundle, drives, k, name, state=None, *, checkpoint=None):
    started = perf_counter()
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            graph=bundle.graph,
            seed=recipe.NETWORK_SEED,
            device="cpu",
            checkpoint=checkpoint,
            input_bindings=tuple(
                DenseArrayBinding(key, torch.from_numpy(value).float())
                for key, value in drives.items()
            ),
            options={
                "inference_overrides": {"projection_scales": recipe.coupling_scales(k)}
            },
            snapshot=SnapshotSpec(
                run.export / f"{name}-execution", stage="both", parameters=True
            ),
        ),
        runtime_state=state,
    )
    np.savez_compressed(
        run.export / f"{name}--spikes.npz",
        **{
            key: value.detach().cpu().numpy().astype(np.uint8)
            for key, value in result.diagnostics.items()
        },
    )
    run.record["execution"].setdefault("simulations", {})[name] = dict(
        coupling_us=k,
        seconds=perf_counter() - started,
        steps=len(next(iter(drives.values()))),
        batches=len(recipe.INPUT_DIFFERENCES_HZ),
        continued_from_prefix=state is not None,
    )
    print(f"Completed {name}: {len(recipe.INPUT_DIFFERENCES_HZ)} detunings", flush=True)
    return result.runtime_state


def comparison_checkpoint(run, bundle, reference):
    """Split each original afferent into eight independent, weaker afferents."""
    original = evidence.parameter_reference(reference)
    model = build(
        ExecutionSpec(
            kind="build", graph=bundle.graph, seed=recipe.NETWORK_SEED, device="cpu"
        )
    ).model
    with torch.no_grad():
        for key, parameter in model.parameter_map().items():
            value = original[key]
            if key in {f"PING_{name}_input_to_E.weight" for name in ("A", "B")}:
                if recipe.N_INPUT % value.shape[0]:
                    raise ValueError(
                        "input channel count must be a multiple of reference"
                    )
                factor = recipe.N_INPUT // value.shape[0]
                value = np.tile(value, (factor, 1)) / (factor * recipe.INPUT_GAIN)
            parameter.copy_(torch.from_numpy(value))
    path = run.scratch / "comparison-parameters.pt"
    torch.save(model.state_dict(), path)
    return path


def acquire(run, parameter_reference=None):
    bundle = recipe.author_network()
    bundle.write(run.export / "paired-ping.bundle", visualise=True)
    checkpoint = (
        comparison_checkpoint(run, bundle, parameter_reference)
        if parameter_reference
        else None
    )
    drives = recipe.make_inputs()
    np.savez_compressed(run.export / "inputs.npz", **drives)
    onset = round(recipe.COUPLING_ONSET_MS / recipe.DT_MS)
    prefix = execute(
        run,
        bundle,
        {k: v[:onset] for k, v in drives.items()},
        0.0,
        "prefix",
        checkpoint=checkpoint,
    )
    if prefix is None or prefix.completed_steps != onset:
        raise RuntimeError("missing or incomplete uncoupled prefix")
    save_runtime_state(run.export / "prefix-state", prefix)
    suffix = {k: v[onset:] for k, v in drives.items()}
    for coupling in recipe.K_VALUES_US:
        execute(
            run,
            bundle,
            suffix,
            float(coupling),
            recipe.label(coupling),
            prefix.detached(),
            checkpoint=checkpoint,
        )


def sender_receiver_checkpoint(run, bundle, reference):
    values = sender_receiver.transferred_parameters(reference)
    model = build(
        ExecutionSpec(
            kind="build", graph=bundle.graph, seed=recipe.NETWORK_SEED, device="cpu"
        )
    ).model
    with torch.no_grad():
        for key, parameter in model.parameter_map().items():
            value = values[key]
            if "input_to_E.weight" in key:
                value = value / sender_receiver.INPUT_GAIN
            parameter.copy_(torch.from_numpy(value))
    path = run.scratch / "sender-receiver-parameters.pt"
    torch.save(model.state_dict(), path)
    return path


def acquire_sender_receiver(run, *, calibration=False, reference=None):
    bundle = sender_receiver.author_network(calibration=calibration)
    bundle.write(run.export / "sender-receiver.bundle", visualise=True)
    checkpoint = (
        sender_receiver_checkpoint(run, bundle, reference) if reference else None
    )
    drives = (
        sender_receiver.calibration_inputs()
        if calibration
        else sender_receiver.make_inputs()
    )
    np.savez_compressed(run.export / "inputs.npz", **drives)
    started = perf_counter()
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            graph=bundle.graph,
            seed=recipe.NETWORK_SEED,
            device="cpu",
            checkpoint=checkpoint,
            input_bindings=tuple(
                DenseArrayBinding(key, torch.from_numpy(value).float())
                for key, value in drives.items()
            ),
            options={
                "inference_overrides": {
                    "projection_scales": sender_receiver.input_scales()
                }
            },
            snapshot=SnapshotSpec(
                run.export / "sender-receiver-execution", stage="both", parameters=True
            ),
        )
    )
    np.savez_compressed(
        run.export / "sender-receiver--spikes.npz",
        **{
            key: value.detach().cpu().numpy().astype(np.uint8)
            for key, value in result.diagnostics.items()
        },
    )
    run.record["execution"]["simulation_seconds"] = perf_counter() - started


def compute(*, run_id=None, protocol="grid", parameter_source=None):
    torch.set_num_threads(1)
    if protocol not in ("grid", "sender-receiver", "sender-receiver-calibration"):
        raise ValueError("unknown protocol")
    cfg = (
        recipe.configuration()
        if protocol == "grid"
        else sender_receiver.calibration_configuration()
        if protocol == "sender-receiver-calibration"
        else sender_receiver.configuration()
    )
    reference = None
    if parameter_source:
        reference = inputs.source(REPO, parameter_source, "compute")
        if protocol == "grid":
            evidence.parameter_reference(reference)
        elif protocol == "sender-receiver":
            sender_receiver.transferred_parameters(reference)
        else:
            raise ValueError("calibration does not consume a parameter reference")
    elif protocol == "sender-receiver":
        raise ValueError("sender–receiver requires an explicit grid --parameter-source")
    sources = {"parameter_reference": reference} if reference else {}
    with inputs.execution(
        REPO, "compute", run_id=run_id, configuration=cfg, sources=sources
    ) as run:
        if protocol == "grid":
            acquire(run, reference)
        else:
            acquire_sender_receiver(
                run,
                calibration=protocol == "sender-receiver-calibration",
                reference=reference,
            )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--parameter-source",
        help="Explicit grid compute reference for transferred or expanded weights",
    )
    parser.add_argument(
        "--protocol",
        choices=("grid", "sender-receiver", "sender-receiver-calibration"),
        default="grid",
    )
    args = parser.parse_args()
    compute(
        run_id=args.run_id,
        protocol=args.protocol,
        parameter_source=args.parameter_source,
    )


if __name__ == "__main__":
    main()
