"""Run a small matched spiking pair from an explicit saved neuronal reference."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
import torch
from experiments.exp160 import recipe
from pingstore.library import source_run, stage_run
from snnlab.sim import (
    DenseArrayBinding,
    NetworkSnapshot,
    Performance,
    RecordingSpec,
    SignalRecording,
    SnapshotSpec,
    build,
    simulate,
)


def compute(*, parameter_source, coupling_us=0.28, difference_hz=6.5, run_id=None):
    started = perf_counter()
    torch.set_num_threads(1)
    reference = source_run(
        REPO / ".pingstore", parameter_source, experiment="exp159", stage="compute"
    )
    network = reference.record["execution"]["configuration"]["network"]
    cfg = recipe.configuration(
        network=network, coupling_us=coupling_us, difference_hz=difference_hz
    )
    original = NetworkSnapshot.read(
        reference.file(recipe.REFERENCE_SNAPSHOT)
    ).load_parameters()
    bundle = recipe.author_network(cfg)
    initialized = build(
        graph=bundle.graph,
        seed=network["network_seed"],
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    parameters = initialized.model.parameter_map()
    if set(original) != set(parameters):
        raise ValueError("reference parameter identities differ")
    with torch.no_grad():
        for key, target in parameters.items():
            if target.shape != original[key].shape:
                raise ValueError("reference matrix shape differs")
            target.copy_(original[key])
    drives = recipe.make_inputs(cfg)
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        inputs=dict(parameter_reference=reference),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        bundle.write(run.export / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        checkpoint = run.scratch / "parameters.pt"
        torch.save(initialized.model.state_dict(), checkpoint)
        np.savez_compressed(run.export / "inputs.npz", **drives)
        pair = {}
        timings = {}
        for name, k in [("control", 0), ("coupled", coupling_us)]:
            tick = perf_counter()
            result = simulate(
                graph=bundle.graph,
                seed=network["network_seed"],
                device="cpu",
                checkpoint=checkpoint,
                input_bindings=tuple(
                    DenseArrayBinding(key, torch.from_numpy(value).float())
                    for key, value in drives.items()
                ),
                inference_overrides={"projection_scales": recipe.scales(cfg, k)},
                snapshot=SnapshotSpec(
                    run.export / f"{name}-execution/start",
                    stage="start",
                    parameters=True,
                ),
                recording=RecordingSpec(
                    signals=tuple(SignalRecording(s) for s in cfg["recording_signals"])
                ),
                performance=Performance(compile_mode=None),
                retain_parameters=False,
                retain_runtime_state=False,
                retain_final_state=False,
            )
            timings[name] = perf_counter() - tick
            pair[name] = {
                key: value.detach().cpu().numpy().astype(np.uint8)
                for key, value in result.diagnostics.items()
            }
            np.savez_compressed(run.export / f"{name}-spikes.npz", **pair[name])
            # Retain native unit/time metadata and all recorded cells losslessly.
            result.write_recording(run.export / f"{name}-conductances")
            print(
                name,
                "seconds",
                timings[name],
                "spike keys",
                list(pair[name]),
                "recorded",
                list(result.recorded_signals),
                flush=True,
            )
        prefix = round(network["onset_ms"] / network["dt_ms"])
        for key in pair["control"]:
            np.testing.assert_array_equal(
                pair["control"][key][:prefix], pair["coupled"][key][:prefix]
            )
        for p in ("E", "I"):
            np.testing.assert_array_equal(
                pair["control"][f"A_{p}"], pair["coupled"][f"A_{p}"]
            )
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={
                f"{name}-execution": ["start"] for name in ("control", "coupled")
            }
        )
        run.record["execution"].update(
            timings_seconds=timings,
            elapsed_seconds=perf_counter() - started,
            matched_prefix_verified=True,
            sender_spikes_identical=True,
            backend="snnlab.sim conductance-LIF CPU",
        )
        reference.check_unchanged()
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameter-source", required=True)
    p.add_argument("--run-id")
    p.add_argument("--coupling-us", type=float, default=0.28)
    p.add_argument("--difference-hz", type=float, default=6.5)
    compute(**vars(p.parse_args()))
