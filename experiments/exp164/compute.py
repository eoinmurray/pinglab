"""Simulate the standalone fixed digit-routing circuit on the local CPU."""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
import torch
from experiments.exp164 import recipe
from pingstore.library import stage_run
from snnlab.sim import DenseArrayBinding, Performance, SnapshotSpec, build, simulate


def compute(*, seed=101, run_id=None):
    started = perf_counter()
    torch.set_num_threads(1)
    cfg = recipe.simulation_configuration(seed=seed)
    recipe.configuration(source_configuration=cfg)
    images = recipe.digit_images(cfg)
    drives = recipe.make_inputs(cfg, images)
    bundle, _ = recipe.author_network(cfg)
    initialized = build(
        graph=bundle.graph,
        seed=cfg["network_seed"],
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    with torch.no_grad():
        recipe.spatial_parameters(initialized.model.parameter_map(), cfg)
    with stage_run(
        REPO, recipe.SLUG, "compute", inputs={}, configuration=cfg, run_id=run_id
    ) as run:
        bundle.write(run.export / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        checkpoint = run.scratch / "parameters.pt"
        torch.save(initialized.model.state_dict(), checkpoint)
        np.savez_compressed(run.export / "images.npz", **images)
        np.savez_compressed(run.export / "inputs.npz", **drives)
        tick = perf_counter()
        result = simulate(
            graph=bundle.graph,
            seed=cfg["network_seed"],
            device="cpu",
            checkpoint=checkpoint,
            input_bindings=tuple(
                DenseArrayBinding(k, torch.from_numpy(v).float())
                for k, v in drives.items()
            ),
            snapshot=SnapshotSpec(
                run.export / "execution/start", stage="start", parameters=True
            ),
            performance=Performance(compile_mode=None),
            retain_parameters=False,
            retain_runtime_state=False,
            retain_final_state=False,
        )
        elapsed = perf_counter() - tick
        np.savez_compressed(
            run.export / "spikes.npz",
            **{
                k.removesuffix("_spikes"): v.detach().cpu().numpy().astype(np.uint8)
                for k, v in result.diagnostics.items()
            },
        )
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={"execution": ["start"]}
        )
        run.record["execution"].update(
            simulation_seconds=elapsed,
            elapsed_seconds=perf_counter() - started,
            snnlab_version=version("snnlab"),
            no_simulation=False,
            backend="native conductance-LIF CPU; one thread; two-condition batch",
            weights_source="standalone graph initializers with pixel-block sensory and normalized Gaussian cross-projection transforms",
        )
    print(f"{run.run_id}: simulation {elapsed:.3f} s", flush=True)
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--run-id")
    compute(**vars(p.parse_args()))
