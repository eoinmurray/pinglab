"""Initialize one network and run the matched uncoupled/one-way pair."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp159 import recipe
from experiments.exp159.custom.initialization import capture_start
from pingstore.library import file_sha256, stage_run
from snnlab import meanfield as mf
from snnlab.sim import Performance, build


def mean_incoming_weights(bundle, parameters):
    targets = {p["id"]: p["size"] for p in bundle.graph["populations"]}
    result = {}
    for p in bundle.graph["projections"]:
        matrix = parameters[p["id"] + ".weight"].detach().cpu().numpy()
        count = targets[p["target"].rsplit(".", 1)[0]]
        if (
            matrix.ndim != 2
            or matrix.shape[1] != count
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("invalid native source-target matrix")
        result[p["id"]] = float(matrix.sum(dtype=np.float64) / count)
    return result


def integrate(model, cfg):
    initial = recipe.initial_state(model, cfg)
    return mf.integrate(
        model,
        duration=cfg["duration_ms"],
        initial=initial,
        history=mf.ConstantHistory(initial),
        sample_interval=cfg["sample_interval_ms"],
        **dict(
            method=cfg["solver"]["method"],
            rtol=cfg["solver"]["rtol"],
            atol=cfg["solver"]["atol"],
            max_step=cfg["solver"]["max_step_ms"],
        ),
    )


def verify_pair(control, coupled, *, onset_ms=500.0):
    if not np.array_equal(control.time_ms, coupled.time_ms):
        raise ValueError("pair observation grids differ")
    np.testing.assert_allclose(
        control.state["PING_A_E.rate"],
        coupled.state["PING_A_E.rate"],
        rtol=0,
        atol=1e-6,
    )
    prefix = control.time_ms < onset_ms
    for name in control.state:
        np.testing.assert_allclose(
            np.asarray(control.state[name])[prefix],
            np.asarray(coupled.state[name])[prefix],
            rtol=0,
            atol=1e-7,
        )


def archive_exploration(source, destination):
    """Optional preservation attachment, never an input to the new simulations."""
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError("exploration archive source must be a directory")
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, Path("exploration") / path.relative_to(source))
        provenance = json.loads((source / "attempt-04/provenance.json").read_text())
        for name, expected in provenance["sources"].items():
            if name.startswith("experiments/"):
                path = REPO / name
                if file_sha256(path).removeprefix("sha256:") != expected:
                    raise ValueError("exploration code dependency changed: " + name)
                archive.write(path, Path("dependencies") / name)
    return dict(
        kind="historical exploration attachment; not a scientific input",
        original_source=str(source),
        archive_sha256=file_sha256(destination),
    )


def compute(
    *, difference_hz=2.0, coupling_us=0.06, run_id=None, preserve_exploration=None
):
    started = perf_counter()
    cfg = recipe.configuration(difference_hz=difference_hz, coupling_us=coupling_us)
    bundle = recipe.author_network(cfg)
    initialized = build(
        graph=bundle.graph,
        seed=cfg["network"]["network_seed"],
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    cfg = recipe.configuration(
        difference_hz=difference_hz,
        coupling_us=coupling_us,
        incoming_weights_us=mean_incoming_weights(bundle, initialized.parameters),
    )
    with stage_run(
        REPO, recipe.SLUG, "compute", inputs={}, configuration=cfg, run_id=run_id
    ) as run:
        bundle.write(run.export / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        capture_start(bundle, initialized, cfg, run.export / recipe.SNAPSHOT)
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={"network-initialization": ["start"]}
        )
        solutions = {}
        timings = {}
        for label, k in [("control", 0), ("coupled", coupling_us)]:
            model = recipe.build_model(
                bundle, cfg, difference_hz=difference_hz, coupling_us=k
            )
            model.write(run.export / f"{label}-model.json")
            tick = perf_counter()
            solution = integrate(model, cfg)
            timings[label] = perf_counter() - tick
            solution.write(run.export / f"{label}-trajectory.json")
            solutions[label] = solution
            print(f"{label}: {timings[label]:.2f} seconds", flush=True)
        verify_pair(
            solutions["control"],
            solutions["coupled"],
            onset_ms=cfg["network"]["onset_ms"],
        )
        run.record["execution"].update(
            snnlab_version=version("snnlab"),
            backend="native delayed RateConductance; DOP853",
            timings_seconds=timings,
            pair_invariants_verified=True,
        )
        if preserve_exploration is not None:
            run.record["execution"]["migration"] = archive_exploration(
                preserve_exploration, run.export / "exploration.zip"
            )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--difference-hz", type=float, default=2)
    p.add_argument("--coupling-us", type=float, default=0.06)
    p.add_argument("--run-id")
    p.add_argument("--preserve-exploration", type=Path)
    compute(**vars(p.parse_args()))
