"""Initialize the symmetric native graph and integrate one cue-switch trajectory."""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp161 import recipe
from experiments.exp161.custom.initialization import capture_start
from pingstore.library import file_sha256, load_json, source_run, stage_run
from snnlab import meanfield as mf
from snnlab.sim import NetworkSnapshot, Performance, build


def archive_exploration(source, destination):
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError("exploration directory is unavailable")
    files = [
        (p, Path("scratch") / p.relative_to(source))
        for p in sorted(source.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    ]
    incomplete = (
        REPO / ".pingstore/runs/.exp159-r007-compute-phase-scratch-competition.tmp"
    )
    if incomplete.is_dir():
        files += [
            (p, Path("incomplete-first-attempt") / p.relative_to(incomplete))
            for p in sorted(incomplete.rglob("*"))
            if p.is_file()
        ]
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as z:
        for path, name in files:
            z.write(path, str(name))
    with ZipFile(destination) as z:
        for path, name in files:
            if z.read(str(name)) != path.read_bytes():
                raise ValueError("exploration archive differs from source")
    return dict(
        original_source=str(source),
        sha256=file_sha256(destination),
        files=len(files),
        purpose="historical exploration attachment, not simulation input",
        byte_verified=True,
    )


def compute(
    *,
    parameter_source,
    coupling_us=None,
    cue_hz=None,
    background_hz=None,
    exploration_source=None,
    preserve_exploration=None,
    run_id=None,
):
    started = perf_counter()
    reference = source_run(
        REPO / ".pingstore", parameter_source, experiment="exp159", stage="compute"
    )
    ref_cfg = reference.record["execution"]["configuration"]
    native = NetworkSnapshot.read(
        reference.file("network-initialization/start/snapshot.json")
    ).load_parameters()
    weights = ref_cfg["incoming_weights_us"]
    for key in ("input_to_E", "E_to_E", "E_to_I", "I_to_E", "I_to_I"):
        p = "PING_B_" + key
        count = ref_cfg["network"]["populations"][key[-1]]["size"]
        actual = float(native[p + ".weight"].numpy().sum(dtype=np.float64) / count)
        np.testing.assert_allclose(actual, weights[p], rtol=1e-12, atol=1e-14)
    overrides = {
        k: v
        for k, v in dict(
            coupling_us=coupling_us, cue_hz=cue_hz, background_hz=background_hz
        ).items()
        if v is not None
    }
    cfg = recipe.configuration(
        reference_network=ref_cfg["network"], incoming_weights_us=weights, **overrides
    )
    bundle, means = recipe.author_network(cfg)
    initialized = build(
        graph=bundle.graph,
        seed=cfg["seed"],
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    sizes = {p["id"]: p["size"] for p in bundle.graph["populations"]}
    for p in bundle.graph["projections"]:
        actual = (
            initialized.parameters[p["id"] + ".weight"].numpy().sum(dtype=np.float64)
            / sizes[p["target"].split(".")[0]]
        )
        np.testing.assert_allclose(actual, means[p["id"]], rtol=1e-6)
    inputs = dict(parameter_reference=reference)
    if exploration_source:
        inputs["exploration_reference"] = source_run(
            REPO / ".pingstore",
            exploration_source,
            experiment="exp159",
            stage="compute",
            phase="scratch-competition",
        )
        old = inputs["exploration_reference"].record["execution"]["configuration"]
        for k in (
            "coupling_us",
            "cue_hz",
            "background_hz",
            "duration_ms",
            "switch_ms",
            "input_rates_hz",
            "windows_ms",
        ):
            if cfg[k] != old[k]:
                raise ValueError("exploration comparison settings differ: " + k)
    with stage_run(
        REPO, recipe.SLUG, "compute", inputs=inputs, configuration=cfg, run_id=run_id
    ) as run:
        bundle.write(run.export / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        capture_start(bundle, initialized, cfg, run.export / recipe.SNAPSHOT)
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={"network-initialization": ["start"]}
        )
        model = recipe.build_model(bundle, cfg, means)
        model.write(run.export / "model.json")
        initial = {
            k: cfg["initial"]["rate_hz" if k.endswith(".rate") else "conductance_us"]
            for k in model.states
        }
        tick = perf_counter()
        solution = mf.integrate(
            model,
            duration=cfg["duration_ms"],
            initial=initial,
            history=mf.ConstantHistory(initial),
            sample_interval=cfg["dt_ms"],
            breakpoints=[cfg["switch_ms"]],
            **cfg["solver"],
        )
        elapsed = perf_counter() - tick
        solution.write(run.export / "trajectory.json")
        if exploration_source:
            original = load_json(
                inputs["exploration_reference"].file("trajectory.json")
            )
            np.testing.assert_array_equal(solution.time_ms, original["time_ms"])
            if set(solution.state) != set(original["state"]):
                raise ValueError("migration changed states")
            for name, values in solution.state.items():
                np.testing.assert_array_equal(values, original["state"][name])
            run.record["execution"]["exploration_trajectory_max_abs_difference"] = 0.0
        if preserve_exploration is not None:
            run.record["execution"]["migration_archive"] = archive_exploration(
                preserve_exploration, run.export / "exploration.zip"
            )
        run.record["execution"].update(
            simulation_seconds=elapsed,
            elapsed_seconds=perf_counter() - started,
            snnlab_version=version("snnlab"),
            backend="native delayed RateConductance; DOP853",
        )
        for item in inputs.values():
            item.check_unchanged()
    print(f"{run.run_id}: simulation {elapsed:.3f} s", flush=True)
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameter-source", required=True)
    p.add_argument("--coupling-us", type=float)
    p.add_argument("--cue-hz", type=float)
    p.add_argument("--background-hz", type=float)
    p.add_argument("--exploration-source")
    p.add_argument("--preserve-exploration", type=Path)
    p.add_argument("--run-id")
    compute(**vars(p.parse_args()))
