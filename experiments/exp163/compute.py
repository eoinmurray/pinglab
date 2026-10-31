"""Run the standalone spatial prototype, preserving and checking explicit exploration evidence."""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
import torch
from experiments.exp163 import recipe
from pingstore.library import file_sha256, source_run, stage_run
from snnlab.sim import (
    DenseArrayBinding,
    NetworkSnapshot,
    Performance,
    RecordingData,
    RecordingSpec,
    SignalRecording,
    SnapshotSpec,
    build,
    simulate,
)


def archive_exploration(source, destination):
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError("exploration source is unavailable")
    files = [
        p
        for p in sorted(source.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and ".pytest_cache" not in p.parts
    ]
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, Path("scratch") / p.relative_to(source))
    with ZipFile(destination) as z:
        for p in files:
            if z.read(str(Path("scratch") / p.relative_to(source))) != p.read_bytes():
                raise ValueError("archive differs from scratch")
    return dict(
        original_source=str(source),
        sha256=file_sha256(destination),
        files=len(files),
        byte_verified=True,
        purpose="historical exploration attachment; not simulation input",
    )


def compute(
    *,
    parameter_source,
    coupling_us=None,
    cue_hz=None,
    background_hz=None,
    contrast=None,
    width=None,
    kernel=None,
    swap=False,
    exploration_source=None,
    mirror_source=None,
    preserve_exploration=None,
    run_id=None,
):
    started = perf_counter()
    torch.set_num_threads(1)
    reference = source_run(
        REPO / ".pingstore", parameter_source, experiment="exp162", stage="compute"
    )
    ref_cfg = reference.record["execution"]["configuration"]
    overrides = {
        k: v
        for k, v in dict(
            coupling_us=coupling_us,
            cue_hz=cue_hz,
            background_hz=background_hz,
            contrast=contrast,
            width=width,
            kernel=kernel,
        ).items()
        if v is not None
    }
    cfg = recipe.configuration(reference_configuration=ref_cfg, swap=swap, **overrides)
    inputs = dict(parameter_reference=reference)
    for role, identity in [
        ("exploration_reference", exploration_source),
        ("mirror_reference", mirror_source),
    ]:
        if identity:
            inputs[role] = source_run(
                REPO / ".pingstore",
                identity,
                experiment="exp162",
                stage="compute",
                phase="scratch-spatial-competition",
            )
    if exploration_source:
        old = inputs["exploration_reference"].record["execution"]["configuration"]
        for key in (
            "coupling_us",
            "cue_hz",
            "background_hz",
            "input_rates_hz",
            "input_seeds",
            "duration_ms",
            "switch_ms",
            "windows_ms",
            "spatial",
        ):
            if cfg[key] != old[key]:
                raise ValueError("exploration comparison differs: " + key)
    original = NetworkSnapshot.read(
        reference.file("execution/start/snapshot.json")
    ).load_parameters()
    weights = recipe.spatial_weights(original, cfg, ref_cfg)
    bundle, _ = recipe.author_network(cfg)
    initialized = build(
        graph=bundle.graph,
        seed=cfg["seed"],
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    parameters = initialized.model.parameter_map()
    if set(parameters) != set(weights):
        raise ValueError("native parameter identities differ")
    with torch.no_grad():
        for key, target in parameters.items():
            if target.shape != weights[key].shape:
                raise ValueError("native parameter shape differs")
            target.copy_(weights[key])
    drives = recipe.make_inputs(cfg)
    with stage_run(
        REPO, recipe.SLUG, "compute", inputs=inputs, configuration=cfg, run_id=run_id
    ) as run:
        bundle.write(run.export / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        checkpoint = run.scratch / "parameters.pt"
        torch.save(initialized.model.state_dict(), checkpoint)
        np.savez_compressed(run.export / "inputs.npz", **drives)
        tick = perf_counter()
        result = simulate(
            graph=bundle.graph,
            seed=cfg["seed"],
            device="cpu",
            checkpoint=checkpoint,
            input_bindings=tuple(
                DenseArrayBinding(
                    k,
                    torch.from_numpy(drives["U" if k.startswith("U_") else k]).float(),
                )
                for k in cfg["input_rates_hz"]
            ),
            snapshot=SnapshotSpec(
                run.export / recipe.SNAPSHOT, stage="start", parameters=True
            ),
            recording=RecordingSpec(
                signals=tuple(SignalRecording(s) for s in cfg["recording_signals"])
            ),
            performance=Performance(compile_mode=None),
            retain_parameters=False,
            retain_runtime_state=False,
            retain_final_state=False,
        )
        elapsed = perf_counter() - tick
        spikes = {
            k.removesuffix("_spikes"): v.detach().cpu().numpy().astype(np.uint8)
            for k, v in result.diagnostics.items()
        }
        np.savez_compressed(run.export / "spikes.npz", **spikes)
        result.write_recording(run.export / "conductances")
        if exploration_source:
            old = inputs["exploration_reference"]
            with np.load(old.file("inputs.npz")) as d:
                assert set(d) == set(drives)
                for k, v in drives.items():
                    np.testing.assert_array_equal(d[k], v)
            with np.load(old.file("spikes.npz")) as d:
                assert set(d) == set(spikes)
                for k, v in spikes.items():
                    np.testing.assert_array_equal(d[k], v)
            recording = RecordingData.read(old.unit("conductances"))
            for k, v in result.recorded_signals.items():
                np.testing.assert_array_equal(
                    recording.signals[k], v.detach().cpu().numpy()
                )
            run.record["execution"][
                "exploration_inputs_spikes_conductances_identical"
            ] = True
        if preserve_exploration is not None:
            run.record["execution"]["migration_archive"] = archive_exploration(
                preserve_exploration, run.export / "exploration.zip"
            )
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={"execution": ["start"]}
        )
        run.record["execution"].update(
            simulation_seconds=elapsed,
            elapsed_seconds=perf_counter() - started,
            snnlab_version=version("snnlab"),
            backend="native conductance-LIF CPU; one thread; spatial checkpoint transforms",
            weights_source="native start parameter arrays after recorded spatial transforms; nominal graph initializers describe the uniform pre-transform reference",
        )
        for item in inputs.values():
            item.check_unchanged()
    print(f"{run.run_id}: simulation {elapsed:.3f} s", flush=True)
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameter-source", required=True)
    for name in [
        "coupling-us",
        "cue-hz",
        "background-hz",
        "contrast",
        "width",
        "kernel",
    ]:
        p.add_argument("--" + name, type=float)
    p.add_argument("--swap", action="store_true")
    p.add_argument("--exploration-source")
    p.add_argument("--mirror-source")
    p.add_argument("--preserve-exploration", type=Path)
    p.add_argument("--run-id")
    compute(**vars(p.parse_args()))
