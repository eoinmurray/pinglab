"""Run fixed PING comparisons or the independent-signal routing assay."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
import torch
from experiments.exp136 import recipe
from experiments.helpers.network_artifacts import snapshot_parameters
from pingstore.contracts import load_json
from pingstore.stages import source_run, stage_run
from snnlab.sim import DenseArrayBinding, ExecutionSpec, SnapshotSpec, build, simulate


def make_inputs(cfg):
    """Paired background trains and one identical 1 s cue for each sender."""
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    drives = {}
    for name, rate in cfg["background_hz"].items():
        uniforms = np.random.default_rng(cfg["input_seeds"][name]).random(
            (steps, 1, cfg["input_channels"]), dtype=np.float32
        )
        drives[f"{name}_sensory_drive"] = (
            uniforms < rate * cfg["dt_ms"] / 1000
        ).astype(np.uint8)
    # Preserve the selected trial's cue stream, including its initial 1 s prefix.
    width = round(cfg["segment_ms"] / cfg["dt_ms"])
    start = round(cfg["cue_segments"][0][1] / cfg["dt_ms"])
    uniforms = np.random.default_rng(cfg["input_seeds"]["cue"]).random(
        (start + width, 1, cfg["cue_input_channels"]), dtype=np.float32
    )
    active = (uniforms[start:] < cfg["cue_drive_hz"] * cfg["dt_ms"] / 1000).astype(
        np.uint8
    )
    for sender, onset, offset in cfg["cue_segments"]:
        cue = np.zeros((steps, 1, cfg["cue_input_channels"]), dtype=np.uint8)
        cue[round(onset / cfg["dt_ms"]) : round(offset / cfg["dt_ms"])] = active
        drives[f"cue_{sender}_drive"] = cue
    return drives


def condition_parameters(values, condition):
    """Only the four sender-to-R matrices differ between the two trials."""
    if condition not in ("coupled", "zero"):
        raise ValueError("expected coupled or zero")
    forward = {f"{s}_E_to_R_{p}.weight" for s in ("S1", "S2") for p in ("E", "I")}
    return {
        k: np.zeros_like(v) if condition == "zero" and k in forward else v
        for k, v in values.items()
    }


def verify_snapshot(directory, expected):
    meta = load_json(directory / "snapshot.json")
    with np.load(directory / "parameters.npz", allow_pickle=False) as arrays:
        if set(meta["parameters"]) != set(expected):
            raise ValueError("snapshot parameter coverage differs")
        for name, entry in meta["parameters"].items():
            if not np.array_equal(arrays[entry["key"]], expected[name]):
                raise ValueError(f"unexpected parameter change: {name}")


def compute(
    identity=recipe.PARAMETER_SOURCE, *, run_id=None, task="synchrony", selection=None
):
    if task in ("receiver", "refine-confirm"):
        from experiments.exp136.receiver_refine import compute_refinement

        return compute_refinement(
            REPO,
            identity,
            selection,
            confirmation=task == "refine-confirm",
            run_id=run_id,
        )
    if task in ("coarse", "fine", "confirm"):
        from experiments.exp136.routing_search import compute_search

        return compute_search(REPO, identity, task, selection, run_id=run_id)
    if task == "attention":
        from experiments.exp136.attention_compute import compute_attention

        return compute_attention(REPO, identity, run_id=run_id)
    torch.set_num_threads(1)
    source = source_run(
        REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    cfg = recipe.configuration()
    values = snapshot_parameters(source, cfg["parameter_unit"], "start")
    for sender in ("S1", "S2"):
        for target in ("E", "I"):
            if not np.allclose(
                values[f"{sender}_E_to_R_{target}.weight"].sum(axis=0),
                recipe.FORWARD_US,
            ):
                raise ValueError("source is not the selected coupling strength")
        if not np.allclose(
            values[f"cue_{sender}_E_to_{sender}_E.weight"].sum(axis=0),
            recipe.CUE_TO_E_US,
        ):
            raise ValueError("source is not the selected cue strength")
    bundle = recipe.author_network()
    model = build(
        ExecutionSpec(
            kind="build", graph=bundle.graph, seed=cfg["network_seed"], device="cpu"
        )
    ).model
    if set(model.parameter_map()) != set(values):
        raise ValueError("retained parameters do not match the fixed circuit")
    drives = make_inputs(cfg)
    bindings = tuple(
        DenseArrayBinding(k, torch.from_numpy(v).float()) for k, v in drives.items()
    )
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        inputs={"parameters": source},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        recipe.write_network_bundle(bundle, run.export / "network.bundle")
        np.savez_compressed(run.export / "inputs.npz", **drives)
        previous = None
        for condition in cfg["conditions"]:
            expected = condition_parameters(values, condition)
            with torch.no_grad():
                for key, parameter in model.parameter_map().items():
                    parameter.copy_(torch.from_numpy(expected[key]))
            checkpoint = run.scratch / f"{condition}.pt"
            torch.save(model.state_dict(), checkpoint)
            started = perf_counter()
            result = simulate(
                ExecutionSpec(
                    kind="simulate",
                    graph=bundle.graph,
                    seed=cfg["network_seed"],
                    device="cpu",
                    checkpoint=checkpoint,
                    input_bindings=bindings,
                    snapshot=SnapshotSpec(
                        run.export / f"{condition}-execution",
                        stage="both",
                        parameters=True,
                    ),
                )
            )
            spikes = {
                k: v.detach().cpu().numpy().astype(np.uint8)
                for k, v in result.outputs.items()
            }
            for stage in ("start", "end"):
                verify_snapshot(run.export / f"{condition}-execution" / stage, expected)
            if previous is not None and any(
                not np.array_equal(v, previous[k])
                for k, v in spikes.items()
                if not k.startswith("R_")
            ):
                raise ValueError("paired sender/cue activity differs")
            previous = spikes
            np.savez_compressed(run.export / f"{condition}--spikes.npz", **spikes)
            seconds = perf_counter() - started
            run.record["execution"].setdefault("simulations", {})[condition] = dict(
                seconds=seconds, batch_size=1
            )
            print(f"Completed {condition} ({seconds:.1f}s)", flush=True)
        run.record["execution"]["paired_parameters_and_inputs_verified"] = True
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=recipe.PARAMETER_SOURCE,
        help="Explicit retained parameter run (default: selected fixed matrices).",
    )
    parser.add_argument(
        "--task",
        choices=(
            "synchrony",
            "attention",
            "coarse",
            "fine",
            "confirm",
            "receiver",
            "refine-confirm",
        ),
        default="synchrony",
    )
    parser.add_argument(
        "--selection",
        help="Explicit coarse/fine analyse run for refinement/confirmation.",
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    compute(args.source, run_id=args.run_id, task=args.task, selection=args.selection)
