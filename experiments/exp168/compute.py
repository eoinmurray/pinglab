"""Simulate the EE/II-on comparison, or run a short explicit diagnostic."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp168 import dataset, recipe
from pingstore.library import stage_run


def execute(cfg, directory, diagram):
    from snnlab.sim import (
        DenseArrayBinding,
        Performance,
        RecordingSpec,
        SignalRecording,
        SnapshotSpec,
        simulate,
    )

    directory.mkdir(parents=True)
    bundle = recipe.author_network(cfg)
    bundle.write(directory / "network.bundle", visualise=False)
    if diagram:
        bundle.visualise(directory.parent / "circuit.svg", view="circuit")
    drive = dataset.encode_drive(cfg).to(cfg["device"])
    events = {"E": [], "I": []}

    def collect(block):
        if block.kind == "spike_events":
            events[block.signal.split(".")[0]].append(block.values.numpy())

    simulate(
        bundle=bundle,
        input_bindings=(DenseArrayBinding("X", drive),),
        seed=cfg["seed"],
        device=cfg["device"],
        performance=Performance(**cfg["performance"]),
        diagnostics=False,
        recording=RecordingSpec(
            signals=tuple(
                SignalRecording(p + ".spikes", kind="spike_events") for p in ("E", "I")
            ),
            sink=collect,
            sink_id="exp168.sparse-spikes/v2",
            block_steps=cfg["recording_block_steps"],
        ),
        snapshot=SnapshotSpec(directory / "start", stage="start", parameters=True),
        retain_outputs=False,
        retain_parameters=True,
        retain_final_state=False,
        retain_runtime_state=False,
    )
    arrays = {}
    for population, blocks in events.items():
        rows = np.concatenate(blocks) if blocks else np.empty((0, 3))
        arrays[population + "_time_ms"] = rows[:, 0] * cfg["dt_ms"]
        arrays[population + "_cell"] = rows[:, 2]
    np.savez_compressed(directory / "events.npz", **arrays)


def simulate_conditions(
    *,
    device="cpu",
    duration_ms=2050.0,
    seed=43,
    ai_rate_hz=None,
    ai_probability=None,
    ai_gain=None,
    ping_rate_hz=None,
    ping_probability=None,
    ping_gain=None,
    run_id=None,
):
    from importlib.metadata import version

    import torch

    if duration_ms != 2050.0:
        raise ValueError("retained compute requires the requested 2050 ms duration")
    started = perf_counter()
    configs = {
        state: recipe.configuration(
            state,
            recurrence="on",
            device=device,
            seed=seed,
            external_rate_hz=ai_rate_hz if state == "AI" else ping_rate_hz,
            input_connection_probability=(
                ai_probability if state == "AI" else ping_probability
            ),
            input_i_gain=ai_gain if state == "AI" else ping_gain,
        )
        for state in recipe.CONDITIONS
    }
    torch.set_num_threads(configs["AI"]["torch_threads"])
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        configuration={"recurrence": "on", "conditions": configs},
        run_id=run_id,
    ) as run:
        run.record["scientific_artifacts"] = {
            "snapshot_stages": {state: ["start"] for state in recipe.CONDITIONS}
        }
        for index, (state, cfg) in enumerate(configs.items()):
            execute(cfg, run.export / state, index == 0)
        run.record["execution"].update(
            elapsed_s=perf_counter() - started,
            snnlab_version=version("snnlab"),
        )
    return run.run_id


def diagnostic(args):
    import tempfile

    import torch

    torch.set_num_threads(4)
    started = perf_counter()
    cfg = recipe.configuration(
        args.condition,
        recurrence="on",
        device=args.device,
        seed=args.seed,
        external_rate_hz=args.rate_hz,
        input_connection_probability=args.probability,
        input_i_gain=args.gain,
    )
    cfg["duration_ms"] = args.duration_ms
    cfg["measurement_ms"] = [50.0, args.duration_ms]
    with tempfile.TemporaryDirectory() as temporary:
        execute(cfg, Path(temporary) / "run", False)
        with np.load(
            Path(temporary) / "run" / "events.npz", allow_pickle=False
        ) as events:
            summary = {}
            for population in ("E", "I"):
                times = events[population + "_time_ms"]
                mask = (times >= 50.0) & (times < args.duration_ms)
                summary[population + "_rate_hz"] = float(
                    mask.sum()
                    / cfg["n_" + population.lower()]
                    / ((args.duration_ms - 50.0) / 1000)
                )
                summary[population + "_events"] = int(mask.sum())
    print(
        {
            "condition": args.condition,
            "rate_hz": args.rate_hz,
            "probability": args.probability,
            "gain": args.gain,
            "summary": summary,
            "elapsed_s": perf_counter() - started,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    simulate_parser = commands.add_parser("simulate")
    simulate_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    simulate_parser.add_argument("--seed", type=int, default=43)
    simulate_parser.add_argument("--ai-rate-hz", type=float)
    simulate_parser.add_argument("--ai-probability", type=float)
    simulate_parser.add_argument("--ai-gain", type=float)
    simulate_parser.add_argument("--ping-rate-hz", type=float)
    simulate_parser.add_argument("--ping-probability", type=float)
    simulate_parser.add_argument("--ping-gain", type=float)
    simulate_parser.add_argument("--run-id")
    diagnostic_parser = commands.add_parser("diagnostic")
    diagnostic_parser.add_argument("condition", choices=recipe.CONDITIONS)
    diagnostic_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    diagnostic_parser.add_argument("--seed", type=int, default=43)
    diagnostic_parser.add_argument("--duration-ms", type=float, default=205.0)
    diagnostic_parser.add_argument("--rate-hz", type=float)
    diagnostic_parser.add_argument("--probability", type=float)
    diagnostic_parser.add_argument("--gain", type=float)
    args = parser.parse_args()
    if args.command == "simulate":
        simulate_conditions(
            device=args.device,
            seed=args.seed,
            ai_rate_hz=args.ai_rate_hz,
            ai_probability=args.ai_probability,
            ai_gain=args.ai_gain,
            ping_rate_hz=args.ping_rate_hz,
            ping_probability=args.ping_probability,
            ping_gain=args.ping_gain,
            run_id=args.run_id,
        )
    else:
        diagnostic(args)
