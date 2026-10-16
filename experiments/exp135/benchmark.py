"""Bounded real-data GPU throughput experiments, isolated from accuracy trials."""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]


def execute_case(config):
    import numpy as np
    import torch
    from experiments.exp135 import dataset, recipe  # noqa: F401
    from pingstore.contracts import write_json_atomic
    from snnlab import sim

    torch.set_num_threads(int(os.environ.get("PINGLAB_CPU_THREADS", "4")))
    output = Path(config["output"])
    output.mkdir(parents=True)
    duration = config["duration_ms"]
    bundle = recipe.author_network(config["model"], presentation_ms=duration, seed=42)
    bundle.write(output / "network.bundle", visualise=True)
    timings = []
    peaks = []

    def report(row):
        timings.append(dict(row.timing))
        peaks.append(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = sim.train(
        sim.ExecutionSpec(
            kind="train",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(
                sim.DatasetSnapshotBinding(
                    path=Path(config["dataset"]),
                    input_id="events",
                    dataset_id="tonic-dvs-gesture-training",
                    split="train",
                    target_id="gesture",
                    sample_cap=config["batch"],
                    encoder=sim.DatasetEncoder(
                        "custom",
                        definition="exp135.random_crop/v1",
                        seed=42,
                        config={"duration_ms": duration, "random": True},
                    ),
                ),
            ),
            epochs=config["epochs"],
            batch_size=config["batch"],
            seed=42,
            device="cuda",
            compile_mode=config["compile_mode"],
            batching=sim.Batching(order="sequential"),
            metrics=sim.TrainingMetrics(
                train="online", validation="none", initial=(), update_history=False
            ),
            checkpoint_selection=sim.CheckpointSelection.disabled(),
            training_timing=True,
            on_epoch_end=report,
            diagnostics=False,
            retain_parameters=False,
            retain_runtime_state=False,
            retain_final_state=False,
            snapshot=sim.SnapshotSpec(
                output / "execution", stage="both", parameters=True
            ),
            tracing=sim.TraceSpec(
                output=output / "profile",
                mode="both",
                skip_updates=0,
                active_updates=1,
                record_shapes=False,
                profile_memory=False,
            )
            if config.get("profile")
            else None,
        )
    )
    wall = time.perf_counter() - start
    losses = [r["train_loss"] for r in result.metrics["epochs"] if "train_loss" in r]
    if not all(np.isfinite(losses)):
        raise ValueError("nonfinite benchmark loss")
    steady = [r["training_s"] for r in timings[1:]]
    row = {
        **config,
        "status": "ok",
        "wall_s": wall,
        "epoch_timings": timings,
        "peak_allocated_bytes": max(peaks),
        "steady_update_s": statistics.median(steady) if steady else None,
        "examples_per_s": config["batch"] / statistics.median(steady)
        if steady
        else None,
        "losses": losses,
        "gpu": torch.cuda.get_device_name(),
    }
    row.pop("output")
    row.pop("dataset")
    write_json_atomic(output / "measurement.json", row)
    print(json.dumps(row), flush=True)


def run_benchmark(model, run_id):
    from experiments.exp135 import dataset, recipe
    from pingstore.contracts import write_json_atomic
    from pingstore.stages import stage_run

    cfg = recipe.benchmark_configuration()
    phase = f"gpu-benchmark-{model}"
    with stage_run(
        REPO, "exp135", "compute", run_id=run_id, phase=phase, configuration=cfg
    ) as run:
        archive = dataset.verified_archive(
            Path("/modal-output/datasets/ibmGestureTrain.tar.gz"), download=False
        )
        cfg["resolved_dataset"] = dataset.prepare(archive, run.export)
        measurements = []
        failures = []

        def case(batch, compile_mode, duration, profile=False):
            name = (
                f"{model}-b{batch}-{'compiled' if compile_mode else 'eager'}-{int(duration)}ms"
                + ("-profile" if profile else "")
            )
            output = run.export / name
            config = dict(
                model=model,
                batch=batch,
                compile_mode=compile_mode,
                duration_ms=duration,
                profile=profile,
                output=str(output),
                dataset=str(run.export / "train-events.npz"),
                epochs=cfg["epochs"],
            )
            config_path = run.scratch / f"{name}.json"
            write_json_atomic(config_path, config)
            log = run.scratch / f"{name}.log"
            print(f"BENCHMARK START {name}", flush=True)
            try:
                with log.open("w") as handle:
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            str(config_path),
                        ],
                        cwd=REPO,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        timeout=cfg["case_timeout_s"],
                        check=True,
                    )
                row = json.loads((output / "measurement.json").read_text())
                measurements.append(row)
                print(
                    f"BENCHMARK DONE {name}: {row['examples_per_s']} examples/s; peak={row['peak_allocated_bytes'] / 2**30:.2f} GiB",
                    flush=True,
                )
                return row
            except (subprocess.SubprocessError, OSError, ValueError) as error:
                tail = log.read_text()[-5000:]
                failures.append(dict(case=name, error=str(error), detail=tail))
                print(f"BENCHMARK FAILED {name}: {tail[-900:]}", flush=True)
                if output.exists():
                    shutil.move(str(output), str(run.scratch / f"failed-{name}"))
                return None

        successful = []
        for batch in cfg["batches"]:
            row = case(batch, "default", cfg["duration_ms"])
            if row:
                successful.append(row)
            else:
                break
        if not successful:
            raise RuntimeError("no compiled crop benchmark succeeded")
        winner = max(successful, key=lambda r: r["examples_per_s"])
        write_json_atomic(
            run.export / f"{model}--benchmark.json",
            dict(
                measurements=measurements,
                failures=failures,
                fastest=winner,
                limitations="First B selected real training examples; fresh seeded random 500 ms crops each epoch. Three measured updates after one warmup. No validation/accuracy or convergence claim.",
            ),
        )
        run.record["execution"].update(
            device="cuda", backend="snnlab.sim.train", benchmark_failures=failures
        )
        write_json_atomic(run.directory / "run.json", run.record)
    print(f"COMPLETED {run.run_id}", flush=True)
    return run.run_id


if __name__ == "__main__":
    execute_case(json.loads(Path(sys.argv[1]).read_text()))
