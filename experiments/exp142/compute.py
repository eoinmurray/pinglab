"""Replay the pinned exp141 classifiers in discrete and continuous time.

Run compute.py with an explicit --source compute ID and optional --input-rate-hz.
Save frozen-model recordings, native snapshots and input/refinement checks in a
new Pingstore compute run. Analysis and presentation are invoked independently."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import torch
from experiments.exp142 import recipe
from experiments.exp142.custom.inputs import encode_sample, event_digest
from experiments.exp142.custom.simulation import simulate_models
from experiments.exp142.custom.source import source_data
from pingstore.contracts import file_sha256, write_json_atomic
from pingstore.stages import source_run, stage_run


def compute(identity, *, run_id=None, input_rate_hz=None):
    cfg = recipe.configuration(input_rate_hz=input_rate_hz)
    source = source_run(
        REPO / ".pingstore",
        identity,
        experiment="exp141",
        stage="compute",
        reference=cfg["source"],
    )
    source_events, models, pixels, label, partition = source_data(source, cfg)
    torch.set_num_threads(cfg["cpu_threads"])
    with stage_run(
        REPO,
        "exp142",
        "compute",
        inputs={"trained": source},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={
                f"{mode}-{case}": ["start"]
                for mode in cfg["models"]
                for case in [
                    "source-control",
                    "native",
                    *[row["id"] for row in cfg["cases"]],
                ]
            }
        )
        graph = json.loads(
            source.file("coba", "best-validation", "start", "snapshot.json").read_text()
        )["network"]
        dataset_path = run.scratch / "sample.npz"
        np.savez_compressed(
            dataset_path,
            sample_features=pixels.reshape(1, -1),
            sample_labels=np.array([label]),
        )
        reconstructed, source_encoding = encode_sample(
            graph, dataset_path, cfg, max_rate_hz=cfg["source_input_rate_hz"]
        )
        for key in source_events:
            np.testing.assert_array_equal(reconstructed[key], source_events[key])
        events, encoding = encode_sample(
            graph, dataset_path, cfg, max_rate_hz=cfg["input_rate_hz"]
        )
        run.record["execution"]["input_encoding"] = dict(
            protocol=encoding,
            source_protocol=source_encoding,
            source_encoding_bitwise_equal=True,
        )
        np.savez_compressed(run.export / "source-input.npz", **source_events)
        np.savez_compressed(
            run.export / "input.npz", **events, pixels=pixels, label=label
        )
        input_digest = file_sha256(run.export / "input.npz")
        audit = dict(
            source=source.reference,
            true_label=label,
            sample_index=cfg["sample_index"],
            sample_partition=partition,
            event_count=len(events["time_ns"]),
            input_rate_hz=cfg["input_rate_hz"],
            input_seed=cfg["input_seed"],
            source_event_count=len(source_events["time_ns"]),
            source_input_rate_hz=cfg["source_input_rate_hz"],
            source_event_sha256=event_digest(source_events),
            input_sha256=input_digest,
            input_event_sha256=event_digest(events),
            source_input_sha256=file_sha256(source.file("recording.npz")),
            cases={},
        )
        checks = simulate_models(run, source, models, source_events, cfg, audit, label)
        if file_sha256(run.export / "input.npz") != input_digest:
            raise ValueError("shared input changed")
        write_json_atomic(run.export / "audit.json", audit)
        write_json_atomic(run.export / "checks.json", checks)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--input-rate-hz", type=float)
    args = parser.parse_args()
    compute(args.source, run_id=args.run_id, input_rate_hz=args.input_rate_hz)
