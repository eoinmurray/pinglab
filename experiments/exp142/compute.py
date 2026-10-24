"""Replay the pinned exp141 classifiers in discrete and continuous time.

Run compute.py with an explicit --source compute ID and optional --input-rate-hz.
Save frozen-model recordings, native snapshots and input/refinement checks in a
new Pingstore compute run. Analysis and presentation are invoked independently."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse
import json

import numpy as np
import torch
from experiments.exp142 import recipe
from experiments.exp142.analyse import arrays, check_anchor, refinement_check, validate
from experiments.exp142.custom.continuous import simulate_continuous
from experiments.exp142.custom.discrete import execute
from experiments.exp142.custom.external_artifacts import external_execution
from experiments.exp142.custom.inputs import encode_sample, event_digest
from experiments.exp142.custom.source import source_data, source_recording
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


def simulate_models(run, source, models, source_events, cfg, audit, label):
    anchors, refinements = {}, {}
    for mode, weights in models.items():
        ni = cfg["n_i"] if mode == "ping" else 0
        originals = source_recording(source, mode)
        # Preserve the original 25 Hz reproduction control separately from
        # all requested-rate inference, including the native-clock case.
        control_dir = run.export / f"{mode}-source-control"
        control, control_diagnostics = execute(
            source_events,
            weights,
            mode,
            cfg,
            cfg["input_grid_ns"],
            control_dir,
            matched=False,
        )
        validate(control, cfg, ni, sample_dt_ns=cfg["native_sample_dt_ns"])
        anchors[mode] = check_anchor(control, originals, cfg, ni)
        np.savez_compressed(control_dir / "recording.npz", **control)
        run.record["execution"].setdefault("cases", {})[f"{mode}-source-control"] = (
            control_diagnostics
        )
        conditions = [("native", cfg["input_grid_ns"], False)] + [
            (row["id"], row["dt_ns"], True) for row in cfg["cases"]
        ]
        continuous = {}
        for case, dt_ns, matched in conditions + [
            ("continuous", None, True),
            ("continuous-refined", None, True),
        ]:
            unit = f"{mode}-{case}"
            directory = run.export / unit
            saved = arrays(run.export / "input.npz")
            replay = {key: saved[key] for key in ("time_ns", "batches", "channels")}
            if event_digest(replay) != audit["input_event_sha256"]:
                raise ValueError("physical input coordinates changed")
            if dt_ns is not None:
                recording, diagnostics = execute(
                    replay,
                    weights,
                    mode,
                    cfg,
                    dt_ns,
                    directory,
                    matched=matched,
                    diagram=case == cfg["cases"][0]["id"],
                )
            else:
                solver = cfg["solver"] if case == "continuous" else cfg["refinement"]
                w = {key: value.copy() for key, value in weights.items()}
                if not ni:
                    w.update(
                        {
                            "E_to_I.weight": np.zeros((cfg["n_e"], 0), np.float32),
                            "I_to_E.weight": np.zeros((0, cfg["n_e"]), np.float32),
                        }
                    )
                promoted = {key: value.astype(np.float64) for key, value in w.items()}
                promoted.update(
                    {key: np.float64(value) for key, value in cfg["physics"].items()}
                )
                with external_execution(
                    directory,
                    backend="SciPy DOP853 hybrid COBA-LIF (E/I/readout)",
                    configuration={"physics": cfg["physics"], "solver": solver},
                    parameters=lambda: promoted,
                ):
                    recording, diagnostics = simulate_continuous(
                        replay,
                        w,
                        defaults=cfg["physics"],
                        duration_ms=cfg["duration_ns"] / 1e6,
                        sample_dt_ms=cfg["sample_dt_ns"] / 1e6,
                        **solver,
                    )
                continuous[case] = recording
            validate(
                recording,
                cfg,
                ni,
                sample_dt_ns=cfg["native_sample_dt_ns"] if case == "native" else None,
            )
            if (
                dt_ns is not None
                and diagnostics["input_event_sha256"] != audit["input_event_sha256"]
            ):
                raise ValueError("native input resolution changed physical coordinates")
            np.savez_compressed(directory / "recording.npz", **recording)
            audit["cases"][unit] = dict(
                event_count=len(replay["time_ns"]),
                input_equivalent=True,
                input_event_sha256=event_digest(replay),
                max_time_error_ns=0,
            )
            run.record["execution"].setdefault("cases", {})[unit] = diagnostics
            print(
                f"{unit}: prediction={recording['class_scores'].argmax()}, label={label}, counts={recording['class_scores'].tolist()}",
                flush=True,
            )
        refinements[mode] = refinement_check(
            continuous["continuous"], continuous["continuous-refined"], cfg, ni
        )
    return dict(anchors=anchors, refinement=refinements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--input-rate-hz", type=float)
    args = parser.parse_args()
    compute(args.source, run_id=args.run_id, input_rate_hz=args.input_rate_hz)
