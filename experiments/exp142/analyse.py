"""Measure the saved physical-time replays without executing networks.

Run analyse.py --source COMPUTE_ID to validate input clocks, source controls and
continuous refinement. Export per-population errors and classification summaries
in a separate Pingstore analysis run, using the source's recorded configuration."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp142.custom.inputs import event_digest
from experiments.exp142.custom.metrics import (
    arrays,
    check_anchor,
    compare,
    decision,
    refinement_check,
    validate,
)
from experiments.exp142.custom.plot_data import add_model, figure_summary, output_clock
from experiments.exp142.custom.source import source_recording
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def analyse(identity, *, run_id=None):
    source = source_run(
        REPO / ".pingstore", identity, experiment="exp142", stage="compute"
    )
    # Interpret the evidence from its recorded recipe, including older retained runs.
    cfg = source.record["execution"]["configuration"]
    if cfg["schema"] not in ("exp142.recipe/v4", "exp142.recipe/v5"):
        raise ValueError("incompatible physical-time protocol")
    audit = load_json(source.file("audit.json"))
    cases = (
        ["native"]
        + [row["id"] for row in cfg["cases"]]
        + ["continuous", "continuous-refined"]
    )
    expected = {f"{mode}-{case}" for mode in cfg["models"] for case in cases}
    if set(audit["cases"]) != expected or any(
        not row["input_equivalent"]
        or row["event_count"] != audit["event_count"]
        or row["input_event_sha256"] != audit["input_event_sha256"]
        or row["max_time_error_ns"] != 0
        for row in audit["cases"].values()
    ):
        raise ValueError("incomplete input replay audit")
    checks = load_json(source.file("checks.json"))
    trained_pin = source.record["inputs"]["trained"]
    trained = source_run(
        REPO / ".pingstore",
        trained_pin["run_id"],
        experiment="exp141",
        stage="compute",
        reference=trained_pin,
    )
    encoding = source.record["execution"]["input_encoding"]
    if (
        audit["input_rate_hz"] != cfg["input_rate_hz"]
        or encoding["protocol"]["dataset_binding"]["encoder"]["max_rate_hz"]
        != cfg["input_rate_hz"]
        or not encoding["source_encoding_bitwise_equal"]
    ):
        raise ValueError("input rate or encoder source control mismatch")
    source_input = arrays(source.file("source-input.npz"))
    if event_digest(source_input) != audit["source_event_sha256"]:
        raise ValueError("original source replay input checksum differs")
    traces, models, output_clocks, plot_arrays = {}, {}, {}, {}
    plots = figure_summary(cfg, audit)
    for mode in cfg["models"]:
        ni = cfg["n_i"] if mode == "ping" else 0
        recordings = {
            case: arrays(source.file(f"{mode}-{case}", "recording.npz"))
            for case in cases
        }
        output_clocks[mode] = {}
        for case, recording in recordings.items():
            validate(
                recording,
                cfg,
                ni,
                sample_dt_ns=cfg["native_sample_dt_ns"] if case == "native" else None,
            )
            clock = output_clock(source, mode, case, cfg)
            output_clocks[mode][case] = clock
            if clock["network_dt_ns"] is not None:
                if recording["spike_time_ns"].dtype.kind not in "iu" or np.any(
                    recording["spike_time_ns"] % clock["network_dt_ns"]
                ):
                    raise ValueError(
                        "discrete spikes do not use the declared network/output clock"
                    )
        refinements = refinement_check(
            recordings["continuous"], recordings["continuous-refined"], cfg, ni
        )
        control_diagnostics = source.record["execution"]["cases"][
            f"{mode}-source-control"
        ]
        if control_diagnostics["input_event_sha256"] != audit["source_event_sha256"]:
            raise ValueError("source replay did not use the original 25 Hz encoding")
        control = arrays(source.file(f"{mode}-source-control", "recording.npz"))
        validate(control, cfg, ni, sample_dt_ns=cfg["native_sample_dt_ns"])
        reproduction = check_anchor(control, source_recording(trained, mode), cfg, ni)
        if (
            refinements != checks["refinement"][mode]
            or reproduction != checks["anchors"][mode]
        ):
            raise ValueError("invalid anchor or refinement check")
        models[mode], measurements = {}, {}
        for case in cases[:-1]:
            recording = recordings[case]
            populations, metrics = compare(recording, recordings["continuous"], cfg, ni)
            models[mode][case] = dict(
                classification=decision(recording["class_scores"]),
                populations=populations,
            )
            measurements[case] = metrics
            traces.update(
                {
                    f"{mode}__{case}__{key}": value
                    for key, value in {**recording, **metrics}.items()
                }
            )
        add_model(
            plots, plot_arrays, mode, cfg, ni, models[mode], recordings, measurements
        )
    with stage_run(
        REPO,
        "exp142",
        "analyse",
        inputs={"compute": source},
        configuration={
            "guard_ms": cfg["guard_ms"],
            "reference": "continuous",
            "schema": "exp142.measurement/v4",
            "output_clock": "native graph timestep equals case timestep; CT includes output roots",
            "pairing": "ordinal, equal nonzero counts; silent neurons ineligible",
        },
        run_id=run_id,
    ) as run:
        write_json_atomic(
            run.export / "numbers.json",
            dict(
                configuration=cfg,
                input=audit,
                checks=checks,
                models=models,
                output_clocks=output_clocks,
                plots=plots,
            ),
        )
        np.savez_compressed(run.export / "comparison.npz", **traces)
        np.savez_compressed(run.export / "plot-comparison.npz", **plot_arrays)
        np.savez_compressed(
            run.export / "input.npz", **arrays(source.file("input.npz"))
        )
        for mode in cfg["models"]:
            (run.export / f"{mode}-circuit.svg").write_bytes(
                source.file(
                    f"{mode}-{cfg['cases'][0]['id']}", "circuit.svg"
                ).read_bytes()
            )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    analyse(args.source, run_id=args.run_id)
