"""Measure the saved physical-time replays without executing networks.

Run analyse.py --source COMPUTE_ID to validate input clocks, source controls and
continuous refinement. Export per-population errors and classification summaries
in a separate Pingstore analysis run, using the source's recorded configuration."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import argparse

import numpy as np
from experiments.exp142.custom.inputs import event_digest
from pingstore.library import load_json, source_run, stage_run, write_json_atomic


def analyse(identity, *, run_id=None):
    from experiments.exp142.custom.source import source_recording

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
            if clock["network_dt_ns"] is not None and (
                recording["spike_time_ns"].dtype.kind not in "iu"
                or np.any(recording["spike_time_ns"] % clock["network_dt_ns"])
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


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def decision(scores):
    winners = np.flatnonzero(scores == np.max(scores))
    ordered = np.sort(scores)
    return dict(
        scores=scores.tolist(),
        prediction=int(winners[0]),
        margin=float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else None,
        tie=len(winners) > 1,
        tied_classes=winners.tolist(),
    )


def validate(recording, cfg, ni, *, sample_dt_ns=None):
    sample_dt_ns = cfg["sample_dt_ns"] if sample_dt_ns is None else sample_dt_ns
    g = np.arange(0, cfg["duration_ns"], sample_dt_ns, dtype=np.int64)
    np.testing.assert_array_equal(recording["g_time_ns"], g)
    np.testing.assert_array_equal(recording["v_time_ns"], g + sample_dt_ns)
    n = cfg["n_e"] + ni + cfg["n_output"]
    for key in ("g_exc_ns", "g_inh_ns", "voltage_mv"):
        if recording[key].shape != (len(g), n) or not np.isfinite(recording[key]).all():
            raise ValueError(f"invalid state {key}")
    times, ids = recording["spike_time_ns"], recording["spike_neurons"]
    if (
        times.ndim != 1
        or ids.shape != times.shape
        or not np.isfinite(times).all()
        or ids.dtype.kind not in "iu"
        or np.any(ids < 0)
        or np.any(ids >= n)
        or np.any(times <= 0)
        or np.any(times > cfg["duration_ns"])
        or np.any(np.diff(times) < 0)
    ):
        raise ValueError("invalid spike coordinates")
    for k in range(n):
        if np.any(np.diff(times[ids == k]) <= 0):
            raise ValueError("duplicate or reversed neuron spikes")
    selected = ids >= cfg["n_e"] + ni
    np.testing.assert_array_equal(recording["output_spike_time_ns"], times[selected])
    np.testing.assert_array_equal(
        recording["output_spike_neurons"], ids[selected] - cfg["n_e"] - ni
    )
    counts = np.bincount(recording["output_spike_neurons"], minlength=cfg["n_output"])
    np.testing.assert_array_equal(recording["class_scores"], counts)
    np.testing.assert_array_equal(recording["readout_scores"][-1], counts)
    np.testing.assert_array_equal(
        recording["readout_voltage"], recording["voltage_mv"][:, cfg["n_e"] + ni :]
    )
    cumulative = np.stack(
        [
            np.searchsorted(
                recording["output_spike_time_ns"][
                    recording["output_spike_neurons"] == k
                ],
                recording["v_time_ns"],
                side="right",
            )
            for k in range(cfg["n_output"])
        ],
        axis=1,
    )
    np.testing.assert_array_equal(recording["readout_scores"], cumulative)


def reference_on_clock(reference, current):
    """Select exact saved samples for the native anchor; never interpolate."""
    selected = dict(reference)
    for clock, signals in (
        ("g_time_ns", ("g_exc_ns", "g_inh_ns")),
        ("v_time_ns", ("voltage_mv", "readout_voltage", "readout_scores")),
    ):
        indices = np.searchsorted(reference[clock], current[clock])
        if np.any(indices >= len(reference[clock])) or not np.array_equal(
            reference[clock][indices], current[clock]
        ):
            raise ValueError("reference lacks exact samples on the comparison clock")
        selected[clock] = reference[clock][indices]
        for signal in signals:
            selected[signal] = reference[signal][indices]
    return selected


def compare(current, reference, cfg, ni):
    """Inclusive reset guards; no trace shifts; ordinal timing only if counts match."""
    reference = reference_on_clock(reference, current)
    n = cfg["n_e"] + ni + cfg["n_output"]
    metrics = {
        key: np.full(n, np.nan)
        for key in ("voltage_mae_mv", "ampa_mae_ns", "gaba_mae_ns", "spike_shift_ms")
    }
    coverage = {
        key: np.zeros(n) for key in ("voltage_fraction", "conductance_fraction")
    }
    counts = []
    for recording in (current, reference):
        counts.append(np.bincount(recording["spike_neurons"], minlength=n))
    for k in range(n):
        a, b = [
            r["spike_time_ns"][r["spike_neurons"] == k] for r in (current, reference)
        ]
        for clock, keys, coverage_key in (
            ("v_time_ns", [("voltage_mv", "voltage_mae_mv")], "voltage_fraction"),
            (
                "g_time_ns",
                [("g_exc_ns", "ampa_mae_ns"), ("g_inh_ns", "gaba_mae_ns")],
                "conductance_fraction",
            ),
        ):
            keep = np.ones(len(current[clock]), dtype=bool)
            for spike in np.r_[a, b]:
                keep &= np.abs(current[clock] - spike) > cfg["guard_ms"] * 1e6
            coverage[coverage_key][k] = keep.mean()
            if keep.any():
                for signal, metric in keys:
                    metrics[metric][k] = np.mean(
                        np.abs(
                            current[signal][keep, k].astype(float)
                            - reference[signal][keep, k]
                        )
                    )
        if len(a) == len(b) and len(a):
            metrics["spike_shift_ms"][k] = np.median((a - b) / 1e6)
    populations = {"E": slice(0, cfg["n_e"]), "readout": slice(cfg["n_e"] + ni, n)}
    if ni:
        populations["I"] = slice(cfg["n_e"], cfg["n_e"] + ni)
    summary = {}
    for name, section in populations.items():
        row = {
            "neurons": len(counts[0][section]),
            "spikes": int(counts[0][section].sum()),
            "reference_spikes": int(counts[1][section].sum()),
        }
        for key, values in {**metrics, **coverage}.items():
            finite = values[section][np.isfinite(values[section])]
            row[f"median_{key}"] = float(np.median(finite)) if finite.size else None
            if key == "spike_shift_ms":
                row["timing_eligible_neurons"] = len(finite)
        summary[name] = row
    return summary, {
        **metrics,
        **coverage,
        "counts": counts[0],
        "reference_counts": counts[1],
    }


def refinement_check(a, b, cfg, ni):
    n = cfg["n_e"] + ni + cfg["n_output"]
    np.testing.assert_array_equal(
        np.bincount(a["spike_neurons"], minlength=n),
        np.bincount(b["spike_neurons"], minlength=n),
    )
    np.testing.assert_array_equal(a["class_scores"], b["class_scores"])
    shift = max(
        (
            float(
                np.max(
                    np.abs(
                        a["spike_time_ns"][a["spike_neurons"] == k]
                        - b["spike_time_ns"][b["spike_neurons"] == k]
                    ),
                    initial=0,
                )
            )
            / 1e6
            for k in range(n)
        ),
        default=0,
    )
    _, measurements = compare(a, b, cfg, ni)
    voltage = measurements["voltage_mae_mv"]
    if not np.isfinite(voltage).all():
        raise ValueError("refinement has no retained voltage samples")
    error = float(voltage.max())
    limits = cfg["refinement_acceptance"]
    if (
        shift > limits["max_spike_shift_ms"]
        or error > limits["max_guarded_voltage_mae_mv"]
    ):
        raise ValueError(
            f"unstable continuous reference: root {shift} ms; voltage {error} mV"
        )
    return dict(
        per_neuron_counts_equal=True,
        output_counts_equal=True,
        max_spike_shift_ms=shift,
        max_guarded_voltage_mae_mv=error,
    )


def check_anchor(recording, original, cfg, ni):
    for name, offset in [("E", 0), ("I", cfg["n_e"]), ("readout", cfg["n_e"] + ni)]:
        if name == "I" and not ni:
            continue
        steps, neurons = np.nonzero(original[name])
        selected = (recording["spike_neurons"] >= offset) & (
            recording["spike_neurons"] < offset + original[name].shape[1]
        )
        np.testing.assert_array_equal(
            recording["spike_time_ns"][selected], (steps + 1) * cfg["input_grid_ns"]
        )
        np.testing.assert_array_equal(
            recording["spike_neurons"][selected] - offset, neurons
        )
    np.testing.assert_array_equal(recording["class_scores"], original["scores"])
    return dict(
        all_population_spikes_bitwise_equal=True, class_scores_bitwise_equal=True
    )


def output_clock(source, mode, case, cfg):
    """Read the actual native graph clock; all COBA-LIF populations share it."""
    from pingstore.library import load_json

    if case.startswith("continuous"):
        diagnostics = source.record["execution"]["cases"][f"{mode}-{case}"]
        if "COBA-LIF" not in diagnostics["readout"]:
            raise ValueError("continuous case lacks the continuous COBA-LIF output")
        return dict(network_dt_ns=None, output_dt_ns=None, kind="continuous COBA-LIF")
    expected = (
        cfg["input_grid_ns"]
        if case == "native"
        else next(row["dt_ns"] for row in cfg["cases"] if row["id"] == case)
    )
    snapshot = load_json(source.file(f"{mode}-{case}", "start", "snapshot.json"))
    graph = snapshot["network"]
    populations = {row["id"]: row["neuron"] for row in graph["populations"]}
    if populations["readout"]["kind"] != "coba_lif":
        raise ValueError("expected the trained COBA-LIF readout")
    for actual in (
        graph["timebase"]["dt"]["value"],
        snapshot["execution"]["timestep_ms"],
    ):
        if actual * 1e6 != expected:
            raise ValueError("readout/network timestep differs from the requested case")
    return dict(network_dt_ns=expected, output_dt_ns=expected, kind="discrete COBA-LIF")


def figure_summary(cfg, audit):
    cases = [dict(**row, label=f"{row['dt_ns'] / 1e6:g} ms") for row in cfg["cases"]]
    cases.append(dict(id="continuous", dt_ns=None, label="Continuous"))
    return dict(
        schema="exp142.physical-time-figures/v1",
        configuration=cfg,
        target_e=cfg["target_e"],
        reset_guard_ms=cfg["guard_ms"],
        cases=cases,
        input_image=dict(shape=[8, 8], name="Scikit digits image"),
        native_anchor_label="Native",
        native_anchor_legend="Native 0.2 ms",
        input={
            **audit,
            "plot_annotation": f"{cfg['input_rate_hz']:g} Hz maximum per pixel; {audit['event_count']} saved spikes; exact replay in both models and all {len(audit['cases'])} executions",
        },
        readout_policy="matched-spiking",
        models={},
    )


def add_model(summary, destination, mode, cfg, ni, models, recordings, measurements):
    """Exclude readout neurons from E/I error markers; keep their trained scores."""
    hidden = cfg["n_e"] + ni
    alias = dict(
        g_exc_guarded_mae_ns="ampa_mae_ns",
        g_inh_guarded_mae_ns="gaba_mae_ns",
        v_reset_aware_mae_mv="voltage_mae_mv",
        spike_median_shift_ms="spike_shift_ms",
        g_retained_sample_fraction="conductance_fraction",
        v_retained_sample_fraction="voltage_fraction",
    )
    rows = []
    for case in [
        dict(id="native", dt_ns=cfg["input_grid_ns"], label="Native 0.2 ms"),
        *summary["cases"],
    ]:
        name = case["id"]
        recording, metrics = recordings[name], measurements[name]
        populations = {
            key: value
            for key, value in models[name]["populations"].items()
            if key in ("E", "I")
        }
        if not ni:
            populations["I"] = dict(neurons=0, spikes=0, timing_eligible_neurons=0)
        rows.append(
            dict(
                **case,
                classification=models[name]["classification"],
                populations=populations,
                spike_count=sum(row["spikes"] for row in populations.values()),
            )
        )
        stem = f"{mode}__{name}__"
        for key in ("g_time_ns", "v_time_ns", "readout_scores", "class_scores"):
            destination[stem + key] = recording[key]
        for key in ("g_exc_ns", "g_inh_ns", "voltage_mv"):
            destination[stem + key] = recording[key][:, :hidden]
        selected = recording["spike_neurons"] < hidden
        destination[stem + "spike_time_ns"] = recording["spike_time_ns"][selected]
        destination[stem + "spike_neurons"] = recording["spike_neurons"][selected]
        for output, input_key in alias.items():
            destination[stem + output] = metrics[input_key][:hidden]
        # Preserve full root resolution. Rendering never rounds these events.
        if (
            name == "continuous"
            and destination[stem + "spike_time_ns"].dtype.kind != "f"
        ):
            raise ValueError("continuous roots must remain floating point")
    summary["models"][mode] = dict(native_anchor=rows[0], cases=rows[1:], n_i=ni)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    analyse(args.source, run_id=args.run_id)
