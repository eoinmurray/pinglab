"""Validate recordings and compare neurons on physical clocks.

Measure conductance and voltage errors outside spike guards, pair ordinal spikes
only for equal nonzero counts, and enforce continuous-reference refinement.
These functions consume saved arrays and never execute a network."""

import numpy as np


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
