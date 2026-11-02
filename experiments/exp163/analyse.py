"""Validate spatial checkpoint weights and measure cue-dependent output patterns."""

import argparse
import copy
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
import torch
from experiments.exp163 import recipe
from pingstore.library import source_run, stage_run, write_json_atomic
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from snnlab.meanfield import Trajectory
from snnlab.sim import NetworkSnapshot, RecordingData


def read_trial(computed, cfg):
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    with np.load(computed.file("spikes.npz")) as d:
        spikes = dict(d)
    if set(spikes) != {c + "_" + p for c in "ABC" for p in "EI"}:
        raise ValueError("missing population spikes")
    state = {}
    for key, v in spikes.items():
        expected = (steps, 1, cfg["network"]["populations"][key[-1]]["size"])
        if (
            v.shape != expected
            or v.dtype != np.uint8
            or not ((v == 0) | (v == 1)).all()
        ):
            raise ValueError("invalid spike array")
        state[key + ".rate"] = gaussian_filter1d(
            v[:, 0, :].mean(axis=1) * 1000 / cfg["dt_ms"],
            cfg["rate_smoothing_ms"] / cfg["dt_ms"],
        )
    recording = RecordingData.read(computed.unit("conductances"))
    if set(recording.signals) != set(cfg["recording_signals"]):
        raise ValueError("missing native conductance")
    for key, v in recording.signals.items():
        info = recording.info[key]
        if (
            info.unit != "uS"
            or info.dt_ms != cfg["dt_ms"]
            or v.shape != (steps, 1, cfg["network"]["populations"]["E"]["size"])
        ):
            raise ValueError("invalid native coordinates or units")
        if not np.isfinite(v).all() or (v < 0).any():
            raise ValueError("invalid conductance values")
        state[key] = v[:, 0, :].mean(axis=1)
    t = np.arange(steps) * cfg["dt_ms"]
    if (
        state["cue_B.conductance"][t < cfg["switch_ms"]].max() > 1e-8
        or state["cue_A.conductance"][t > cfg["switch_ms"] + 100].max() > 1e-8
    ):
        raise ValueError("cue did not switch targets")
    return spikes, Trajectory(t, state)


def measure(sol, cfg):
    t = sol.time_ms
    s = sol.state
    peaks = {}
    phases = {}
    for c in "ABC":
        y = np.array(s[c + "_E.rate"])
        p, _ = find_peaks(
            y,
            distance=round(cfg["phase"]["min_peak_distance_ms"] / cfg["dt_ms"]),
            prominence=max(
                cfg["phase"]["min_prominence_hz"],
                cfg["phase"]["prominence_fraction"]
                * np.ptp(y[t > cfg["timing"]["reference_start_ms"]]),
            ),
        )
        peaks[c] = p
        if len(p) < 2:
            return dict(
                switched=False, reason="insufficient observed volleys", epochs=[]
            ), dict(
                time=t, phases={k: np.full(t.shape, np.nan) for k in "ABC"}, peaks=peaks
            )
        phases[c] = np.interp(t, t[p], np.arange(len(p)), left=np.nan, right=np.nan)
    result = []
    for (start, end), winner in zip(cfg["windows_ms"], "AB", strict=True):
        w = (t >= start) & (t <= end)
        row = dict(window_ms=[start, end], cued=winner, senders={})
        for c in "ABC":
            p = peaks[c]
            pt = t[p[(t[p] >= start) & (t[p] <= end)]]
            row[c] = dict(
                mean_rate_hz=float(np.mean(s[c + "_E.rate"][w])),
                frequency_hz=float(1000 / np.mean(np.diff(pt)))
                if len(pt) > 1
                else None,
                amplitude_hz=float(np.ptp(s[c + "_E.rate"][w])),
                volleys=len(pt),
                interval_cv=float(np.std(np.diff(pt)) / np.mean(np.diff(pt)))
                if len(pt) > 2
                else None,
            )
        for c in "AB":
            valid = w & np.isfinite(phases[c]) & np.isfinite(phases["C"])
            delta = (phases[c] - phases["C"])[valid]
            row["senders"][c] = dict(
                plv=float(abs(np.mean(np.exp(2j * np.pi * delta)))),
                drift_cycles_s=float(np.polyfit(t[valid] / 1000, delta, 1)[0]),
                phase_span_cycles=float(np.ptp(delta)),
            )
        other = "B" if winner == "A" else "A"
        win = row["senders"][winner]
        lose = row["senders"][other]
        row["passed"] = bool(
            win["plv"] > cfg["phase"]["min_plv"]
            and abs(win["drift_cycles_s"]) < cfg["phase"]["max_abs_drift_cycles_s"]
            and win["phase_span_cycles"] < cfg["phase"]["max_cued_phase_span_cycles"]
            and lose["phase_span_cycles"] > cfg["phase"]["min_uncued_phase_span_cycles"]
            and row["C"]["amplitude_hz"] > cfg["phase"]["min_receiver_amplitude_hz"]
        )
        row["rhythms_valid"] = all(
            row[c]["volleys"] >= cfg["rhythm_criteria"]["min_volleys"]
            and row[c]["frequency_hz"] is not None
            and cfg["rhythm_criteria"]["frequency_band_hz"][0]
            <= row[c]["frequency_hz"]
            <= cfg["rhythm_criteria"]["frequency_band_hz"][1]
            and row[c]["interval_cv"] is not None
            and row[c]["interval_cv"] <= cfg["rhythm_criteria"]["max_interval_cv"]
            for c in "ABC"
        )
        row["passed"] = row["passed"] and row["rhythms_valid"]
        result.append(row)
    return dict(epochs=result, switched=all(r["passed"] for r in result)), dict(
        time=t, phases=phases, peaks=peaks
    )


def spatial_measure(spikes, t, cfg, metrics):
    x = recipe.coordinates(cfg)
    centers = cfg["spatial"]["centers"]
    masks = {
        c: abs(x - center) <= cfg["spatial"]["roi_half_width"]
        for c, center in centers.items()
    }
    rows = []
    passed = []
    for epoch in metrics["epochs"]:
        start, end = epoch["window_ms"]
        w = (t >= start) & (t <= end)
        profiles = {
            c: spikes[c + "_E"][w, 0, :].mean(axis=0) * 1000 / cfg["dt_ms"]
            for c in "ABC"
        }

        def contrast(y):
            a, b = [float(y[masks[c]].mean()) for c in "AB"]
            return (a - b) / (a + b) if a + b else 0.0

        out = {c + "_contrast": contrast(profiles[c]) for c in "ABC"}
        winner = epoch["cued"]
        expected = centers[winner]
        out["C_peak_position"] = float(x[np.argmax(profiles["C"])])
        out["C_peak_error"] = abs(out["C_peak_position"] - expected)
        out["passed"] = bool(
            out["A_contrast"] >= cfg["spatial"]["min_sender_contrast"]
            and out["B_contrast"] <= -cfg["spatial"]["min_sender_contrast"]
            and (out["C_contrast"] if winner == "A" else -out["C_contrast"])
            >= cfg["spatial"]["min_receiver_contrast"]
            and out["C_peak_error"] <= cfg["spatial"]["max_peak_error"]
        )
        epoch["spatial"] = out
        rows.append(profiles)
        passed.append(out["passed"])
    metrics["spatial_switched"] = bool(passed and all(passed))
    metrics["prototype_passed"] = metrics["switched"] and metrics["spatial_switched"]
    return rows


def raster_summary(spikes, cfg, windows):
    """Raw-spike diversity and recruitment in complete E-defined population cycles."""
    dt = cfg["dt_ms"]
    rows = []
    for batch in range(next(iter(spikes.values())).shape[1]):
        for start, end in windows:
            row = dict(batch=batch, window_ms=[start, end], populations={})
            for c in "ABC":
                e = spikes[c + "_E"][:, batch, :]
                rate = gaussian_filter1d(e.mean(1) * 1000 / dt, 1 / dt)
                peaks, _ = find_peaks(
                    rate,
                    distance=round(15 / dt),
                    prominence=max(
                        0.05, 0.1 * np.ptp(rate[round(start / dt) : round(end / dt)])
                    ),
                )
                boundaries = (peaks[1:] + peaks[:-1]) / 2
                boundaries = boundaries[
                    (boundaries * dt >= start) & (boundaries * dt <= end)
                ]
                for p in "EI":
                    x = spikes[c + "_" + p][:, batch, :]
                    late = x[round(start / dt) : round(end / dt)]
                    counts = np.array(
                        [
                            x[round(a) : round(b)].sum(0)
                            for a, b in zip(
                                boundaries[:-1], boundaries[1:], strict=True
                            )
                        ]
                    ).reshape(-1, x.shape[1])
                    active = late.sum(0) > 0
                    participation = (counts > 0).mean(1)
                    isi_cvs = []
                    for train in late.T:
                        intervals = np.diff(np.flatnonzero(train)) * dt
                        if len(intervals) >= 2:
                            isi_cvs.append(float(intervals.std() / intervals.mean()))
                    row["populations"][c + "_" + p] = dict(
                        mean_hz=float(late.mean() * 1000 / dt),
                        active_fraction=float(active.mean()),
                        distinct_train_fraction=float(
                            len(np.unique(np.packbits(late.T, axis=1), axis=0))
                            / x.shape[1]
                        ),
                        max_same_bin_fraction=float(late.mean(1).max()),
                        complete_cycles=len(counts),
                        mean_cycle_participation=float(participation.mean())
                        if len(counts)
                        else None,
                        fully_recruited_cycle_fraction=float(
                            (participation == 1).mean()
                        )
                        if len(counts)
                        else None,
                        cycle_skipping_cell_fraction=float(
                            ((counts == 0).any(0) & active).mean()
                        ),
                        median_isi_cv=float(np.median(isi_cvs)) if isi_cvs else None,
                    )
            rows.append(row)
    return rows


def validate_mirror(computed, mirror, original, reference_cfg, cfg, spikes, profiles):
    mirror_cfg = mirror.record["execution"]["configuration"]
    swapped = copy.deepcopy(cfg)
    swapped["spatial"]["centers"] = {
        c: 1 - v for c, v in cfg["spatial"]["centers"].items()
    }
    if swapped != mirror_cfg:
        raise ValueError("mirror control differs beyond spatial reflection")
    mspikes, msol = read_trial(mirror, mirror_cfg)
    with (
        np.load(computed.file("inputs.npz")) as a,
        np.load(mirror.file("inputs.npz")) as b,
    ):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    expected = recipe.spatial_weights(original, mirror_cfg, reference_cfg)
    actual = NetworkSnapshot.read(
        mirror.file("execution/start/snapshot.json")
    ).load_parameters()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    metrics, _ = measure(msol, mirror_cfg)
    mprofiles = spatial_measure(mspikes, msol.time_ms, mirror_cfg, metrics)
    exact_spikes = all(
        np.array_equal(spikes[key][:, :, ::-1], value) for key, value in mspikes.items()
    )
    exact_profiles = all(
        np.array_equal(a[c][::-1], b[c])
        for a, b in zip(profiles, mprofiles, strict=True)
        for c in "ABC"
    )
    correlations = [
        {c: float(np.corrcoef(a[c][::-1], b[c])[0, 1]) for c in "ABC"}
        for a, b in zip(profiles, mprofiles, strict=True)
    ]
    fixed_heterogeneity = "heterogeneity" in cfg and not cfg["heterogeneity"].get(
        "reflect_with_centers", True
    )
    if fixed_heterogeneity:
        forward_arrays = NetworkSnapshot.read(
            computed.file("execution/start/snapshot.json")
        ).load_parameters()
        for key in actual:
            if key not in ("sensory_A.weight", "sensory_B.weight"):
                np.testing.assert_array_equal(actual[key], forward_arrays[key])
    elif "heterogeneity" not in cfg:
        if not exact_spikes or not exact_profiles:
            raise ValueError(
                "homogeneous mirror did not reproduce exact spikes/profiles"
            )
    elif not metrics["prototype_passed"] or not all(
        v >= 0.95 for row in correlations for v in row.values()
    ):
        raise ValueError(
            "heterogeneous mirror failed task criteria or reflected-profile agreement"
        )
    mirror.check_unchanged()
    return dict(
        validated=True,
        passed=bool(metrics["prototype_passed"]),
        control_type="centres swapped with fixed heterogeneity"
        if fixed_heterogeneity
        else "complete network reflection",
        nonsensory_weights_held_fixed=fixed_heterogeneity,
        source_run=mirror.directory.name,
        metrics=metrics,
        spikes_and_profiles_exactly_mirrored=exact_spikes and exact_profiles,
        reflected_profile_correlations=correlations,
        interpretation="task outcomes checked with identical inputs; full reflection also checks numerical permutation symmetry",
    )


def analyse(*, source, run_id=None, mirror_source=None):
    started = perf_counter()
    torch.set_num_threads(1)
    computed = source_run(
        REPO / ".pingstore", source, experiment=recipe.SLUG, stage="compute"
    )
    cfg = computed.record["execution"]["configuration"]
    if cfg["schema"] != recipe.SCHEMA:
        raise ValueError("unsupported recipe")
    spikes, sol = read_trial(computed, cfg)
    pin = computed.record["inputs"]["parameter_reference"]
    reference = source_run(
        REPO / ".pingstore",
        pin["run_id"],
        reference=pin,
        experiment="exp162",
        stage="compute",
    )
    original = NetworkSnapshot.read(
        reference.file("execution/start/snapshot.json")
    ).load_parameters()
    expected = recipe.spatial_weights(
        original, cfg, reference.record["execution"]["configuration"]
    )
    actual = NetworkSnapshot.read(
        computed.file(recipe.SNAPSHOT + "/snapshot.json")
    ).load_parameters()
    if set(actual) != set(expected):
        raise ValueError("native parameter identities differ")
    for key in actual:
        np.testing.assert_array_equal(actual[key].numpy(), expected[key].numpy())
    metrics, diag = measure(sol, cfg)
    profiles = spatial_measure(spikes, sol.time_ms, cfg, metrics)
    mirror = None
    if mirror_source:
        mirror = source_run(
            REPO / ".pingstore", mirror_source, experiment=recipe.SLUG, stage="compute"
        )
    elif "mirror_reference" in computed.record["inputs"]:
        pin = computed.record["inputs"]["mirror_reference"]
        mirror = source_run(
            REPO / ".pingstore",
            pin["run_id"],
            reference=pin,
            experiment="exp162",
            phase="scratch-spatial-competition",
        )
    mirror_result = (
        validate_mirror(
            computed,
            mirror,
            original,
            reference.record["execution"]["configuration"],
            cfg,
            spikes,
            profiles,
        )
        if mirror
        else None
    )
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs=dict(compute=computed, **({"mirror": mirror} if mirror else {})),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        write_json_atomic(
            run.export / "results.json",
            dict(
                schema="exp163.analysis/v1",
                metrics=metrics,
                mirror_control=mirror_result,
                rasters=raster_summary(spikes, cfg, cfg["windows_ms"]),
            ),
        )
        np.savez_compressed(
            run.export / "phase.npz",
            time_ms=sol.time_ms,
            **{c + "_phase": v for c, v in diag["phases"].items()},
        )
        np.savez_compressed(
            run.export / "spatial-profiles.npz",
            position=recipe.coordinates(cfg),
            **{
                f"epoch_{i}_{c}": v
                for i, row in enumerate(profiles)
                for c, v in row.items()
            },
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
        computed.check_unchanged()
        reference.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    p.add_argument("--mirror-source")
    analyse(**vars(p.parse_args()))
