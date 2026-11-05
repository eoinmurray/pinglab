"""Validate imported or newly computed spatial relay evidence without simulation."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp172.recipe import gaussian
from pingstore.library import source_run, stage_run
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares, nnls
from scipy.signal import find_peaks
from snnlab.sim import NetworkSnapshot


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fit_profile(y, centres, sigma, minimum):
    x = np.arange(len(y)) / len(y)
    templates = np.column_stack([gaussian(x, c, sigma) for c in centres])
    design = np.column_stack([np.ones(len(y)), templates])
    coefficients, _ = nnls(design, y)
    prediction = np.sum(design * coefficients[None, :], axis=1)
    denom = np.sum((y - y.mean()) ** 2)
    amplitudes = coefficients[1:]
    total = float(amplitudes.sum())
    # Shape recovery is a diagnostic, not a second choice of ER estimator.
    p0 = [*coefficients, *centres, sigma, sigma]
    lower = [0, 0, 0, centres[0] - 0.08, centres[1] - 0.08, 0.035, 0.035]
    upper = [np.inf, np.inf, np.inf, centres[0] + 0.08, centres[1] + 0.08, 0.18, 0.18]

    def residual(p):
        return (
            p[0] + p[1] * gaussian(x, p[3], p[5]) + p[2] * gaussian(x, p[4], p[6]) - y
        )

    shape = least_squares(residual, p0, bounds=(lower, upper))
    return (
        dict(
            background_hz=float(coefficients[0]),
            amplitudes_hz=amplitudes.tolist(),
            total_amplitude_hz=total,
            fraction_A=float(amplitudes[0] / total) if total >= minimum else None,
            r_squared=float(1 - np.sum((y - prediction) ** 2) / denom)
            if denom > 0
            else None,
            rmse_hz=float(np.sqrt(np.mean((y - prediction) ** 2))),
            recovered_centres=shape.x[3:5].tolist(),
            recovered_sigmas=shape.x[5:7].tolist(),
            shape_fit_success=bool(shape.success),
        ),
        prediction,
        templates,
    )


def copy_fidelity(target, output, n=8):
    """Direct spatial-rate reconstruction, with no fitted gain or offset."""
    target = np.asarray(target).reshape(-1, n).mean(1)
    output = np.asarray(output).reshape(-1, n).mean(1)
    mse = float(np.mean((output - target) ** 2))
    variance = float(np.var(target))
    return dict(
        bin_neurons=n,
        copy_r_squared=1 - mse / variance if variance > 0 else None,
        rmse_hz=float(np.sqrt(mse)),
        correlation=float(np.corrcoef(target, output)[0, 1])
        if np.std(target) > 0 and np.std(output) > 0
        else None,
        mean_rate_ratio=float(output.mean() / target.mean())
        if target.mean() > 0
        else None,
    )


def phase_metrics(spikes, cfg, start, end):
    dt = cfg["dt_ms"]
    t = np.arange(len(spikes["A_E"])) * dt
    phases, frequencies, rhythms = {}, {}, {}
    for c in "ABC":
        rate = gaussian_filter1d(spikes[c + "_E"].mean(1) * 1000 / dt, 1 / dt)
        peaks, _ = find_peaks(
            rate,
            distance=round(15 / dt),
            prominence=max(0.05, 0.1 * np.ptp(rate[t > 500])),
        )
        times = t[peaks]
        if len(times) < 2:
            continue
        phases[c] = np.interp(
            t, times, np.arange(len(times)), left=np.nan, right=np.nan
        )
        local = times[(times >= start) & (times < end)]
        frequencies[c] = (
            float(1000 / np.mean(np.diff(local))) if len(local) > 2 else None
        )
        intervals = np.diff(local)
        cv = float(intervals.std() / intervals.mean()) if len(intervals) > 1 else None
        rhythms[c] = dict(
            volleys=len(local),
            interval_cv=cv,
            valid=bool(
                len(local) >= 12
                and cv is not None
                and cv < 0.2
                and 30 <= frequencies[c] <= 80
            ),
        )
    plv = {}
    for c in "AB":
        if c in phases and "C" in phases:
            valid = (
                (t >= start)
                & (t < end)
                & np.isfinite(phases[c])
                & np.isfinite(phases["C"])
            )
            plv[c] = (
                float(
                    abs(
                        np.mean(
                            np.exp(2j * np.pi * (phases[c][valid] - phases["C"][valid]))
                        )
                    )
                )
                if valid.any()
                else None
            )
    return dict(
        frequencies_hz=frequencies,
        rhythms=rhythms,
        plv=plv,
        caveat="diagnostic peak-based phase; no CTC mechanism claim",
    )


def analyse(spikes, cfg, swap):
    centres = cfg["centres"][::-1] if swap else cfg["centres"]
    profiles, epochs = {}, []
    for epoch, (start, end) in enumerate(cfg["windows_ms"]):
        sl = slice(round(start / cfg["dt_ms"]), round(end / cfg["dt_ms"]))
        rates = {
            key: value[sl].mean(0) * 1000 / cfg["dt_ms"]
            for key, value in spikes.items()
        }
        fits = {}
        for c in "ABC":
            sigma = np.sqrt(
                cfg["spatial_sigma"] ** 2
                + cfg["sensory_kernel_sigma"] ** 2
                + (cfg["cross_kernel_sigma"] ** 2 if c == "C" else 0)
            )
            fits[c], _, _ = fit_profile(
                rates[c + "_E"], centres, sigma, cfg["minimum_total_amplitude_hz"]
            )
        fraction = fits["C"]["fraction_A"]
        attended = "AB"[epoch]
        er = (
            (fraction if epoch == 0 else 1 - fraction) if fraction is not None else None
        )
        fidelity = copy_fidelity(
            rates[attended + "_E"], rates["C_E"], cfg.get("fidelity_bin_neurons", 8)
        )
        epochs.append(
            dict(
                attended=attended,
                window_ms=[start, end],
                er=er,
                fits=fits,
                fidelity=fidelity,
                mean_rates_hz={k: float(v.mean()) for k, v in rates.items()},
                phase=phase_metrics(spikes, cfg, start, end),
            )
        )
        profiles.update({f"epoch{epoch}_{k}": v for k, v in rates.items()})
    return epochs, profiles


def spatial_timecourse(spikes, cfg, swap, window_ms=100.0, hop_ms=5.0):
    """Fit nonnegative background + fixed A/B templates in overlapping windows."""
    dt = cfg["dt_ms"]
    s = spikes["C_E"]
    width = round(window_ms / dt)
    hop = round(hop_ms / dt)
    x = np.arange(s.shape[1]) / s.shape[1]
    centres = cfg["centres"][::-1] if swap else cfg["centres"]
    sigma = np.sqrt(
        cfg["spatial_sigma"] ** 2
        + cfg["sensory_kernel_sigma"] ** 2
        + cfg["cross_kernel_sigma"] ** 2
    )
    design = np.column_stack(
        [np.ones(len(x)), *[gaussian(x, c, sigma) for c in centres]]
    )
    cumulative = np.vstack(
        [np.zeros((1, s.shape[1])), np.cumsum(s, axis=0, dtype=np.float64)]
    )
    rows = []
    for start in range(0, len(s) - width + 1, hop):
        end = start + width
        y = (cumulative[end] - cumulative[start]) * 1000 / (width * dt)
        coef, _ = nnls(design, y)
        prediction = np.sum(design * coef[None, :], axis=1)
        variance = np.sum((y - y.mean()) ** 2)
        rows.append(
            dict(
                time_ms=(start + end) * dt / 2,
                background_hz=float(coef[0]),
                A_amplitude_hz=float(coef[1]),
                B_amplitude_hz=float(coef[2]),
                fit_R2=float(1 - np.sum((y - prediction) ** 2) / variance)
                if variance > 0
                else None,
            )
        )
    return dict(
        window_ms=width * dt,
        hop_ms=hop * dt,
        method="NNLS: nonnegative uniform background + fixed unit-peak A/B Gaussian templates; overlapping windows, no extra temporal smoothing; coefficients are pattern amplitudes, not causal source attribution",
        template_sigma=float(sigma),
        centres=centres,
        points=rows,
    )


def stability_metrics(spikes, cfg, swap=False):
    """Quantify spatial competition without hiding excursions behind long smoothing."""
    settings = cfg["stability_search"]
    course = spatial_timecourse(
        spikes, cfg, swap, settings["fit_window_ms"], settings["fit_hop_ms"]
    )
    rows = course["points"]
    times = np.array([r["time_ms"] for r in rows])
    amplitudes = np.array([[r["A_amplitude_hz"], r["B_amplitude_hz"]] for r in rows])
    total = amplitudes.sum(1)
    resolved = total >= cfg["minimum_total_amplitude_hz"]
    dominance = amplitudes[:, 0] - amplitudes[:, 1]
    windows = []
    for i, (start, end) in enumerate(settings["windows_ms"]):
        mask = (times >= start) & (times < end)
        attended = amplitudes[mask, i]
        other = amplitudes[mask, 1 - i]
        usable = resolved[mask]
        wrong = (other >= attended) | ~usable
        runs = []
        length = 0
        for failed in wrong:
            if failed:
                length += 1
            elif length:
                runs.append(length)
                length = 0
        if length:
            runs.append(length)
        windows.append(
            dict(
                attended="AB"[i],
                window_ms=[start, end],
                wrong_or_unresolved_fraction=float(wrong.mean()),
                leakage_fraction=float(
                    other.sum() / max((attended + other).sum(), 1e-12)
                ),
                maximum_excursion_centres_ms=float(
                    max(runs, default=0) * course["hop_ms"]
                ),
                unresolved_fraction=float((~usable).mean()),
                mean_attended_amplitude_hz=float(attended.mean()),
            )
        )
    # Use only post-switch observations; report window ends to avoid look-ahead.
    half = course["window_ms"] / 2
    eligible = (times - half >= cfg["switch_ms"]) & resolved
    need = round(settings["sustained_dominance_ms"] / course["hop_ms"])
    latency = None
    for i in np.flatnonzero(eligible):
        if i + need > len(times):
            break
        if np.all(eligible[i : i + need] & (dominance[i : i + need] < 0)):
            latency = float(times[i] + half - cfg["switch_ms"])
            break
    return dict(
        window_ms=course["window_ms"],
        hop_ms=course["hop_ms"],
        epochs=windows,
        worst_window_wrong_fraction=max(
            w["wrong_or_unresolved_fraction"] for w in windows
        ),
        mean_wrong_fraction=float(
            np.mean([w["wrong_or_unresolved_fraction"] for w in windows])
        ),
        switch_latency_ms=latency,
        caveat="Overlapping fits are correlated; fitted spatial identity does not establish input origin. Latency is the end of the first exclusively post-switch fit beginning a 100 ms run of B dominance.",
    )


def materialise_compute(source, destination):
    """Restore canonical native paths into disposable stage work, never into a run."""
    destination.mkdir(parents=True, exist_ok=False)
    reverse = {
        stored: original
        for original, stored in source.record.get("scientific_artifacts", {})
        .get("native_paths", {})
        .items()
    }
    for path in source.outputs.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source.outputs).as_posix()
        target = destination / reverse.get(relative, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return destination


def compare_metrics(actual, expected):
    """Require recomputation to agree with retained original measurements."""
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            raise ValueError("Metric keys differ")
        for key, value in actual.items():
            compare_metrics(value, expected[key])
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError("Metric lengths differ")
        for a, b in zip(actual, expected, strict=False):
            compare_metrics(a, b)
    elif isinstance(actual, (float, int)) and not isinstance(actual, bool):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
    elif actual != expected:
        raise ValueError(f"Metric mismatch: {actual!r} != {expected!r}")


def validate_parameters(directory, cfg):
    params = NetworkSnapshot.read(directory / "start/snapshot.json").load_parameters()
    for value in params.values():
        if not np.isfinite(value.numpy()).all() or (value < 0).any():
            raise ValueError("Invalid weights")
    np.testing.assert_array_equal(
        params["A_E_to_C_E.weight"], params["B_E_to_C_E.weight"]
    )
    if cfg.get("cross_total_us") is not None:
        np.testing.assert_allclose(
            params["A_E_to_C_E.weight"].sum(0), cfg["cross_total_us"], rtol=2e-6
        )
    if cfg.get("c_ee_scale") == 0 and params["C_E_to_E.weight"].count_nonzero():
        raise ValueError("C recurrence enabled")


def validate(data):
    cfg = json.loads((data / "configuration.json").read_text())
    summary = json.loads((data / "summary.json").read_text())
    trials = summary["trials"]
    if len({r["id"] for r in trials}) != len(trials):
        raise ValueError("Duplicate trial")
    expected = {
        (seed, swap, cue, "AB")
        for seed in cfg["seeds"]
        for swap in cfg["spatial_assignments"]
        for cue in cfg["cue_strengths_hz"]
    }
    expected |= {
        (cfg["seeds"][0], swap, 0.0, c)
        for swap in cfg["spatial_assignments"]
        for c in "AB"
    }
    actual = {(r["seed"], r["swap"], r["cue_hz"], r["active"]) for r in trials}
    if actual != expected:
        raise ValueError("Incomplete trial grid")
    for trial in trials:
        directory = data / trial["id"]
        local = json.loads((directory / "configuration.json").read_text())
        if local["trial"] != {
            k: trial[k] for k in ["id", "seed", "swap", "cue_hz", "active"]
        }:
            raise ValueError("Trial configuration disagrees")
        for key in cfg:
            if key in local:
                compare_metrics(local[key], cfg[key])
        with np.load(directory / "spikes.npz") as z:
            spikes = dict(z)
        if set(spikes) != {c + "_" + p for c in "ABC" for p in "EI"}:
            raise ValueError("Missing spikes")
        for key, value in spikes.items():
            shape = (
                round(cfg["duration_ms"] / cfg["dt_ms"]),
                cfg["network"]["populations"][key[-1]]["size"],
            )
            if value.shape != shape or not np.isin(value, [0, 1]).all():
                raise ValueError("Invalid spikes")
        epochs, profiles = analyse(spikes, cfg, trial["swap"])
        compare_metrics(epochs, trial["epochs"])
        if "stability" in trial:
            compare_metrics(
                stability_metrics(spikes, cfg, trial["swap"]), trial["stability"]
            )
        compare_metrics(json.loads((directory / "metrics.json").read_text()), trial)
        with np.load(directory / "profiles.npz") as z:
            for key, value in profiles.items():
                np.testing.assert_array_equal(value, z[key])
        validate_parameters(directory, cfg)
    return cfg, summary


def main(source, run_id=None):
    started = perf_counter()
    computed = source_run(
        REPO / ".pingstore", source, experiment="exp172", stage="compute"
    )
    with stage_run(
        REPO,
        "exp172",
        "analyse",
        inputs={"compute": computed},
        run_id=run_id,
        configuration=dict(method="recompute all saved-spike spatial/phase metrics"),
    ) as run:
        work = materialise_compute(computed, run.scratch / "evidence")
        cfg, summary = validate(work)
        dump(
            run.export / "results.json",
            dict(
                validated_trials=len(summary["trials"]),
                trials=summary["trials"],
                single_seed=len(cfg["seeds"]) == 1,
                limitation="Tuned single-seed selected-cue demonstration; no held-out test or isolated CTC mechanism",
                elapsed_seconds=perf_counter() - started,
            ),
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    print(f"Analysis {perf_counter() - started:.2f} s; no simulation")
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    main(**vars(parser.parse_args()))
