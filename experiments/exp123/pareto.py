"""Accuracy, deadline and network-spike frontiers from explicit paired runs."""

import json
import numpy as np
from snnlab import analysis
from experiments.exp123 import recipe
from pingstore.stages import source_run, stage_run


def frontier(correct, cost, time):
    """Nondomination with objectives correct count up, spike sum/time down."""
    dominates = (
        (correct[..., :, None] >= correct[..., None, :])
        & (cost[..., :, None] <= cost[..., None, :])
        & (time[:, None] <= time[None, :])
        & (
            (correct[..., :, None] > correct[..., None, :])
            | (cost[..., :, None] < cost[..., None, :])
            | (time[:, None] < time[None, :])
        )
    )
    return ~dominates.any(axis=-2)


def analyse_pareto(source, added_identity):
    added = source_run(
        recipe.REPO / ".pingstore",
        added_identity,
        stage="compute",
        experiment=recipe.SLUG,
    )
    base = source.record["execution"]["configuration"]
    extra = added.record["execution"]["configuration"]
    if base.get("protocol") != "accuracy" or extra.get("protocol") != "pareto":
        raise ValueError("expected a baseline accuracy run and a added-rate pareto run")
    for key in (
        "image_indices",
        "seeds",
        "tau_gaba_ms",
        "dt_ms",
        "duration_ms",
        "burn_in_ms",
        "input_rate_hz",
        "checkpoint_sha256",
        "biological_defaults",
        "generator_seed_rule",
        "ping_helper_sha256",
    ):
        if key == "input_rate_hz":
            if base[key] != recipe.RATE_HZ:
                raise ValueError("baseline rate mismatch")
        elif base[key] != extra[key]:
            raise ValueError(f"incompatible paired compute inputs: {key}")
    if tuple(extra["input_rates_hz"]) != recipe.PARETO_NEW_RATES_HZ:
        raise ValueError("unexpected added rates")
    rates = recipe.PARETO_RATES_HZ
    deadlines = np.asarray(recipe.PARETO_DEADLINES_MS)
    images, seeds, taus = base["image_indices"], base["seeds"], base["tau_gaba_ms"]
    ni, nr, nt, ns, nd = len(images), len(rates), len(taus), len(seeds), len(deadlines)
    dt = base["dt_ms"]
    steps = np.rint(deadlines / dt).astype(int) - 1
    correct = np.zeros((ni, nr, nt, ns, nd), dtype=bool)
    spike_cost = np.zeros_like(correct, dtype=np.int64)
    input_cost = np.zeros((ni, nr, ns, nd), dtype=np.int64)
    trial_frequency = np.full((ni, nr, nt, ns), np.nan)
    psd_sum = None
    labels = []
    measurement = dict(
        schema="exp123.pareto.measurement/v1",
        deadlines_ms=deadlines.tolist(),
        input_rates_hz=list(rates),
        spike_cost="all E + I + output spikes through deadline; input spikes excluded",
        classification="instantaneous cumulative output count argmax; lowest class index wins ties",
        aggregation="equal mean over three draws within each image, then over all 500 images; all trials included",
        dominance="within encoding rate: accuracy up, time and mean network spike count down; weak improvements in all and strict in at least one",
        frontier_selection="exact integer correct counts and summed network spikes before division; empirical sampled candidates only",
        bootstrap="1000 paired stratified image resamples, 50 per digit; percentile 95% intervals and nondomination membership frequency",
        bootstrap_seed=12345,
        frequency="peak of mean full-200-ms E-population Welch PSD; Hann, native dt, centered, no detrend, 5-150 Hz, parabolic spectrum-neighbor interpolation; no harmonic rejection",
        pairing="identical inputs across decay; per-image/draw uniforms reused across rates; verify low-input subset of baseline subset of high-input",
        script_sha256={
            p.name: recipe.sha256(p)
            for p in recipe.REPO.joinpath("experiments/exp123").glob("*.py")
        },
    )
    with stage_run(
        recipe.REPO,
        recipe.SLUG,
        "analyse",
        inputs={"baseline_compute": source, "added_rates_compute": added},
        configuration=measurement,
    ) as run:
        for j, image_index in enumerate(images):
            paired_inputs = []
            label = None
            for ri, rate in enumerate(rates):
                owner = source if rate == recipe.RATE_HZ else added
                protocol = "accuracy" if rate == recipe.RATE_HZ else "pareto"
                unit = owner.unit(recipe.recording_unit(image_index, rate, protocol))
                with np.load(unit / "input.npz", allow_pickle=False) as data:
                    pixels, encoded = data["pixels"], data["spikes"]
                    current_label = int(data["label"])
                    if not np.array_equal(data["seeds"], seeds) or not np.array_equal(
                        data["generator_seeds"], [1000 * image_index + s for s in seeds]
                    ):
                        raise ValueError("input stream identity mismatch")
                    if label is None:
                        label, reference_pixels = current_label, pixels.copy()
                    elif label != current_label or not np.array_equal(
                        pixels, reference_pixels
                    ):
                        raise ValueError("image changed across rates")
                    paired_inputs.append(encoded.copy())
                    input_cost[j, ri] = np.cumsum(
                        encoded.sum(axis=2), axis=0, dtype=np.int64
                    )[steps].T
                for ti, tau in enumerate(taus):
                    with np.load(
                        unit / f"tau{tau:g}--spikes.npz", allow_pickle=False
                    ) as data:
                        e, i, out = (data[k] for k in ("spk_e", "spk_i", "spk_out"))
                        if (
                            e.shape != (recipe.STEPS, ns, 1024)
                            or i.shape != (recipe.STEPS, ns, 256)
                            or out.shape != (recipe.STEPS, ns, 10)
                        ):
                            raise ValueError("unexpected recording shape")
                        counts = np.cumsum(out, axis=0, dtype=np.int32)
                        correct[j, ri, ti] = (counts[steps].argmax(axis=-1) == label).T
                        population_counts = (
                            e.sum(axis=2) + i.sum(axis=2) + out.sum(axis=2)
                        )
                        spike_cost[j, ri, ti] = np.cumsum(
                            population_counts, axis=0, dtype=np.int64
                        )[steps].T
                        for b in range(ns):
                            spectrum = analysis.power_spectrum(
                                e[:, b].sum(axis=1),
                                dt,
                                center=True,
                                detrend=False,
                                nperseg=len(e),
                                window="hann",
                                scaling="density",
                            )
                            if psd_sum is None:
                                frequencies = spectrum["frequencies_hz"]
                                psd_sum = np.zeros((nr, nt, len(frequencies)))
                            psd_sum[ri, ti] += spectrum["power"]
                            if np.sum(spectrum["power"]) > 0:
                                peak = analysis.spectral_peak(
                                    frequencies,
                                    spectrum["power"],
                                    [5.0, 150.0],
                                    interpolation="parabolic",
                                    interpolation_boundary="spectrum",
                                    clamp_band=False,
                                )
                                trial_frequency[j, ri, ti, b] = peak["frequency_hz"]
            if np.any(paired_inputs[0] & ~paired_inputs[1]) or np.any(
                paired_inputs[1] & ~paired_inputs[2]
            ):
                raise ValueError("cross-rate random-stream pairing is not nested")
            labels.append(label)
            if (j + 1) % 100 == 0:
                print(f"Pareto measurements: {j + 1}/{ni} images", flush=True)
        labels = np.asarray(labels)
        rng = np.random.default_rng(measurement["bootstrap_seed"])
        weights = np.zeros((1000, ni), dtype=np.int64)
        for digit in range(10):
            pos = np.flatnonzero(labels == digit)
            samples = rng.choice(pos, (1000, len(pos)), replace=True)
            for b in range(1000):
                weights[b] += np.bincount(samples[b], minlength=ni)
        image_correct = correct.sum(axis=3, dtype=np.int64)
        image_cost = spike_cost.sum(axis=3, dtype=np.int64)
        boot_correct = np.einsum(
            "bi,irtd->brtd", weights, image_correct, optimize=False
        )
        boot_cost = np.einsum("bi,irtd->brtd", weights, image_cost, optimize=False)
        total_correct, total_cost = image_correct.sum(axis=0), image_cost.sum(axis=0)
        aci = np.quantile(boot_correct / (ni * ns), [0.025, 0.975], axis=0)
        cci = np.quantile(boot_cost / (ni * ns), [0.025, 0.975], axis=0)
        times = np.tile(deadlines, nt)
        rows, frequency_rows = [], []
        for ri, rate in enumerate(rates):
            empirical = frontier(
                total_correct[ri].ravel(), total_cost[ri].ravel(), times
            )
            membership = frontier(
                boot_correct[:, ri].reshape(1000, -1),
                boot_cost[:, ri].reshape(1000, -1),
                times,
            ).mean(axis=0)
            for ti, tau in enumerate(taus):
                peak = analysis.spectral_peak(
                    frequencies,
                    psd_sum[ri, ti],
                    [5.0, 150.0],
                    interpolation="parabolic",
                    interpolation_boundary="spectrum",
                    clamp_band=False,
                )
                finite = trial_frequency[:, ri, ti][
                    np.isfinite(trial_frequency[:, ri, ti])
                ]
                frequency_rows.append(
                    dict(
                        input_rate_hz=rate,
                        tau_gaba_ms=tau,
                        peak_of_mean_psd_hz=peak["frequency_hz"],
                        individual_peak_quartiles_hz=np.quantile(
                            finite, [0.25, 0.5, 0.75]
                        ).tolist()
                        if len(finite)
                        else None,
                        non_silent_presentations=len(finite),
                    )
                )
                for di, deadline in enumerate(deadlines):
                    rows.append(
                        dict(
                            input_rate_hz=rate,
                            tau_gaba_ms=tau,
                            deadline_ms=float(deadline),
                            accuracy=float(total_correct[ri, ti, di] / (ni * ns)),
                            accuracy_ci95=aci[:, ri, ti, di].tolist(),
                            mean_network_spikes=float(
                                total_cost[ri, ti, di] / (ni * ns)
                            ),
                            network_spikes_ci95=cci[:, ri, ti, di].tolist(),
                            mean_input_spikes=float(input_cost[:, ri, :, di].mean()),
                            correct_presentations=int(total_correct[ri, ti, di]),
                            total_network_spikes=int(total_cost[ri, ti, di]),
                            presentations=ni * ns,
                            nondominated=bool(empirical[ti * nd + di]),
                            bootstrap_frontier_fraction=float(membership[ti * nd + di]),
                        )
                    )
        np.savez_compressed(
            run.export / "pareto_trials.npz",
            image_indices=images,
            labels=labels,
            seeds=seeds,
            input_rates_hz=rates,
            tau_gaba_ms=taus,
            deadlines_ms=deadlines,
            correct=correct,
            network_spikes=spike_cost,
            input_spikes=input_cost,
            individual_psd_peak_hz=trial_frequency,
            psd_frequencies_hz=frequencies,
            mean_psd=psd_sum / (ni * ns),
        )
        report = dict(
            schema="exp123.pareto.analysis/v1",
            design=dict(
                images=ni,
                image_indices=images,
                images_per_digit=50,
                seeds=seeds,
                input_rates_hz=list(rates),
                tau_gaba_ms=taus,
                deadlines_ms=deadlines.tolist(),
                duration_ms=base["duration_ms"],
                dt_ms=dt,
                presentations=ni * ns * nt * nr,
            ),
            measurement={k: v for k, v in measurement.items() if k != "script_sha256"},
            rows=rows,
            frequencies=frequency_rows,
            pairing_validation=dict(nested_input_sets_verified=True, images_checked=ni),
        )
        (run.export / "pareto.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        identity = run.run_id
    return identity
