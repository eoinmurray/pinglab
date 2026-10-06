"""Matched-coordinate accuracy with paired coverage and image-level uncertainty."""

import json
import numpy as np
from experiments.exp123 import recipe
from experiments.exp123.analyse import volley_times
from pingstore.stages import stage_run


def analyse_accuracy(source):
    cfg = source.record["execution"]["configuration"]
    measurement = recipe.measurement()
    measurement.update(
        schema="exp123.accuracy.measurement/v1",
        accuracy_prediction="instantaneous cumulative-count argmax; lowest class index wins ties",
        time_grid_ms=list(range(201)),
        cycle_grid=np.arange(0, 8.01, 0.25).tolist(),
        input_spike_grid=list(range(0, 1501, 25)),
        coordinate_sampling="last completed output update at query time; input counts sampled at first budget crossing",
        coverage="within each image/seed, include a coordinate only if available in all three settings; no extrapolation",
        aggregation="mean eligible draws within image, then equal mean over eligible images",
        bootstrap="1000 paired stratified image resamples; 50 images per digit with replacement; percentile 95% intervals",
        bootstrap_seed=12345,
    )
    grids = {
        "time": np.array(measurement["time_grid_ms"], dtype=float),
        "cycles": np.array(measurement["cycle_grid"]),
        "input": np.array(measurement["input_spike_grid"]),
    }
    ni, nt, ns = len(cfg["image_indices"]), len(cfg["tau_gaba_ms"]), len(cfg["seeds"])
    correct = {
        key: np.full((ni, nt, ns, len(grid)), np.nan, dtype=np.float32)
        for key, grid in grids.items()
    }
    labels = []
    valid_rows = []
    dt = cfg["dt_ms"]
    with stage_run(
        recipe.REPO,
        recipe.SLUG,
        "analyse",
        inputs={"compute": source},
        configuration=measurement,
    ) as run:
        for j, image_index in enumerate(cfg["image_indices"]):
            unit = source.unit(f"image{image_index:05d}")
            with np.load(unit / "input.npz", allow_pickle=False) as data:
                label = int(data["label"])
                input_counts = np.concatenate(
                    (
                        np.zeros((1, ns), dtype=np.int32),
                        np.cumsum(data["spikes"].sum(axis=2), axis=0, dtype=np.int32),
                    )
                )
            labels.append(label)
            outputs, peaks_all, valid_all = [], [], []
            for ti, tau in enumerate(cfg["tau_gaba_ms"]):
                with np.load(
                    unit / f"tau{tau:g}--spikes.npz", allow_pickle=False
                ) as data:
                    cumulative = np.concatenate(
                        (
                            np.zeros((1, ns, 10), dtype=np.int32),
                            np.cumsum(data["spk_out"], axis=0, dtype=np.int32),
                        )
                    )
                    outputs.append(cumulative.argmax(axis=-1) == label)
                    peaks, accepted = [], []
                    for b in range(ns):
                        p, valid, participation = volley_times(
                            data["spk_i"][:, b], dt, measurement
                        )
                        peaks.append(p)
                        accepted.append(valid)
                        valid_rows.append(
                            dict(
                                image_index=image_index,
                                seed=cfg["seeds"][b],
                                tau_gaba_ms=tau,
                                valid=valid,
                                volleys=len(p),
                                i_participation_pct=participation,
                            )
                        )
                    peaks_all.append(peaks)
                    valid_all.append(accepted)
            for b in range(ns):
                reaches_input = grids["input"] <= input_counts[-1, b]
                input_steps = np.searchsorted(
                    input_counts[:, b], grids["input"][reaches_input], side="left"
                )
                cycle_ok = all(v[b] for v in valid_all)
                end = min(len(p[b]) - 1 for p in peaks_all) if cycle_ok else -1
                reaches_cycle = grids["cycles"] <= end
                for ti in range(nt):
                    correct["time"][j, ti, b] = outputs[ti][
                        np.rint(grids["time"] / dt).astype(int), b
                    ]
                    correct["input"][j, ti, b, reaches_input] = outputs[ti][
                        input_steps, b
                    ]
                    if cycle_ok:
                        peaks = peaks_all[ti][b]
                        query = np.interp(
                            grids["cycles"][reaches_cycle], np.arange(len(peaks)), peaks
                        )
                        steps = np.floor(query / dt + 1e-8).astype(int)
                        correct["cycles"][j, ti, b, reaches_cycle] = outputs[ti][
                            steps, b
                        ]
            if (j + 1) % 100 == 0:
                print(f"Accuracy measured for {j + 1}/{ni} images", flush=True)
        labels = np.asarray(labels)
        rng = np.random.default_rng(measurement["bootstrap_seed"])
        weights = np.zeros((1000, ni), dtype=float)
        for digit in range(10):
            positions = np.flatnonzero(labels == digit)
            draws = rng.choice(positions, (1000, len(positions)), replace=True)
            for b in range(1000):
                weights[b] += np.bincount(draws[b], minlength=ni)
        curves = {}
        arrays = dict(
            image_indices=cfg["image_indices"],
            labels=labels,
            seeds=cfg["seeds"],
            tau_gaba_ms=cfg["tau_gaba_ms"],
        )
        for key, grid in grids.items():
            values = correct[key]
            count = np.isfinite(values).sum(axis=2)
            per_image = np.divide(
                np.nansum(values, axis=2),
                count,
                out=np.full((ni, nt, len(grid)), np.nan),
                where=count > 0,
            )
            means, lows, highs = [], [], []
            for ti in range(nt):
                available = np.isfinite(per_image[:, ti])
                denominator = available.sum(axis=0)
                mean = np.divide(
                    np.nansum(per_image[:, ti], axis=0),
                    denominator,
                    out=np.full(len(grid), np.nan),
                    where=denominator > 0,
                )
                numer = np.einsum(
                    "bi,ig->bg",
                    weights,
                    np.nan_to_num(per_image[:, ti]),
                    optimize=False,
                )
                denom = np.einsum(
                    "bi,ig->bg", weights, available.astype(float), optimize=False
                )
                bootstrap = np.divide(
                    numer, denom, out=np.full_like(numer, np.nan), where=denom > 0
                )
                low, high = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
                for g in np.flatnonzero(denominator):
                    low[g], high[g] = np.nanquantile(bootstrap[:, g], [0.025, 0.975])
                means.append(mean)
                lows.append(low)
                highs.append(high)
            coverage = np.isfinite(values[:, 0]).sum(axis=(0, 1)) / (ni * ns)
            curves[key] = dict(
                grid=grid.tolist(),
                accuracy=[
                    [float(v) if np.isfinite(v) else None for v in row] for row in means
                ],
                ci_low=[
                    [float(v) if np.isfinite(v) else None for v in row] for row in lows
                ],
                ci_high=[
                    [float(v) if np.isfinite(v) else None for v in row] for row in highs
                ],
                trial_coverage=coverage.tolist(),
                images_available=np.isfinite(per_image[:, 0]).sum(axis=0).tolist(),
            )
            arrays[f"{key}_grid"] = grid
            arrays[f"{key}_correct"] = values
            arrays[f"{key}_per_image_accuracy"] = per_image
        np.savez_compressed(run.export / "accuracy.npz", **arrays)
        report = dict(
            schema="exp123.accuracy.analysis/v1",
            design={
                key: cfg[key]
                for key in (
                    "protocol",
                    "image_indices",
                    "image_selection",
                    "image_selection_seed",
                    "partition",
                    "seeds",
                    "generator_seed_rule",
                    "tau_gaba_ms",
                    "dt_ms",
                    "duration_ms",
                    "burn_in_ms",
                    "input_rate_hz",
                    "biological_defaults",
                )
            },
            measurement={
                key: value
                for key, value in measurement.items()
                if key != "script_sha256"
            },
            curves=curves,
            volley_diagnostics=valid_rows,
            presentations=ni * nt * ns,
            final_time_accuracy=[c[-1] for c in curves["time"]["accuracy"]],
        )
        (run.export / "accuracy.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        identity = run.run_id
    return identity
