"""Measure retained samples and analytical responses; never simulate or publish."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "tools")]

import numpy as np
from experiments.exp081 import inputs, recipe
from pingstore.contracts import PingstoreError, write_json_atomic
from pingstore.stages import stage_run


def summarize(predicted: np.ndarray, empirical: np.ndarray) -> dict:
    valid = (predicted > 0) | (empirical > 0)
    positive = valid & (empirical > 0)
    correlation = None
    if (
        valid.sum() > 1
        and np.std(predicted[valid]) > 0
        and np.std(empirical[valid]) > 0
    ):
        correlation = float(np.corrcoef(predicted[valid], empirical[valid])[0, 1])
    return {
        "pearson_r": correlation,
        "mean_absolute_error_mV": float(
            np.mean(np.abs(predicted[valid] - empirical[valid]))
        )
        if valid.any()
        else 0.0,
        "median_predicted_empirical_ratio": float(
            np.median(predicted[positive] / empirical[positive])
        )
        if positive.any()
        else None,
    }


def samples(
    path: Path, shape: tuple, rates: np.ndarray, probes: np.ndarray | None = None
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as retained:
        values = retained["samples_mV"]
        if (
            values.shape != shape
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
            or not np.array_equal(retained["input_rates_hz"], rates)
        ):
            raise PingstoreError(
                "sample shape, values or rates disagree with the retained recipe"
            )
        if probes is not None and not np.array_equal(retained["probes_uS"], probes):
            raise PingstoreError(
                "sample conductances disagree with the retained recipe"
            )
    return values


def analyse(identity: str, *, run_id: str | None = None) -> str:
    compute = inputs.source(REPO, identity, "compute")
    cfg = inputs.configuration(compute)
    if compute.record["inputs"]:
        raise PingstoreError("standalone exp081 compute must have no upstream inputs")
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs={"compute": compute},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        rates, probes = np.meshgrid(cfg["input_rate_grid_hz"], cfg["probes_uS"])
        features = samples(
            compute.export / "feature_samples.npz",
            (*rates.shape, cfg["moment_draws"]),
            rates,
            probes,
        )
        distribution = samples(
            compute.export / "distribution_samples.npz",
            (len(cfg["distribution_rates_hz"]), cfg["distribution_draws"]),
            np.asarray(cfg["distribution_rates_hz"]),
        )
        empirical_mean = features.mean(axis=-1)
        empirical_sd = features.std(axis=-1, ddof=cfg["sd_ddof"])
        _, stationary_voltage = recipe.linear_operating_point(rates, probes, config=cfg)
        analytical_mean = stationary_voltage - cfg["membrane"]["E_L_mV"]
        variance = recipe.predicted_variance(
            rates.ravel(),
            probes.ravel(),
            grid_points=cfg["fine_grid_points"],
            config=cfg,
        ).reshape(rates.shape)
        analytical_sd = np.sqrt(variance)
        coarse = recipe.predicted_variance(
            rates.ravel(),
            probes.ravel(),
            grid_points=cfg["coarse_grid_points"],
            config=cfg,
        ).reshape(rates.shape)
        relative = np.divide(
            np.abs(variance - coarse),
            variance,
            out=np.zeros_like(variance),
            where=variance > 0,
        )
        np.savez_compressed(
            run.export / "moments.npz",
            input_rates_hz=rates,
            probes_uS=probes,
            empirical_mean_mV=empirical_mean,
            empirical_sd_mV=empirical_sd,
            analytical_mean_mV=analytical_mean,
            analytical_sd_mV=analytical_sd,
        )
        # Retain measured bins rather than duplicating the upstream samples.
        upper = float(np.ceil(np.max(distribution) / 5.0) * 5.0)
        bins = np.linspace(0.0, upper if upper > 0 else 5.0, cfg["histogram_bins"] + 1)
        probabilities = np.asarray(
            [
                np.histogram(
                    values, bins=bins, weights=np.full(values.shape, 1.0 / values.size)
                )[0]
                for values in distribution
            ]
        )
        np.savez_compressed(
            run.export / "histograms.npz", edges_mV=bins, probability=probabilities
        )
        frequency = np.geomspace(
            *cfg["frequency_plot_bounds_hz"], cfg["frequency_plot_points"]
        )
        operating_rates = cfg["frequency_response_rates_hz"]
        probe = cfg["nominal_probe_uS"]
        reference = abs(
            recipe.synapse_membrane_transfer(
                np.asarray([0.0]), operating_rates[0], probe, config=cfg
            )[0]
        )
        responses = {}
        for name, transfer in (
            ("unaveraged_db", recipe.synapse_membrane_transfer),
            ("averaged_db", recipe.complete_transfer),
        ):
            responses[name] = np.asarray(
                [
                    20
                    * np.log10(
                        np.maximum(
                            np.abs(transfer(frequency, rate, probe, config=cfg))
                            / reference,
                            1e-8,
                        )
                    )
                    for rate in operating_rates
                ]
            )
        np.savez_compressed(
            run.export / "frequency_response.npz", frequency_hz=frequency, **responses
        )
        write_json_atomic(
            run.export / "results.json",
            {
                "schema": "exp081.analysis/v1",
                "parameters": cfg,
                "comparison": {
                    "mean": summarize(analytical_mean, empirical_mean),
                    "standard_deviation": summarize(analytical_sd, empirical_sd),
                },
                "quadrature": {
                    "maximum_relative_refinement_change": float(relative.max()),
                    "fine_grid_points": cfg["fine_grid_points"],
                    "coarse_grid_points": cfg["coarse_grid_points"],
                },
            },
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp081 v3 compute run ID"
    )
    parser.add_argument("--run-id", help="unused v3 identity reserved before dispatch")
    args = parser.parse_args()
    analyse(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
