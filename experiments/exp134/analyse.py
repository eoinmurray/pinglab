"""Measure population dynamics of the two paired membrane sweeps."""

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
from experiments.exp134 import recipe
from experiments.exp134.dynamics import dynamics
from pingstore.stages import source_run, stage_run
from snnlab import analysis


def measure_condition(condition, cfg, units):
    settings = dict(recipe.measurement(), dt_ms=cfg["dt_ms"])
    name = condition["condition_id"]
    samples = {
        key: []
        for key in (
            "participation_e_pct",
            "e_spikes_per_second",
            "rhythmicity_contrast",
        )
    }
    accepted = 0
    power_sum = None
    image_coverage = []
    for index in cfg["image_indices"]:
        participation = []
        with np.load(
            Path(units[index]) / f"{name}--spikes.npz", allow_pickle=False
        ) as data:
            excitatory, inhibitory = data["spk_e"], data["spk_i"]
            for draw in range(len(cfg["seeds"])):
                e, i = excitatory[:, draw], inhibitory[:, draw]
                metrics, diagnostic = dynamics(e, i, settings)
                accepted += int(metrics["valid_cycles"])
                if metrics["participation_e_pct"] is not None:
                    participation.append(metrics["participation_e_pct"])
                samples["e_spikes_per_second"].append(
                    float(e.sum() / (cfg["duration_ms"] / 1000))
                )
                if metrics["rhythmicity_contrast"] is not None:
                    samples["rhythmicity_contrast"].append(
                        metrics["rhythmicity_contrast"]
                    )
                power = diagnostic["psd"]
                if power_sum is None:
                    power_sum = np.zeros_like(power)
                power_sum += power
                frequency = diagnostic["frequency_hz"]
        image_coverage.append(
            dict(image_index=index, accepted_draws=len(participation))
        )
        if participation:
            samples["participation_e_pct"].append(float(np.mean(participation)))
    peak = analysis.spectral_peak(
        frequency,
        power_sum,
        settings["frequency_search_hz"],
        interpolation="parabolic",
        interpolation_boundary="spectrum",
        clamp_band=False,
    )["frequency_hz"]
    summary = dict(
        condition,
        psd_frequency_hz=peak,
        accepted_draws=accepted,
        total_draws=len(cfg["image_indices"]) * len(cfg["seeds"]),
        participation_images=len(samples["participation_e_pct"]),
        image_coverage=image_coverage,
    )
    for key, values in samples.items():
        summary[key] = float(np.mean(values)) if values else None
        summary[f"{key}_n"] = len(values)
    return summary


def analyse(identity, workers=4):
    if workers < 1:
        raise ValueError("workers must be positive")
    source = source_run(
        REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    cfg = source.record["execution"]["configuration"]
    if source.record.get("scope") != "full" or cfg.get("schema") != "exp134.compute/v1":
        raise ValueError("requires a complete membrane-sweep compute run")
    units = {
        index: str(source.unit(f"image{index:05d}")) for index in cfg["image_indices"]
    }
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs={"compute": source},
        configuration=recipe.measurement(),
    ) as run:
        measured = {}

        def retain(summary):
            name = summary["condition_id"]
            measured[name] = summary
            (run.export / f"dynamics-{name}.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n"
            )
            print(
                f"Measured {name} ({len(measured)}/{len(cfg['conditions'])})",
                flush=True,
            )

        if workers == 1:
            for condition in cfg["conditions"]:
                retain(measure_condition(condition, cfg, units))
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as executor:
                futures = [
                    executor.submit(measure_condition, c, cfg, units)
                    for c in cfg["conditions"]
                ]
                for future in as_completed(futures):
                    retain(future.result())
        rows = [measured[c["condition_id"]] for c in cfg["conditions"]]
        report = dict(
            schema="exp134.analysis/v1",
            design=cfg,
            measurement=recipe.measurement(),
            summaries=rows,
            raster_design=dict(
                image_index=cfg["image_indices"][0],
                seed=cfg["seeds"][0],
                multipliers=[0.5, 1.0, 2.0],
            ),
        )
        (run.export / "metrics.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        # Raster events are measured here so present never reads compute activity.
        events = dict(dt_ms=cfg["dt_ms"])
        for condition in cfg["conditions"]:
            if condition["multiplier"] not in (0.5, 1.0, 2.0):
                continue
            name = condition["condition_id"]
            with np.load(
                Path(units[cfg["image_indices"][0]]) / f"{name}--spikes.npz",
                allow_pickle=False,
            ) as data:
                for key in ("spk_e", "spk_i"):
                    step, neuron = np.nonzero(data[key][:, 0])
                    events[f"{name}--{key}--step"] = step
                    events[f"{name}--{key}--neuron"] = neuron
        np.savez_compressed(run.export / "rasters.npz", **events)
        result = run.run_id
    print(result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    analyse(args.source, args.workers)
