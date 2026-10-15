"""All measurements of the uniform, paired inhibitory-decay sweep."""

import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from experiments.exp123 import recipe
from experiments.exp123.accuracy import measure_accuracy
from experiments.exp123.analyse import measure_trajectories
from experiments.exp123.overview_analysis import dynamics
from pingstore.stages import stage_run
from snnlab import analysis


def condition_dynamics(condition, cfg, settings, units):
    """Measure one decay setting without changing image or draw aggregation."""
    tau = condition["tau_gaba_ms"]
    values = {key: [] for key in ("participation_e_pct", "e_spikes_per_second", "rhythmicity_contrast")}
    power_sum = None
    accepted = 0
    # Equal draw counts make this pooled mean equal to mean-within-image,
    # followed by an equal mean over images. Undefined participation is
    # handled separately by first averaging accepted draws within image.
    for image_index in cfg["image_indices"]:
        unit = Path(units[image_index])
        participation = []
        with np.load(unit / f"tau{tau:g}--spikes.npz", allow_pickle=False) as data:
            excitatory, inhibitory = data["spk_e"], data["spk_i"]
            for b in range(len(cfg["seeds"])):
                e, i = excitatory[:, b], inhibitory[:, b]
                metrics, diagnostic = dynamics(e, i, settings)
                accepted += int(metrics["valid_cycles"])
                if metrics["participation_e_pct"] is not None:
                    participation.append(metrics["participation_e_pct"])
                values["e_spikes_per_second"].append(float(e.sum() / (cfg["duration_ms"] / 1000)))
                contrast = metrics["rhythmicity_contrast"]
                if contrast is not None:
                    values["rhythmicity_contrast"].append(contrast)
                power = diagnostic["psd"]
                if power_sum is None:
                    power_sum = np.zeros_like(power)
                power_sum += power
                frequency = diagnostic["frequency_hz"]
        if participation:
            values["participation_e_pct"].append(float(np.mean(participation)))
    peak = analysis.spectral_peak(
        frequency, power_sum, settings["frequency_search_hz"],
        interpolation="parabolic", interpolation_boundary="spectrum", clamp_band=False,
    )["frequency_hz"]
    summary = dict(condition, psd_frequency_hz=peak,
                   accepted_draws=accepted,
                   total_draws=len(cfg["image_indices"]) * len(cfg["seeds"]),
                   participation_images=len(values["participation_e_pct"]))
    for key, samples in values.items():
        summary[key] = float(np.mean(samples)) if samples else None
        summary[f"{key}_n"] = len(samples)
    return summary, dict(condition, frequency_hz=peak)


def measure_dynamics(source, run, cfg, workers=1):
    settings = recipe.overview_configuration()
    settings.update(cfg)
    settings = recipe.overview_analysis_configuration(settings)
    conditions = [dict(condition_id=f"tau{j:02d}", sweep="tau", condition_index=j,
                       axis_value=tau, tau_gaba_ms=tau)
                  for j, tau in enumerate(cfg["tau_gaba_ms"])]
    units = {index: str(source.unit(recipe.recording_unit(index, recipe.RATE_HZ, "benchmark")))
             for index in cfg["image_indices"]}
    measured = {}

    def retain(result):
        summary, psd_summary = result
        measured[summary["condition_index"]] = result
        (run.export / f"dynamics-{summary['condition_id']}.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        print(f"Dynamics measured at GABA {summary['tau_gaba_ms']:g} ms "
              f"({len(measured)}/{len(conditions)} settings)", flush=True)

    if workers == 1:
        for condition in conditions:
            retain(condition_dynamics(condition, cfg, settings, units))
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(condition_dynamics, condition, cfg, settings, units)
                       for condition in conditions]
            for future in as_completed(futures):
                retain(future.result())
    ordered = [measured[j] for j in range(len(conditions))]
    summaries = [result[0] for result in ordered]
    psd_summaries = [result[1] for result in ordered]
    report = dict(
        schema="exp123.overview.analysis/v1",
        design=dict(cfg, conditions=conditions,
                    aggregation="peak of pooled PSD; E spike rate averages all equally sampled draws/images; participation averages accepted draws within image then eligible images; contrast averages finite draws",
                    frequency_search_hz=settings["frequency_search_hz"]),
        summaries=summaries, psd_summaries=psd_summaries, rows=summaries,
    )
    (run.export / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def measure_overview_rasters(source, run, definition):
    """Export sparse plotting events for the three existing illustrative rasters."""
    cfg = definition["sweep"]
    index = definition["illustrative_image_indices"][0]
    unit = source.unit(recipe.recording_unit(index, recipe.RATE_HZ, "benchmark"))
    events = dict(tau_gaba_ms=np.array(recipe.TAUS_MS), dt_ms=cfg["dt_ms"],
                  image_index=index, seed=cfg["seeds"][0])
    for j, tau in enumerate(recipe.TAUS_MS):
        with np.load(unit / f"tau{tau:g}--spikes.npz", allow_pickle=False) as spikes:
            for key in ("spk_e", "spk_i"):
                step, neuron = np.nonzero(spikes[key][:, 0])
                events[f"tau{j}--{key}--step"] = step
                events[f"tau{j}--{key}--neuron"] = neuron
    np.savez_compressed(run.export / "overview_rasters.npz", **events)


def analyse_shared(source, workers=1, reuse=None):
    definition = source.record["execution"]["configuration"]
    cfg = dict(definition["sweep"], shared_sweep=True)
    inputs = {"compute": source}
    if reuse is not None:
        if (reuse.record["execution"]["configuration"].get("schema") != "exp123.shared.measurement/v1"
                or reuse.record["inputs"].get("compute") != source.reference):
            raise ValueError("Reused shared measurements must reference this exact compute payload")
        inputs["previous_analysis"] = reuse
    with stage_run(
        recipe.REPO, recipe.SLUG, "analyse", inputs=inputs,
        configuration=dict(schema="exp123.shared.measurement/v1", sweep=cfg,
                           illustrative_image_indices=definition["illustrative_image_indices"],
                           dynamics_workers=workers,
                           reused_measurements="trajectory, accuracy and dynamics exports from explicit previous analysis" if reuse is not None else None,
                           measures=["trajectories", "matched_accuracy", "network_budget_latency", "dynamics", "deadline_cost", "pooled_sustained_decisions", "target_accuracy_frontier", "overview_raster_events"],
                           target_accuracy_pct=recipe.TARGET_ACCURACY_PCT),
    ) as run:
        if reuse is None:
            examples = dict(cfg, image_indices=definition["illustrative_image_indices"])
            measure_trajectories(source, run, examples)
            accuracy = measure_accuracy(source, run, cfg)
            measure_dynamics(source, run, cfg, workers=workers)
        else:
            shutil.copytree(reuse.directory / "export", run.export, dirs_exist_ok=True)
            accuracy = json.loads(reuse.file("accuracy.json").read_text())
        from experiments.exp123.pooled_decisions import measure_pooled_decisions

        if reuse is None or not (run.export / "pooled_decisions.npz").exists():
            measure_pooled_decisions(source, run, cfg, workers=workers)
        rows = []
        for ti, tau in enumerate(cfg["tau_gaba_ms"]):
            for j, time in enumerate(accuracy["curves"]["time"]["grid"]):
                rows.append(dict(
                    input_rate_hz=cfg["input_rate_hz"], tau_gaba_ms=tau,
                    deadline_ms=time, accuracy=accuracy["curves"]["time"]["accuracy"][ti][j],
                    mean_network_spikes=accuracy["mean_network_spikes_by_time"][ti][j],
                ))
        tradeoffs = dict(
            schema="exp123.shared.tradeoffs/v1",
            design=dict(tau_gaba_ms=cfg["tau_gaba_ms"], input_rates_hz=[cfg["input_rate_hz"]],
                        deadlines_ms=[20, 40, 80, 120, 200]), rows=rows,
            measurement="all 500 images and ten paired encodings; no trial selection at elapsed-time coordinates; mean cumulative E+I+output spikes excluding input",
        )
        (run.export / "tradeoffs.json").write_text(json.dumps(tradeoffs, indent=2, allow_nan=False) + "\n")
        from experiments.exp123.target_accuracy import measure_target_accuracy

        with np.load(run.export / "pooled_decisions.npz", allow_pickle=False) as trials:
            target = measure_target_accuracy(tradeoffs, trials, recipe.TARGET_ACCURACY_PCT)
        (run.export / "target_accuracy.json").write_text(
            json.dumps(target, indent=2, allow_nan=False) + "\n")
        measure_overview_rasters(source, run, definition)
        identity = run.run_id
    return identity
