"""Record, measure and plot explicit SHD gamma probes without optimizer updates."""

import argparse
import os
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
import torch
from experiments.exp124 import recipe
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run
from scipy.signal import welch
from snnlab.sim import GraphExecutor, plan_graph


def record(identity, device):
    from experiments.exp124.compute import batch
    source = source_run(REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG)
    cfg = dict(source.record["execution"]["configuration"])
    if cfg.get("schema") != "exp124.recipe/v3":
        raise ValueError("gamma probe requires the current two-layer schema")
    cfg.update(record_spikes=True, compile_mode=None, phase_definition=recipe.PHASES["gamma-probe"],
               probe_selection="first two validation utterances per class, source order",
               probe_noise_seed=cfg["seed"] + 4, weight_selection="final completed epoch",
               gamma_band_hz=[30, 90], discard_initial_ms=100, shuffle_block_ms=25,
               shuffle_replicates=50, shuffle_seed=1729)
    with np.load(source.file("spikes.npz"), allow_pickle=False) as data:
        selected = np.concatenate([np.flatnonzero((data["split"] == 1) & (data["labels"] == label))[:2]
                                   for label in range(cfg["classes"])])
        events = []
        for index in selected:
            mask = data["event_sample"] == index
            coordinates = np.unique(np.floor(data["event_time_ms"][mask] / cfg["dt_ms"]).astype(np.int64)
                                    * cfg["n_input"] + data["event_channel"][mask])
            events.append((coordinates // cfg["n_input"], coordinates % cfg["n_input"]))
        arrays = {"labels": data["labels"][selected], "official_train_indices": data["official_train_indices"][selected]}
    drive = batch(events, np.arange(len(events)), cfg, device, np.random.default_rng(cfg["probe_noise_seed"]))
    arrays["input_counts"] = drive.sum(-1).cpu().numpy().astype(np.int32)
    with stage_run(REPO, recipe.SLUG, "compute", phase="gamma-probe", inputs={"trained_network": source}, configuration=cfg) as run:
        with torch.no_grad():
            for model_name in cfg["models"]:
                bundle = recipe.author_network(cfg, model_name)
                model = GraphExecutor(plan_graph(bundle.graph), seed=cfg["seed"],
                                      projection_mode=cfg["projection_mode"], extension_validation=cfg["extension_validation"]).to(device)
                with np.load(source.unit(model_name) / "final_weights.npz", allow_pickle=False) as weights:
                    parameters = model.parameter_map()
                    if set(parameters) != set(weights.files):
                        raise ValueError("probe graph differs from trained parameters")
                    for name, parameter in parameters.items():
                        parameter.copy_(torch.as_tensor(weights[name], device=device))
                for condition in ("trained", "feedback_off") if model_name == "ping" else ("trained",):
                    if condition == "feedback_off":
                        for name in ("I1_to_E1.weight", "I2_to_E2.weight"):
                            parameters[name].zero_()
                    model.eval()
                    outputs = model({"drive": drive}, diagnostics=False, **cfg["forward_retention"]).outputs
                    for label in ("e1", "e2", "i1", "i2") if model_name == "ping" else ("e1", "e2"):
                        arrays[f"{model_name}_{condition}_{label}"] = outputs[f"{label}_spikes"].cpu().numpy().astype(np.uint8)
                del model
        np.savez_compressed(run.export / "rasters.npz", **arrays)
        run.record["execution"].update(device=str(device), snnlab_version=version("snnlab"),
                                       training_snnlab_version=source.record["execution"]["snnlab_version"],
                                       presentations=len(selected), optimizer_updates=0)
    return run.run_id


def spectrum(counts, dt):
    return welch(counts, fs=1000 / dt, window="hann", nperseg=min(256, len(counts)),
                 noverlap=min(128, len(counts) // 2), detrend="constant", axis=0, scaling="density")


def analyse(identity):
    source = source_run(REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG, phase="gamma-probe")
    cfg = source.record["execution"]["configuration"]
    offset = round(cfg["discard_initial_ms"] / cfg["dt_ms"])
    block = round(cfg["shuffle_block_ms"] / cfg["dt_ms"])
    rng = np.random.default_rng(cfg["shuffle_seed"])
    measured, exports = {}, {}
    with np.load(source.file("rasters.npz"), allow_pickle=False) as data:
        for key in data.files:
            if key in ("labels", "official_train_indices"):
                continue
            spikes = data[key][offset:]
            counts = spikes.sum(-1) if spikes.ndim == 3 else spikes
            frequencies, power = spectrum(counts.astype(float), cfg["dt_ms"])
            mean_power = power.mean(axis=1)
            band = (frequencies >= cfg["gamma_band_hz"][0]) & (frequencies <= cfg["gamma_band_hz"][1])
            broad = (frequencies >= 1) & (frequencies <= 200)
            comparison_band = (frequencies >= 5) & (frequencies <= 100)
            peak_index = np.flatnonzero(band)[np.argmax(mean_power[band])]
            centered = counts - counts.mean(0)
            variance = np.mean(centered ** 2, axis=0)
            active = variance > 0
            autocorrelation = np.array([np.mean(np.mean(centered[:len(centered)-lag or None, active]
                                                        * centered[lag:, active], axis=0) / variance[active])
                                        if active.any() else 0.0 for lag in range(101)])
            summary = {"mean_peak_30_90_hz": float(frequencies[peak_index]),
                       "mean_peak_5_100_hz": float(frequencies[np.flatnonzero(comparison_band)[np.argmax(mean_power[comparison_band])]]),
                       "mean_rate_hz": float(spikes.mean() * 1000 / cfg["dt_ms"]),
                       "gamma_power_fraction_1_200": float(mean_power[band].sum() / mean_power[broad].sum()) if mean_power[broad].sum() else None,
                       "trials": int(counts.shape[1]), "silent_trials": int((~active).sum())}
            summary["rate_definition"] = "per-neuron mean Hz" if spikes.ndim == 3 else "population-total input spikes/s"
            if "_i" in key and spikes.ndim == 3:
                synchronous = bool(np.all(spikes == spikes[:, :, :1]))
                intervals = [np.diff(np.flatnonzero(spikes[:, trial, 0])) * cfg["dt_ms"]
                             for trial in range(spikes.shape[1])]
                intervals = np.concatenate(intervals)
                summary.update(identical_inhibitory_spike_trains=synchronous,
                               inhibitory_interval_median_ms=float(np.median(intervals)) if len(intervals) else None,
                               inhibitory_intervals_in_30_90_hz_fraction=float(((intervals >= 1000 / 90) & (intervals <= 1000 / 30)).mean()) if len(intervals) else None)
            if spikes.ndim == 3:
                times, trials, neurons = np.nonzero(spikes)
                null = []
                for _ in range(cfg["shuffle_replicates"]):
                    start = times // block * block
                    shuffled_times = start + (rng.random(len(times)) * np.minimum(block, len(spikes) - start)).astype(int)
                    shuffled = np.zeros_like(counts)
                    np.add.at(shuffled, (shuffled_times, trials), 1)
                    null.append(spectrum(shuffled.astype(float), cfg["dt_ms"])[1].mean(axis=1))
                null = np.asarray(null)
                null_mean = null.mean(0)
                # The reference compares the selected maximum with surrogate maxima over the same band.
                ratio = mean_power[band] / np.maximum(null_mean[band], np.finfo(float).tiny)
                excess_peak = np.flatnonzero(band)[np.argmax(ratio)]
                null_maxima = (null[:, band] / np.maximum(null_mean[band], np.finfo(float).tiny)).max(1)
                summary.update(excess_peak_hz=float(frequencies[excess_peak]), peak_to_shuffle=float(ratio.max()),
                               shuffle_maximum_exceedance=float((1 + (null_maxima >= ratio.max()).sum()) / (len(null) + 1)),
                               shuffle_reference="independent spike relocation inside original 25 ms block; per-neuron/trial/block counts preserved; exploratory reference, not a gamma classifier")
                exports[f"{key}_null_mean"] = null_mean
                exports[f"{key}_null_p95"] = np.percentile(null, 95, axis=0)
            measured[key] = summary
            exports.update({f"{key}_power": mean_power, f"{key}_autocorrelation": autocorrelation,
                            f"{key}_counts": counts, "frequencies_hz": frequencies})
    with stage_run(REPO, recipe.SLUG, "analyse", inputs={"gamma_probe": source}, configuration=cfg) as run:
        write_json_atomic(run.export / "gamma.json", {"measurements": measured, "selection": "all selected trials, PSD averaged within each population; no concatenation across utterances",
                                                     "limitation": "SHD input is nonstationary; a band maximum alone does not demonstrate an endogenous PING rhythm"})
        np.savez_compressed(run.export / "gamma.npz", **exports)
    return run.run_id


def present(identity):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/pinglab-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    source = source_run(REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG)
    cfg = source.record["execution"]["configuration"]
    results = load_json(source.file("gamma.json"))
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    with np.load(source.file("gamma.npz"), allow_pickle=False) as data, stage_run(REPO, recipe.SLUG, "present", inputs={"gamma_analysis": source}, configuration=cfg) as run:
        keys = list(results["measurements"])
        for model in cfg["models"]:
            figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.3), constrained_layout=True)
            for layer, row in ((1, 0), (2, 1)):
                for population, color in (("e", "#202020"), ("i", "#a34d45")):
                    key = f"{model}_trained_{population}{layer}"
                    if key not in keys:
                        continue
                    time = np.arange(data[f"{key}_counts"].shape[0]) * cfg["dt_ms"] + cfg["discard_initial_ms"]
                    trace = gaussian_filter1d(data[f"{key}_counts"][:, 0].astype(float), sigma=3 / cfg["dt_ms"])
                    axes[row, 0].plot(time, trace, color=color, label=population.upper())
                    axes[row, 1].plot(data["frequencies_hz"], data[f"{key}_power"], color=color, label=population.upper())
                    axes[row, 1].plot(data["frequencies_hz"], data[f"{key}_null_mean"], color=color, linestyle=":", label=f"{population.upper()} shuffled")
                control = f"{model}_feedback_off_e{layer}_power"
                if control in data:
                    axes[row, 1].plot(data["frequencies_hz"], data[control], color="#416a94", linestyle="--", label="E, feedback off")
                axes[row, 0].set(title=f"{'AC'[row]}  Layer {layer}: first selected utterance", xlabel="Time (ms)", ylabel="Population spikes / ms", xlim=(100, 1000))
                axes[row, 1].set(title=f"{'BD'[row]}  Layer {layer}: mean trial spectrum", xlabel="Frequency (Hz)", ylabel="PSD (spikes²/ms²/Hz)", xlim=(1, 200))
                axes[row, 1].set_yscale("log")
                axes[row, 1].axvspan(30, 90, color="#888888", alpha=.12)
                for axis in axes[row]:
                    axis.spines[["top", "right"]].set_visible(False)
                    axis.legend(frameon=False, fontsize=7)
            scope = "full-data" if cfg.get("full_dataset") else "pilot"
            figure.suptitle(f"{model.upper()} — {scope} final-weight SHD gamma probe ({results['measurements'][f'{model}_trained_e1']['trials']} utterances)", fontsize=10)
            for extension in ("png", "svg"):
                figure.savefig(run.export / f"{model}_gamma_probe.{extension}", dpi=180)
            plt.close(figure)
        write_json_atomic(run.export / "numbers.json", results)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("analyse", "present"), required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    print((analyse if args.stage == "analyse" else present)(args.source))
