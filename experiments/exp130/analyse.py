"""Measure both explicitly paired recordings; no simulation or rendering.

The CLI requires an explicit completed compute run. It validates the retained
recordings and writes measurements to a new analysis run, without simulating
or launching presentation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
from experiments.exp130 import dataset, recipe
from experiments.exp130.custom import inputs
from pingstore.library import stage_run, write_json_atomic
from scipy.signal import butter, hilbert, sosfiltfilt, welch


def measure(data, cfg):
    dt = cfg["dt_ms"]
    times = np.arange(len(data["spk_e"])) * dt
    rates = dataset.source_rates(times, cfg)
    arrays = dict(
        time_ms=times,
        rate_private_e=rates[0],
        rate_private_i=rates[1],
        mean_g_e=data["mean_E_to_E"] + data["mean_private_e_to_E"],
        mean_g_i=data["mean_I_to_E"],
        mean_v_e=data["mean_v_e"],
        mean_v_i=data["mean_v_i"],
    )
    summary = {}
    width = round(1 / dt)
    n_bins = len(times) // width
    bin_times = np.arange(n_bins, dtype=float)  # left edges of 1 ms bins
    selection = (bin_times >= cfg["analysis_start_ms"]) & (
        bin_times < cfg.get("analysis_end_ms", cfg["t_ms"] - 500.0)
    )
    phase = (
        2 * np.pi * cfg["theta_hz"] * (bin_times - cfg["modulation_onset_ms"]) / 1000
    ) % (2 * np.pi)
    phase_bin = np.minimum((phase / (2 * np.pi) * 18).astype(int), 17)
    arrays["phase_centres_rad"] = (np.arange(18) + 0.5) * 2 * np.pi / 18
    bandpass = butter(4, [30, 90], btype="bandpass", fs=1000, output="sos")
    for pop in ("e", "i"):
        spikes = data[f"spk_{pop}"]
        counts = (
            spikes[: n_bins * width].reshape(n_bins, width, cfg[f"n_{pop}"]).sum((1, 2))
        )
        rate = counts * 1000 / cfg[f"n_{pop}"]
        arrays[f"population_rate_{pop}"] = rate
        freq, power = welch(rate[selection], fs=1000, nperseg=500, noverlap=250)
        arrays[f"frequency_{pop}_hz"], arrays[f"power_{pop}"] = freq, power
        filtered = sosfiltfilt(bandpass, rate)
        envelope = np.abs(hilbert(filtered))
        arrays[f"gamma_envelope_{pop}"] = envelope
        arrays[f"phase_gamma_amplitude_{pop}"] = np.array(
            [envelope[selection & (phase_bin == index)].mean() for index in range(18)]
        )
        amplitude = envelope[selection]
        vector = (
            np.sum(amplitude * np.exp(1j * phase[selection])) / amplitude.sum()
            if amplitude.sum()
            else 0j
        )
        band = (freq >= 30) & (freq <= 90)
        summary[pop] = dict(
            mean_firing_rate_hz=float(rate[selection].mean()),
            afferent_mean_hz=float(
                rates[0 if pop == "e" else 1][
                    (times >= cfg["analysis_start_ms"])
                    & (times < cfg.get("analysis_end_ms", cfg["t_ms"] - 500.0))
                ].mean()
            ),
            peak_30_90_hz=float(freq[band][np.argmax(power[band])])
            if power[band].sum()
            else None,
            mean_gamma_amplitude_hz=float(amplitude.mean()),
            amplitude_weighted_phase_concentration=float(abs(vector)),
            preferred_phase_rad=float(np.angle(vector) % (2 * np.pi))
            if abs(vector)
            else None,
            total_recorded_spikes=int(spikes.sum()),
        )
    return arrays, summary


def analyse(identity, *, run_id=None):
    compute = inputs.source(REPO, identity, "compute")
    cfg = inputs.configuration(compute)
    arrays, summaries = {}, {}
    for condition in recipe.CONDITIONS:
        values, summary = measure(
            inputs.recording(compute, condition),
            recipe.condition_configuration(cfg, condition),
        )
        arrays.update({f"{condition}__{key}": value for key, value in values.items()})
        summaries[condition] = summary
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs={"compute": compute},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        np.savez_compressed(run.export / "measurements.npz", **arrays)
        write_json_atomic(
            run.export / "results.json",
            dict(
                schema="exp130.analysis/v1",
                parameters=cfg,
                results=summaries,
                measurements=dict(
                    analysis_interval_ms=[
                        cfg["analysis_start_ms"],
                        cfg.get("analysis_end_ms", cfg["t_ms"] - 500.0),
                    ],
                    spectrum="Welch of 1 ms per-neuron population rates; Hann; 500 ms segments; 50% overlap; raw bin maximum in 30–90 Hz; 2 Hz bin spacing",
                    amplitude="fourth-order Butterworth 30–90 Hz SOS forward-backward filter; Hilbert magnitude; computed on full recording, then cropped away from endpoints",
                    phase=f"prescribed {cfg['theta_hz']:g} Hz sine phase, radians; same reference phase assigned to constant comparison",
                    concentration="absolute amplitude-weighted mean phase vector; descriptive, no surrogate significance test",
                    phase_profile="mean gamma-band amplitude within 18 equal prescribed-phase bins",
                    limitations="one paired seed; rate matching is prescribed input expectation, not realized count or output-rate matching; harmonics of theta-locked bursts can contribute to bandpassed amplitude; no independent gamma mechanism established",
                ),
            ),
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    print(analyse(args.source))


if __name__ == "__main__":
    main()
