"""Measure explicit compute evidence; never simulate, render or publish."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "tools")]

import numpy as np
from experiments.exp099 import inputs, recipe
from pingstore.contracts import PingstoreError, write_json_atomic
from pingstore.stages import stage_run
from tools.snnsim.metrics import (  # noqa: TID251
    rhythmicity_metrics,
    rolling_conductance_loop_score,
)
from tools.snnviz import exponential_trace  # noqa: TID251


def measure(record, settings: dict) -> tuple[dict, dict]:
    data, dt = record.signals, record.dt_ms
    e, i = data["spk_e"].astype(bool), data["spk_i"].astype(bool)
    half_ms = settings["rhythm_window_ms"] / 2
    centres = np.arange(
        half_ms, record.duration_ms - half_ms + 1e-9, settings["rhythm_stride_ms"]
    )
    half_steps = round(half_ms / dt)
    contrast = np.empty(len(centres))
    for index, centre_ms in enumerate(centres):
        centre = round(centre_ms / dt)
        value = rhythmicity_metrics(
            e[centre - half_steps : centre + half_steps],
            dt,
            max_lag_ms=settings["rhythm_max_lag_ms"],
            bin_ms=settings["rhythm_bin_ms"],
        )["contrast"]
        contrast[index] = 0.0 if value is None else float(value)
    arrays = {"rhythm_centres": centres, "rhythm_contrast": contrast}
    for output, signal in (
        ("mean_v_e", "v_e_1"),
        ("mean_v_i", "v_i_1"),
        ("mean_g_e", "ge_e_1"),
        ("mean_g_i", "gi_e_1"),
    ):
        arrays[output] = data[signal].mean(1)
    for label, signal in (
        ("E AMPA", "input_excitatory_e_executed"),
        ("E GABA", "input_inhibitory_e_executed"),
        ("I AMPA", "input_excitatory_i_executed"),
        ("I GABA", "input_inhibitory_i_executed"),
    ):
        arrays[label] = exponential_trace(
            data[signal], dt_ms=dt, tau_ms=settings["external_tau_ms"][label]
        ).mean(1)
    view = slice(
        round(settings["view_start_ms"] / dt), round(settings["view_end_ms"] / dt)
    )
    loop = rolling_conductance_loop_score(
        arrays["mean_g_e"][view],
        arrays["mean_g_i"][view],
        dt,
        window_ms=settings["loop_window_ms"],
        stride_ms=settings["loop_stride_ms"],
    )
    samples = max(1, round(settings["loop_smoothing_ms"] / settings["loop_stride_ms"]))
    smooth = np.convolve(loop["raw"], np.ones(samples) / samples, mode="same")
    lo, hi = np.percentile(smooth, settings["loop_percentiles"])
    arrays.update(
        loop_times=settings["view_start_ms"] + loop["times_ms"],
        loop_raw=loop["raw"],
        loop_raw_smooth=smooth,
        loop_display=np.clip((smooth - lo) / max(float(hi - lo), 1e-12), 0, 1),
    )
    # Preserve the original summary's exclusive endpoint, distinct from the
    # production renderer's inclusive final 400 ms window.
    summary_mask = centres < record.duration_ms - half_ms
    summary_rhythm, summary_times = contrast[summary_mask], centres[summary_mask]
    summary = {
        "condition": "richer-input",
        "e_spikes": int(e.sum()),
        "i_spikes": int(i.sum()),
        "peak_rhythmicity": float(summary_rhythm.max()),
        "peak_rhythmicity_time_ms": float(summary_times[np.argmax(summary_rhythm)]),
    }
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise PingstoreError("nonfinite exp099 measurements")
    return arrays, summary


def analyse(identity: str, *, run_id: str | None = None) -> str:
    compute = inputs.source(REPO, identity, "compute")
    cfg = inputs.configuration(compute)
    if compute.record["inputs"]:
        raise PingstoreError("standalone exp099 compute must have no upstream inputs")
    settings = recipe.analysis_configuration()
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs={"compute": compute},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        run.record["execution"]["measurements"] = settings
        arrays, summary = measure(inputs.recording(compute), settings)
        np.savez_compressed(run.export / "measurements.npz", **arrays)
        write_json_atomic(
            run.export / "results.json",
            {
                "schema": "exp099.analysis/v1",
                "parameters": cfg,
                "measurements": settings,
                "question": "Does richer input preserve or destabilize a reference PING state?",
                "results": {"richer-input": summary},
                "disposition": "draft",
            },
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp099 v4 compute run ID"
    )
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    analyse(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
