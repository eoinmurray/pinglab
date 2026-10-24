"""Measure saved exp082 counts and streams; never run inference or plotting.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse
from typing import Any

import numpy as np
from experiments.exp082 import recipe
from experiments.exp082.custom import evidence, inputs
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def analyse(identity, showcase_identity=None, *, run_id=None):
    source = inputs.source(REPO, identity, "compute")
    cfg, bank, contract = inputs.compute_evidence(REPO, source)
    showcase = (
        inputs.source(REPO, showcase_identity, "compute")
        if showcase_identity is not None and showcase_identity != identity
        else source
    )
    showcase_bank, showcase_record = evidence.showcase_evidence(REPO, showcase)
    if showcase_bank.reference != bank.reference:
        raise PingstoreError("showcase and evaluation use different training banks")
    rows = [evidence.condition(source, j, cfg) for j in recipe.jobs(cfg)]
    preflight = grid_output_preflight(rows)
    streams = {
        name: stream_result(*evidence.stream(source, name))
        for name in ("matched", "variable")
    }
    streams.update(
        {
            name: stream_result(
                *evidence.stream(
                    showcase,
                    name,
                    conditions=showcase_record["configuration"]["conditions"],
                )
            )
            for name in recipe.SHOWCASE_TARGETS
        }
    )
    selected = list(streams["matched"]["correct"]).index(1)
    streams["single_trial"] = first_correct_trial_from_stream(streams["matched"])
    metadata, derived = {}, {}
    for name, stream in streams.items():
        stream.update(display_values(stream))
        metadata[name] = {
            k: v for k, v in stream.items() if not isinstance(v, np.ndarray)
        }
        for key in ("probabilities", "counts", "final_counts"):
            derived[f"{name}_{key}"] = stream[key]
    result = {
        "schema": "exp082.analysis/v2",
        "status": "complete",
        "profile": cfg["profile"],
        "condition_evidence": load_json(source.export / "evidence.json").get(
            "condition_evidence", "per-presentation-counts/v1"
        ),
        "config": {
            k: v
            for k, v in cfg.items()
            if k not in ("schema", "profile", "checkpoint_policy")
        },
        "checkpoint_policy": recipe.CHECKPOINT_POLICY,
        "checkpoint_provenance": contract["checkpoints"],
        "training_cells": [recipe.training_cell_name(s) for s in recipe.SEEDS],
        "training_source": "exp022 variable-rate streaming training",
        "readout": {
            "mode": "spike-count",
            "definition": "total output-LIF spikes over the matched presentation window",
            "reported_activity": "output spike rate in Hz may be derived as count divided by window duration in seconds",
        },
        **{
            name + ("_stream" if name != "single_trial" else ""): meta
            for name, meta in metadata.items()
        },
        "single_trial_segment_index": selected,
        "showcase_selection": {
            key: showcase_record[key]
            for key in ("configuration", "candidates", "selected")
        },
        "grid_per_seed": rows,
        "duration_200ms_psychometric": [r for r in rows if r["duration_ms"] == 200.0],
        "scientific_preflight": {
            "evaluation_grid": preflight,
            **{
                name + "_stream": streams[name]["output_activity"]
                for name in ("matched", "variable")
            },
        },
        "plot_data": plot_data(rows, cfg),
    }
    with inputs.execution(
        REPO,
        "analyse",
        sources={
            "compute": source,
            "bank": bank,
            **({"showcase": showcase} if showcase is not source else {}),
        },
        run_id=run_id,
        configuration={
            "schema": "exp082.analysis/v2",
            "uncertainty": "sample SD / sqrt(3)",
        },
    ) as run:
        write_json_atomic(run.export / "numbers.json", result)
        np.savez_compressed(run.export / "display.npz", **derived)
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument(
        "--showcase-source",
        help="explicit separate showcase for an older split compute",
    )
    p.add_argument("--run-id")
    a = p.parse_args()
    try:
        print(analyse(a.source, a.showcase_source, run_id=a.run_id))
    except (PingstoreError, OSError, KeyError, ValueError, RuntimeError) as exc:
        p.exit(1, f"exp082 analyse: {exc}\n")


def spike_count_logits(
    spikes_out: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    """Output-LIF spike counts over exactly ``[start, stop)``."""
    if stop <= start:
        raise ValueError("spike-count window must contain at least one timestep")
    return spikes_out[start:stop].sum(axis=0)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def output_activity_summary(
    spikes_out: np.ndarray,
    boundaries: list[int],
) -> dict[str, Any]:
    """Summarize output activity over declared presentation windows."""
    counts = [
        spikes_out[start:stop].sum(axis=0)
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    per_presentation = np.asarray(counts, dtype=np.int64)
    totals = per_presentation.sum(axis=1)
    return {
        "n_presentations": len(counts),
        "total_output_spikes": int(totals.sum()),
        "spikes_per_presentation": totals.tolist(),
        "silent_presentations": int((totals == 0).sum()),
        "silent_fraction": float((totals == 0).mean()),
        "class_spike_totals": per_presentation.sum(axis=0).tolist(),
    }


def single_trial_from_stream(
    stream: dict[str, Any],
    segment_index: int = 0,
) -> dict[str, Any]:
    """Extract one independently readable presentation from a stream result."""
    start = int(stream["boundaries"][segment_index])
    stop = int(stream["boundaries"][segment_index + 1])
    trial: dict[str, Any] = {
        "conditions": [stream["conditions"][segment_index]],
        "pixels": np.asarray(stream["pixels"])[segment_index : segment_index + 1],
        "labels": [stream["labels"][segment_index]],
        "predictions": [stream["predictions"][segment_index]],
        "correct": [stream["correct"][segment_index]],
        "boundaries": [0, stop - start],
        **{
            key: np.asarray(stream[key])[start:stop]
            for key in ("spikes_e", "spikes_i", "spikes_out", "probabilities")
        },
    }
    trial["output_activity"] = output_activity_summary(
        trial["spikes_out"], trial["boundaries"]
    )
    return trial


def first_correct_trial_from_stream(stream: dict[str, Any]) -> dict[str, Any]:
    """Select the first successful presentation for explanatory figures."""
    try:
        segment_index = list(stream["correct"]).index(1)
    except ValueError as error:
        raise RuntimeError(
            "matched stream contains no correctly classified trial"
        ) from error
    return single_trial_from_stream(stream, segment_index)


def grid_output_preflight(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report and reject a wholly silent grid-level output readout."""
    n_presentations = sum(int(row["n_total"]) for row in rows)
    n_silent = sum(
        round(float(row["silent_fraction"]) * int(row["n_total"])) for row in rows
    )
    total_spikes = sum(
        float(row["output_spikes_per_presentation"]) * int(row["n_total"])
        for row in rows
    )
    summary = {
        "n_presentations": n_presentations,
        "total_output_spikes": int(round(total_spikes)),
        "silent_presentations": n_silent,
        "silent_fraction": n_silent / n_presentations,
    }
    if summary["total_output_spikes"] == 0:
        raise RuntimeError(
            "exp082 scientific preflight failed: output readout is silent "
            "across the complete duration-rate grid"
        )
    return summary


def condition_row(job, counts, cfg):
    labels = counts["labels"]
    out = counts["out_counts"]
    totals = out.sum(axis=2)
    n = labels.size
    duration = n * job["duration_ms"] / 1000.0
    correct = int((out.argmax(axis=2) == labels).sum())
    return {
        **{k: job[k] for k in ("seed", "duration_ms", "rate_hz")},
        "stream_batch_size": cfg["stream_batch_size"],
        "n_correct": correct,
        "n_total": n,
        "accuracy": correct / n,
        "output_spikes_per_presentation": int(totals.sum()) / n,
        "silent_fraction": int((totals == 0).sum()) / n,
        "class_spike_totals": out.sum(axis=(0, 1)).tolist(),
        "rate_e_hz": int(counts["e_counts"].sum()) / (1024 * duration),
        "rate_i_hz": int(counts["i_counts"].sum()) / (256 * duration),
    }


def stream_result(raw, meta):
    result = {**meta, **raw}
    probabilities = np.zeros((len(raw["spikes_out"]), 10), dtype=np.float32)
    predictions, correct = [], []
    for label, start, stop in zip(
        meta["labels"], meta["boundaries"][:-1], meta["boundaries"][1:], strict=True
    ):
        for t in range(start, stop):
            probabilities[t] = softmax(
                spike_count_logits(raw["spikes_out"], start, t + 1)
            )
        prediction = int(np.argmax(spike_count_logits(raw["spikes_out"], start, stop)))
        predictions.append(prediction)
        correct.append(int(prediction == label))
    result.update(
        probabilities=probabilities,
        predictions=predictions,
        correct=correct,
        output_activity=output_activity_summary(raw["spikes_out"], meta["boundaries"]),
    )
    return result


def plot_data(rows, cfg):
    rates, durations = cfg["psychometric_rates_hz"], cfg["durations_ms"]
    grid = np.zeros((len(rates), len(durations)), dtype=np.float32)
    sem = np.zeros(len(rates), dtype=np.float32)
    means, sems = [], []
    for i, rate in enumerate(rates):
        for j, duration in enumerate(durations):
            values = np.asarray(
                [
                    r["accuracy"]
                    for r in rows
                    if r["rate_hz"] == rate and r["duration_ms"] == duration
                ]
            )
            grid[i, j] = values.mean()
            if duration == cfg["matched_duration_ms"]:
                value = float(values.std(ddof=1) / np.sqrt(len(values)))
                sem[i] = value
                means.append(float(values.mean()))
                sems.append(value)
    return {
        "rates": rates,
        "durations": durations,
        "grid": grid.tolist(),
        "grid_sem": sem.tolist(),
        "means": means,
        "sems": sems,
    }


def display_values(stream):
    counts = np.asarray(stream["spikes_out"]).cumsum(axis=0)
    final = np.asarray(stream["spikes_out"]).sum(axis=0).astype(int)
    winner = int(final.argmax())
    other = final.copy()
    other[winner] = -1
    runner_up = int(other.argmax())
    return {
        "counts": counts,
        "final_counts": final,
        "winner": winner,
        "runner_up": runner_up,
        "margin": int(final[winner] - final[runner_up]),
    }


if __name__ == "__main__":
    main()
