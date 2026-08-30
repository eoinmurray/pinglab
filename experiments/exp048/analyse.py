"""Replay saved streams and aggregate measurements; no simulator or dataset access."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp048 import evidence, inputs, measurements, recipe
from pingstore.contracts import PingstoreError, write_json_atomic

MEASUREMENT = {
    "schema": "exp048.measurement/v1",
    "readout": "matched trailing mean",
    "aggregation": "seed mean and sample SEM; legacy float32 grid/tau",
    "output_delay_steps": 1,
    "output_time_constant_default_ms": 2.0,
}


def decode(job, raw, stimulus, w, tau_out):
    e = evidence.dense(raw, "e", recipe.N_E)
    steps = [round(tau / recipe.DT) for tau, _ in job["segments"]]
    ends = np.cumsum(steps) - 1
    if job["kind"] == "varying":
        v = measurements._v_out_series(e, w, tau_out)
        csum = np.concatenate(
            [np.zeros((1, recipe.N_CLASSES), dtype=np.float32), np.cumsum(v, axis=0)]
        )
        logits = np.zeros_like(v)
        cur = 0
        for length in steps:
            for t in range(cur, cur + length):
                lo = max(0, t + 1 - length)
                logits[t] = (csum[t + 1] - csum[lo]) / max(1, t + 1 - lo)
            cur += length
    else:
        logits = measurements.sliding_readout(e, w, tau_out, job["segments"][0][0])
    probs = measurements.softmax_rowwise(logits)
    # The fixed headline historically used softmax argmax; sweeps and the
    # varying headline selected directly from logits. Preserve that distinction.
    pred = (probs if job["kind"] == "headline" else logits).argmax(axis=-1)
    labels = stimulus["labels"]
    correct = (pred[ends] == labels).astype(int)
    summary = {
        "labels": labels.tolist(),
        "seg_preds": pred[ends].tolist(),
        "seg_correct": correct.tolist(),
    }
    figure = None
    if job["kind"] in ("headline", "varying"):
        figure = {
            "segments": np.array(job["segments"]),
            "segment_steps": np.array(steps),
            "T_stream_steps": np.array(len(e)),
            "labels": labels,
            "pixels": stimulus["pixels"],
            "spk_e": e,
            "spk_i": evidence.dense(raw, "i", recipe.N_I),
            "probs": probs,
            "pred_per_t": pred,
            "seg_preds": pred[ends],
            "seg_correct": correct,
            "seg_ends": ends,
        }
        if job["kind"] == "headline":
            figure.update(
                tau_ms=np.array(job["segments"][0][0]),
                n_digits=np.array(len(steps)),
                tau_steps=np.array(steps[0]),
                input_rate_hz=np.array(job["segments"][0][1]),
            )
    return int(correct.sum()), len(correct), summary, figure


def analyse(identity, *, run_id=None):
    candidate = inputs.lineage(REPO, identity)[identity]
    if candidate.record["execution"].get("configuration") == evidence.IMPORT:
        return evidence.analyse_retained(REPO, candidate, run_id=run_id)
    source = inputs.source(REPO, identity, "compute")
    bank, contract = inputs.compute_evidence(REPO, source)
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": source, "bank": bank},
        run_id=run_id,
        configuration=MEASUREMENT,
    ) as run:
        weights = {}
        for seed in recipe.SEEDS:
            weights[seed] = evidence.load_arrays(source.export / f"readout-{seed}.npz")[
                "W_out"
            ]
            evidence.readout(weights[seed])
        tau_rows, grid_rows, low_rows, headlines, stream_rows = [], [], [], {}, []
        for job in recipe.jobs():
            correct = total = 0
            for index in range(job["streams"]):
                directory = source.unit(job["id"], f"stream-{index:03d}")
                raw, stimulus = evidence.stream(directory, job, index)
                cfg = contract["configs"][recipe.cell_name(job["seed"])]
                n_correct, n_total, summary, figure = decode(
                    job,
                    raw,
                    stimulus,
                    weights[job["seed"]],
                    float(cfg.get("tau_out_ms", 2.0)),
                )
                correct += n_correct
                total += n_total
                stream_rows.append({"job": job["id"], "stream": index, **summary})
                if figure is not None:
                    np.savez_compressed(run.export / f"{job['kind']}.npz", **figure)
                    headlines[job["kind"]] = summary
            row = {
                "tau_ms": job["segments"][0][0],
                "input_rate_hz": job["segments"][0][1],
                "train_seed": job["seed"],
                "n_correct": correct,
                "n_total": total,
            }
            if job["kind"] == "tau":
                tau_rows.append(
                    {
                        **row,
                        "acc": 100.0 * correct / total,
                        "rate_compensate": job["rate_compensate"],
                        "n_streams": job["streams"],
                        "n_per_stream": len(job["segments"]),
                    }
                )
            elif job["kind"] == "grid":
                grid_rows.append({**row, "acc": 100.0 * correct / total})
            elif job["kind"] == "low":
                low_rows.append({**row, "accuracy": correct / total})
        # Legacy numbers place all constant rows before all compensated rows.
        tau_rows = [r for r in tau_rows if not r["rate_compensate"]] + [
            r for r in tau_rows if r["rate_compensate"]
        ]
        tau_agg = measurements.aggregate_tau_rows(
            [r for r in tau_rows if not r["rate_compensate"]]
        )
        tau_agg += measurements.aggregate_tau_rows(
            [r for r in tau_rows if r["rate_compensate"]]
        )
        grid_agg = measurements.aggregate_grid_rows(grid_rows)
        low_agg = []
        for rate in recipe.LOW_RATE_HZ:
            rows = [r for r in low_rows if r["input_rate_hz"] == rate]
            values = np.array([r["accuracy"] for r in rows])
            low_agg.append(
                {
                    "tau_ms": recipe.TRAINED_T_MS,
                    "input_rate_hz": rate,
                    "accuracy": float(values.mean()),
                    "accuracy_sem": float(values.std(ddof=1) / math.sqrt(len(values))),
                    "n_seeds": len(values),
                    "n_total": sum(r["n_total"] for r in rows),
                    "source": "exp048 low-rate sweep",
                }
            )
        result = {
            "schema": "exp048.analysis/v1",
            "measurement": MEASUREMENT,
            "config": recipe.configuration(),
            "checkpoint_provenance": contract["checkpoints"],
            "headline": {
                k: headlines["headline"][k] for k in ("labels", "seg_correct")
            },
            "varying_headline": {
                "segments": [list(s) for s in recipe.VARYING_HEADLINE],
                **headlines["varying"],
            },
            "tau_sweep_per_seed": tau_rows,
            "tau_sweep_agg": tau_agg,
            "grid_sweep_per_seed": grid_rows,
            "grid_sweep_agg": grid_agg,
            "encoding_rate_psychometric": {
                "presentation_ms": recipe.TRAINED_T_MS,
                "trained_rate_hz": recipe.INPUT_RATE_HZ,
                "new_rates_hz": recipe.LOW_RATE_HZ,
                "new_streams_per_seed": recipe.LOW_RATE_STREAMS,
                "digits_per_stream": recipe.LOW_RATE_DIGITS_PER_STREAM,
                "per_seed_new_cells": low_rows,
                "curve": measurements.rate_curve(grid_agg, low_agg),
            },
        }
        write_json_atomic(run.export / "results.json", result)
        write_json_atomic(run.export / "segments.json", {"streams": stream_rows})
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        analyse(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp048 analyse: {exc}\n")


if __name__ == "__main__":
    main()
