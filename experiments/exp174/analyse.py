"""Measure cohort accuracy, sustained threshold time and population spike budgets."""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
from experiments.exp174 import recipe
from pingstore.library import source_run, stage_run, write_json_atomic


def threshold_index(accuracy, threshold):
    """First member of the final uninterrupted above-threshold suffix."""
    accuracy = np.asarray(accuracy)
    if accuracy.ndim != 1 or not accuracy.size or not np.isfinite(accuracy).all():
        raise ValueError("Accuracy must be a finite time series")
    stable = np.logical_and.accumulate((accuracy >= threshold)[::-1])[::-1]
    hits = np.flatnonzero(stable)
    return int(hits[0]) if len(hits) else None


def summarize(predictions, labels, counts, cfg):
    shape = (cfg["steps"], cfg["samples"])
    if (
        predictions.shape != shape
        or labels.shape != (cfg["samples"],)
        or not np.isin(predictions, np.arange(10)).all()
    ):
        raise ValueError("Unexpected prediction evidence")
    accuracy = (predictions == labels[None]).mean(1)
    curves = {"accuracy": accuracy}
    for name, values in counts.items():
        if (
            values.shape != shape
            or not np.isfinite(values).all()
            or (values < 0).any()
            or not np.equal(values, np.rint(values)).all()
        ):
            raise ValueError("Invalid population counts")
        curves[name] = values.astype(np.float64).mean(1).cumsum()
    index = threshold_index(accuracy, cfg["threshold_accuracy"])
    first = np.flatnonzero(accuracy >= cfg["threshold_accuracy"])
    row = dict(
        final_accuracy_pct=float(100 * accuracy[-1]),
        correct=int((predictions[-1] == labels).sum()),
        first_crossing_ms=float((first[0] + 1) * cfg["dt_ms"]) if len(first) else None,
        latency_ms=float((index + 1) * cfg["dt_ms"]) if index is not None else None,
        e_spikes_at_threshold=float(curves["e_spikes"][index])
        if index is not None
        else None,
        i_spikes_at_threshold=float(curves["i_spikes"][index])
        if index is not None
        else None,
        artificial_events_at_threshold=float(curves["artificial_events"][index])
        if index is not None
        else None,
        e_spikes_at_end=float(curves["e_spikes"][-1]),
        i_spikes_at_end=float(curves["i_spikes"][-1]),
        artificial_events_at_end=float(curves["artificial_events"][-1]),
    )
    return row, curves


def analyse(identity):
    started = time.perf_counter()
    source = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="compute"
    )
    cfg = source.record["execution"]["configuration"]
    rows, curves = [], {"time_ms": (np.arange(cfg["steps"]) + 1) * cfg["dt_ms"]}
    reference_ids = reference_labels = None
    for kind, label in zip(recipe.KINDS, recipe.LABELS, strict=True):
        with np.load(source.file(kind, "temporal.npz"), allow_pickle=False) as data:
            ids, labels = data["ids"], data["labels"]
            if reference_ids is not None:
                np.testing.assert_array_equal(ids, reference_ids)
                np.testing.assert_array_equal(labels, reference_labels)
            reference_ids, reference_labels = ids, labels
            np.testing.assert_array_equal(
                data["predictions"][-1], data["endpoint_scores"].argmax(-1)
            )
            if kind != "ping" and data["i_counts"].any():
                raise ValueError("Biological I must be silent outside PING")
            if kind != "artificial" and data["artificial_events"].any():
                raise ValueError("Artificial events must be absent in baselines")
            row, series = summarize(
                data["predictions"],
                labels,
                {
                    "e_spikes": data["e_counts"],
                    "i_spikes": data["i_counts"],
                    "artificial_events": data["artificial_events"],
                },
                cfg,
            )
            rows.append(dict(kind=kind, label=label, **row))
            curves.update({kind + "_" + key: value for key, value in series.items()})
    with stage_run(
        ROOT,
        recipe.SLUG,
        "analyse",
        inputs={"compute": source},
        configuration={"threshold_accuracy": cfg["threshold_accuracy"]},
    ) as run:
        np.savez_compressed(run.export / "curves.npz", **curves)
        write_json_atomic(
            run.export / "numbers.json",
            {
                "configuration": cfg,
                "conditions": rows,
                "timings": {"compute": source.record["execution"]["elapsed_seconds"]},
                "limitations": [
                    f"{cfg['samples']}-image retained test subset, one model seed and one encoding/inhibition draw",
                    "Sustained threshold is retrospective cohort accuracy, not an image-wise online stopping rule",
                    "E spikes are not total computational or energy cost",
                    "CPU Poisson draw differs from original GPU draw; retained sensory stream matches exactly",
                    "Original baseline and artificial-I refractory timing conventions differ",
                ],
            },
        )
        run.record["execution"]["elapsed_seconds"] = time.perf_counter() - started
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    analyse(parser.parse_args().source)
