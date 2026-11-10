"""Summarize frozen TR02 (no rate penalty) population firing rates from verified spike counts."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
from experiments.exp173 import recipe
from pingstore.library import source_run, stage_run, write_json_atomic


def summarize(counts, population_size, duration_ms):
    if (
        counts.ndim != 2
        or not counts.size
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
    ):
        raise ValueError("Invalid image-by-draw population counts")
    rates = counts.astype(np.float64) / population_size / (duration_ms / 1000)
    image_rates = rates.mean(axis=0)
    return dict(
        mean_hz=float(image_rates.mean()),
        image_sd_hz=float(image_rates.std(ddof=1)) if len(image_rates) > 1 else None,
        images=len(image_rates),
        draws=counts.shape[0],
        total_spikes=int(counts.sum()),
    )


def analyse(identity):
    started = time.perf_counter()
    source = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="compute"
    )
    cfg = source.record["execution"]["configuration"]
    rows = []
    for n_i in cfg["pool_sizes"]:
        with np.load(
            source.file(f"i{n_i}", "counts.npz"), allow_pickle=False
        ) as counts:
            rows.append(
                dict(
                    n_i=n_i,
                    ei_multiplier=1,
                    ie_multiplier=cfg["original_n_i"] / n_i,
                    E=summarize(counts["E"], cfg["n_e"], cfg["duration_ms"]),
                    I=summarize(counts["I"], n_i, cfg["duration_ms"]),
                )
            )
    with np.load(source.file("dataset.npz"), allow_pickle=False) as dataset:
        labels = dataset["accuracy_labels"]
    for row in rows:
        with np.load(
            source.file(f"i{row['n_i']}", "accuracy.npz"), allow_pickle=False
        ) as values:
            predictions = values["predictions"]
            if predictions.shape != labels.shape or not np.array_equal(
                values["correct"], predictions == labels
            ):
                raise ValueError("Accuracy evidence mismatch")
            correct = int((predictions == labels).sum())
            row["test_accuracy"] = dict(
                correct=correct, images=len(labels), percent=100 * correct / len(labels)
            )
    with stage_run(
        ROOT,
        recipe.SLUG,
        "analyse",
        inputs={"compute": source},
        configuration={"aggregation": cfg["rate_aggregation"]},
    ) as run:
        write_json_atomic(
            run.export / "numbers.json",
            dict(
                schema="exp173.analysis/v1",
                configuration=cfg,
                rates=rows,
                compute_seconds=source.record["execution"]["elapsed_s"],
                condition_seconds=source.record["execution"]["condition_seconds"],
            ),
        )
        run.record["execution"]["elapsed_s"] = time.perf_counter() - started
    print(run.run_id, flush=True)
    print(json.dumps(rows, indent=2))
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    analyse(parser.parse_args().source)
