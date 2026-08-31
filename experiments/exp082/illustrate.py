"""Compute deterministic exp082 showcase candidates without rerunning the grid."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp082 import evidence, inputs, recipe
from experiments.exp082.inference import Inference, encode_stream, pick_digits
from pingstore.contracts import PingstoreError, write_json_atomic


def _dense(raw, steps) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for prefix, key, width in (
        ("e", "spikes_e", 1024),
        ("i", "spikes_i", 256),
        ("out", "spikes_out", 10),
    ):
        keep = raw[prefix + "_trial"] == 0
        value = np.zeros((steps, width), dtype=np.int8)
        value[raw[prefix + "_t"][keep], raw[prefix + "_cell"][keep]] = 1
        arrays[key] = value
    return arrays


def _predictions(spikes_out, boundaries):
    return [
        int(spikes_out[start:stop].sum(axis=0).argmax())
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def compute(identity, *, run_id=None):
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    contract = evidence.training_contract(bank.export)
    conditions = recipe.SHOWCASE_CONDITIONS
    boundaries = np.cumsum(
        [0, *[int(round(duration / recipe.DT_MS)) for duration, _ in conditions]]
    ).tolist()
    configuration = {
        "schema": "exp082.showcase-selection/v1",
        "conditions": [list(value) for value in conditions],
        "candidate_order": "ascending integer index",
        "digit_seed_base": recipe.SHOWCASE_DIGIT_SEED_BASE,
        "encoding_seed_base": recipe.SHOWCASE_ENCODING_SEED_BASE,
        "candidate_limit": recipe.SHOWCASE_CANDIDATE_LIMIT,
        "targets": recipe.SHOWCASE_TARGETS,
        "training_seed": recipe.SEEDS[0],
    }
    with inputs.execution(
        REPO,
        "compute",
        sources={"bank": bank},
        run_id=run_id,
        configuration=configuration,
        operation="showcase-selection",
    ) as run:
        worker = Inference(bank, run.directory, recipe.configuration())
        train = bank.export / recipe.training_cell_name(recipe.SEEDS[0])
        candidates, selected = [], {}
        for index in range(recipe.SHOWCASE_CANDIDATE_LIMIT):
            digit_seed = recipe.SHOWCASE_DIGIT_SEED_BASE + index
            encoding_seed = recipe.SHOWCASE_ENCODING_SEED_BASE + index
            pixels, labels = pick_digits(
                worker.images, worker.labels, len(conditions), digit_seed
            )
            spikes = encode_stream(
                pixels,
                conditions,
                torch.Generator().manual_seed(encoding_seed),
            )
            raw = worker.simulate(
                train,
                spikes,
                tuple(boundaries[:-1]),
                run.scratch / "candidates" / f"candidate-{index:03d}",
                "rasters",
            )
            arrays: dict[str, np.ndarray] = {
                "pixels": pixels,
                **_dense(raw, len(spikes)),
            }
            predictions = _predictions(arrays["spikes_out"], boundaries)
            correct = [int(a == b) for a, b in zip(labels, predictions, strict=True)]
            summary = {
                "candidate_index": index,
                "digit_seed": digit_seed,
                "encoding_seed": encoding_seed,
                "labels": labels.tolist(),
                "predictions": predictions,
                "correct": correct,
                "n_correct": sum(correct),
            }
            candidates.append(summary)
            for name, target in recipe.SHOWCASE_TARGETS.items():
                if name not in selected and summary["n_correct"] == target:
                    folder = run.export / "streams" / name
                    folder.mkdir(parents=True)
                    np.savez_compressed(
                        folder / "recording.npz", **arrays  # ty: ignore[invalid-argument-type]
                    )
                    write_json_atomic(
                        folder / "stream.json",
                        {
                            "labels": summary["labels"],
                            "boundaries": boundaries,
                            "conditions": configuration["conditions"],
                        },
                    )
                    selected[name] = index
            if set(selected) == set(recipe.SHOWCASE_TARGETS):
                break
        if set(selected) != set(recipe.SHOWCASE_TARGETS):
            raise RuntimeError("showcase candidate limit did not satisfy both targets")
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp082.showcase-selection/v1",
                "configuration": configuration,
                "training_contract": contract,
                "candidates": candidates,
                "selected": selected,
            },
        )
        write_json_atomic(run.scratch / "dataset.json", worker.dataset)
        evidence.validate_showcase(run.export)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="explicit exp022 bank run")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        print(compute(args.source, run_id=args.run_id))
    except (PingstoreError, OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"exp082 illustrate: {exc}\n")


if __name__ == "__main__":
    main()
