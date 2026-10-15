"""Measure a named SHD compute run without training or rendering."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
from experiments.exp124 import recipe
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def analyse(identities):
    identities = [identities] if isinstance(identities, str) else identities
    sources = [source_run(REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG)
               for identity in identities]
    if not sources:
        raise ValueError("explicit compute inputs are required")
    source = sources[0]
    cfg = dict(source.record["execution"]["configuration"])
    scientific = {key: value for key, value in cfg.items() if key != "models"}
    model_sources = {}
    for candidate in sources:
        candidate_cfg = candidate.record["execution"]["configuration"]
        if {key: value for key, value in candidate_cfg.items() if key != "models"} != scientific:
            raise ValueError("compute inputs have different scientific configurations")
        for key in ("dataset", "snnlab_version"):
            if candidate.record["execution"][key] != source.record["execution"][key]:
                raise ValueError(f"compute inputs have different {key}")
        with np.load(source.file("spikes.npz"), allow_pickle=False) as first, np.load(candidate.file("spikes.npz"), allow_pickle=False) as other:
            if set(first.files) != set(other.files) or any(not np.array_equal(first[key], other[key]) for key in first.files):
                raise ValueError("paired compute inputs have different stimulus data")
        for model in candidate_cfg["models"]:
            if model in model_sources:
                raise ValueError(f"duplicate compute condition: {model}")
            model_sources[model] = candidate
    cfg["models"] = [model for model in recipe.MODELS if model in model_sources]
    if len(sources) > 1 and set(model_sources) != set(recipe.MODELS):
        raise ValueError("paired compute inputs must cover COBA and PING exactly")
    if cfg.get("schema") not in ("exp124.recipe/v1", "exp124.recipe/v2", "exp124.recipe/v3"):
        raise ValueError("unsupported SHD recipe")
    with np.load(source.file("spikes.npz"), allow_pickle=False) as data:
        labels, split, indices = data["labels"], data["split"], data["official_train_indices"]
        if len(np.unique(indices)) != len(indices):
            raise ValueError("overlapping source examples")
        for code, per_class in ((0, cfg["train_per_class"]), (1, cfg["validation_per_class"])):
            expected = (cfg["train_class_counts" if code == 0 else "validation_class_counts"]
                        if cfg.get("full_dataset", False) else np.full(cfg["classes"], per_class))
            if not np.array_equal(np.bincount(labels[split == code], minlength=cfg["classes"]), expected):
                raise ValueError("incorrect class-balanced partition")
        dataset = source.record["execution"]["dataset"]
        if not np.isin(split, (0, 1)).all():
            raise ValueError("invalid split codes")
        if cfg.get("full_dataset", False) and len(indices) != dataset["official_train_count"]:
            raise ValueError("full-data result does not cover official training set")
        if cfg.get("full_dataset", False) and ((indices < 0).any() or (indices >= dataset["official_train_count"]).any()):
            raise ValueError("invalid official training index")
    conditions = {}
    for condition in cfg["models"]:
        source = model_sources[condition]
        learning = load_json(source.unit(condition) / "learning.json")
        trainable = tuple(learning["weight_change_l2"])
        readout_name = "readout_projection.weight"
        hidden_names = tuple(name for name in trainable if name != readout_name)
        rows = learning["epochs"]
        loss_baseline = next(row for row in rows if row["train"]["loss"] is not None)
        measured = {name: all(row["gradient_l2_mean"].get(name) is not None for row in rows[1:])
                    for name in trainable}
        if [row["epoch"] for row in rows] != list(range(cfg["epochs"] + 1)):
            raise ValueError("incomplete epoch history")
        with np.load(source.unit(condition) / "initial_weights.npz") as initial, np.load(source.unit(condition) / "final_weights.npz") as final:
            changes = {name: float(np.linalg.norm(final[name] - initial[name])) for name in trainable}
            if not all(np.isfinite(final[name]).all() for name in final.files):
                raise ValueError("nonfinite final weights")
        conditions[condition] = {
            **learning, "weight_change_l2": changes,
            "learned_weight_changes": {name: changes[name] > 0 for name in trainable},
            "learned_nonzero_gradients": {name: all(row["gradient_l2_mean"][name] > 0 for row in rows[1:]) if measured[name] else None for name in trainable},
            "training_loss_decreased": rows[-1]["train"]["loss"] < loss_baseline["train"]["loss"],
            "training_loss_baseline_epoch": loss_baseline["epoch"],
            "hidden_weights_changed": all(changes[name] > 0 for name in hidden_names),
            "readout_weights_changed": changes[readout_name] > 0,
            "nonzero_hidden_gradients": all(row["gradient_l2_mean"][name] > 0 for row in rows[1:] for name in hidden_names) if all(measured[name] for name in hidden_names) else None,
        }
    matched_pair = set(cfg["models"]) == set(recipe.MODELS)
    if matched_pair:
        with np.load(model_sources["coba"].unit("coba") / "initial_weights.npz") as coba, np.load(model_sources["ping"].unit("ping") / "initial_weights.npz") as ping:
            if any(not np.array_equal(coba[name], ping[name]) for name in trainable):
                raise ValueError("conditions have unmatched initial learned weights")
    inputs = ({"compute": sources[0]} if len(sources) == 1 else
              {f"compute_{model}": candidate for model, candidate in model_sources.items()})
    with stage_run(REPO, recipe.SLUG, "analyse", inputs=inputs, configuration=cfg) as run:
        write_json_atomic(run.export / "results.json", {"conditions": conditions})
        if len(sources) == 1 and "sample_raster_configuration" in sources[0].record["execution"]:
            from experiments.exp124.rasters import measure
            write_json_atomic(run.export / "raster_measurements.json",
                              measure(sources[0], sources[0].record["execution"]["sample_raster_configuration"]))
        run.record["execution"]["source_measurements"] = {
            "models": {model: candidate.record["execution"]["models"][model]
                       for model, candidate in model_sources.items()},
            "dataset": source.record["execution"]["dataset"],
            "origin": sources[0].record["execution"].get("scientific_origin", sources[0].record["origin"]) if len(sources) == 1 else "paired-independent-workers",
            "condition_origins": {model: candidate.record["execution"].get("condition_origins", {}).get(model, candidate.record["origin"]) for model, candidate in model_sources.items()},
            "snnlab_version": source.record["execution"]["snnlab_version"],
            "matched_pair_checked": matched_pair,
        }
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    analyse(parser.parse_args().source)
