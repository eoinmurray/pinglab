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


def analyse(identity):
    source = source_run(REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG)
    cfg = source.record["execution"]["configuration"]
    if cfg.get("schema") not in ("exp124.recipe/v1", "exp124.recipe/v2"):
        raise ValueError("unsupported SHD recipe")
    with np.load(source.file("spikes.npz"), allow_pickle=False) as data:
        labels, split, indices = data["labels"], data["split"], data["official_train_indices"]
        if len(np.unique(indices)) != len(indices):
            raise ValueError("overlapping source examples")
        for code, per_class in ((0, cfg["train_per_class"]), (1, cfg["validation_per_class"])):
            if not np.array_equal(np.bincount(labels[split == code], minlength=cfg["classes"]), np.full(cfg["classes"], per_class)):
                raise ValueError("incorrect class-balanced partition")
    conditions = {}
    for condition in cfg["models"]:
        learning = load_json(source.unit(condition) / "learning.json")
        trainable = tuple(learning["weight_change_l2"])
        readout_name = "readout_projection.weight"
        hidden_names = tuple(name for name in trainable if name != readout_name)
        rows = learning["epochs"]
        if [row["epoch"] for row in rows] != list(range(cfg["epochs"] + 1)):
            raise ValueError("incomplete epoch history")
        with np.load(source.unit(condition) / "initial_weights.npz") as initial, np.load(source.unit(condition) / "final_weights.npz") as final:
            changes = {name: float(np.linalg.norm(final[name] - initial[name])) for name in trainable}
            if not all(np.isfinite(final[name]).all() for name in final.files):
                raise ValueError("nonfinite final weights")
        conditions[condition] = {
            **learning, "weight_change_l2": changes,
            "learned_weight_changes": {name: changes[name] > 0 for name in trainable},
            "learned_nonzero_gradients": {name: all(row["gradient_l2_mean"][name] > 0 for row in rows[1:]) for name in trainable},
            "training_loss_decreased": rows[-1]["train"]["loss"] < rows[0]["train"]["loss"],
            "hidden_weights_changed": all(changes[name] > 0 for name in hidden_names),
            "readout_weights_changed": changes[readout_name] > 0,
            "nonzero_hidden_gradients": all(row["gradient_l2_mean"][name] > 0 for row in rows[1:] for name in hidden_names),
        }
    matched_pair = set(cfg["models"]) == set(recipe.MODELS)
    if matched_pair:
        with np.load(source.unit("coba") / "initial_weights.npz") as coba, np.load(source.unit("ping") / "initial_weights.npz") as ping:
            if any(not np.array_equal(coba[name], ping[name]) for name in trainable):
                raise ValueError("conditions have unmatched initial learned weights")
    with stage_run(REPO, recipe.SLUG, "analyse", inputs={"compute": source}, configuration=cfg) as run:
        write_json_atomic(run.export / "results.json", {"conditions": conditions})
        run.record["execution"]["source_measurements"] = {
            "models": source.record["execution"]["models"],
            "dataset": source.record["execution"]["dataset"],
            "origin": source.record["origin"],
            "snnlab_version": source.record["execution"]["snnlab_version"],
            "matched_pair_checked": matched_pair,
        }
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    analyse(parser.parse_args().source)
