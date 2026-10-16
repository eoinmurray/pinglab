"""Measure paired SHD training, report parameters and one matched response."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
from experiments.exp124 import recipe
from experiments.helpers.network_artifacts import snapshot_parameters
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run
from snnlab.analysis import spike_autocorrelogram


def parameters(source, model, stage):
    if source.file(model, "train", stage, "snapshot.json").exists():
        return snapshot_parameters(source, model, "train", stage)
    # The report's immutable r149 compute predates native snapshot retention.
    filename = "initial_weights.npz" if stage == "start" else "final_weights.npz"
    with np.load(source.file(model, filename), allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def network_definition(source, model, role):
    native = source.file(model, "train", "network.bundle", role)
    return load_json(native if native.exists() else source.file(model, role))


def parameter_table(source):
    """Read the executed graph/training definitions, never today's recipe defaults."""
    cfg = source.record["execution"]["configuration"]
    graph = network_definition(source, "ping", "graph.json")
    coba = network_definition(source, "coba", "graph.json")
    training = network_definition(source, "ping", "training.json")
    populations = {p["id"]: p for p in graph["populations"]}
    projections = {p["id"]: p for p in graph["projections"]}
    pars = {p["id"]: p for p in graph["parameters"]}
    operations = {p["id"]: p for p in graph["operations"]}
    e, i = (populations[key]["neuron"] for key in ("E1", "I1"))
    normalization = operations["normalized_scores"]["config"]["settings"]
    dt = graph["timebase"]["dt"]["value"]
    duration = training["presentation_duration"]["value"]
    gain = pars["score_gain.weight"]["initializer"]["config"]["gain"]
    ne, ni = populations["E1"]["size"], populations["I1"]["size"]
    coba_sizes = {p["id"]: p["size"] for p in coba["populations"]}
    groups = {g["id"]: g for g in training["parameter_groups"]}

    def number(value):
        return f"{value:g}"

    rows = [
        [
            "Input",
            f"{graph['inputs'][0]['shape'][-1]} SHD cochlear spike channels [1]",
            "Same",
        ],
        [
            "Hidden neuron model",
            "Conductance-based leaky integrate-and-fire (COBA LIF)",
            "Same for both E and I",
        ],
        [
            "Synapses onto hidden neurons",
            "Exponentially decaying AMPA conductances",
            "Exponentially decaying AMPA and GABA conductances",
        ],
        *[
            [
                f"Hidden layer {layer}",
                f"{coba_sizes[f'E{layer}']} excitatory neurons",
                f"{populations[f'E{layer}']['size']} excitatory + {populations[f'I{layer}']['size']} inhibitory neurons",
            ]
            for layer in (1, 2)
        ],
        ["Feedforward pathways", "Input → E1 → E2 → readout", "Same"],
        [
            "Within-layer feedback",
            "None",
            "Fixed E→I and I→E connections in each layer",
        ],
        [
            "Readout",
            f"{populations['readout']['size']} non-spiking leaky-integrator units",
            "Same",
        ],
        [
            "Readout synapse",
            f"Signed leaky-integrator filter, {number(projections['readout_projection']['synapse']['tau']['value'])} ms decay",
            "Same",
        ],
        [
            "Classification score",
            f"{number(gain)} × mean readout voltage × {number(normalization['reference_rate'] * 1000 / dt)} Hz / max(mean E2 rate, {number(normalization['minimum_rate'] * 1000 / dt)} Hz)",
            "Same",
        ],
        [
            {"math": r"C_m\;\text{(membrane capacitance)}"},
            f"E: {number(e['capacitance_nf'])} nF",
            f"E: {number(e['capacitance_nf'])} nF; I: {number(i['capacitance_nf'])} nF",
        ],
        [
            {"math": r"g_L\;\text{(leak conductance)}"},
            f"E: {number(e['leak_us'])} μS",
            f"E: {number(e['leak_us'])} μS; I: {number(i['leak_us'])} μS",
        ],
        [
            {"math": r"\tau_{\mathrm{passive}}=C_m/g_L"},
            f"{number(e['capacitance_nf'] / e['leak_us'])} ms (derived)",
            f"{number(i['capacitance_nf'] / i['leak_us'])} ms for both E and I (derived)",
        ],
        [
            {"math": r"\tau_{\mathrm{AMPA}}"},
            f"{number(projections['input_to_E1']['synapse']['tau']['value'])} ms",
            "Same",
        ],
        [
            {"math": r"\tau_{\mathrm{GABA}}"},
            "Not used",
            f"{number(projections['I1_to_E1']['synapse']['tau']['value'])} ms",
        ],
        [
            "Refractory period",
            f"{number(e['refractory']['value'])} ms",
            f"{number(i['refractory']['value'])} ms",
        ],
        [
            "Simulation timestep / presentation",
            f"{number(dt)} ms / {number(duration / 1000)} s",
            "Same",
        ],
        [
            "Learned weights",
            "Input→E1, E1→E2, E2→readout",
            "Same; inhibitory feedback remains fixed",
        ],
        [
            "Training / validation utterances",
            f"{cfg['train_examples']:,} / {cfg['validation_examples']:,}",
            "Same",
        ],
        [
            "Epochs / batch size / seed",
            f"{cfg['epochs']} / {cfg['batch_size']} / {cfg['seed']}",
            "Same",
        ],
        ["Optimizer / objective", "Adam / equal-class cross-entropy", "Same"],
        [
            "Learning rates: input / interlayer / readout",
            " / ".join(
                number(groups[name]["lr"])
                for name in ("input", "interlayer", "readout")
            ),
            "Same",
        ],
        [
            "Input insertion / deletion",
            f"{number(cfg['input_noise_hz'])} Hz per channel / {number(cfg['spike_deletion_probability'])} probability",
            "Same",
        ],
        [
            "Initial hidden weights: Gaussian parent mean / SD",
            f"{number(pars['input_to_E1.weight']['initializer']['mean'])} / {number(pars['input_to_E1.weight']['initializer']['std'])} μS",
            "Same",
        ],
        [
            "Initial zero fractions: input / interlayer",
            " / ".join(
                number(pars[key]["initializer"]["initial_zero_fraction"])
                for key in ("input_to_E1.weight", "E1_to_E2.weight")
            ),
            "Same",
        ],
        [
            "Fixed summed feedback: E→I / I→E",
            "None",
            f"{number(pars['E1_to_I1.weight']['initializer']['value'] * ne)} / {number(pars['I1_to_E1.weight']['initializer']['value'] * ni)} μS",
        ],
    ]
    return rows


def spike_autocorrelation(spikes, dt_ms, max_lag_ms=200):
    """Serialize SNNLab's population autocorrelogram; undefined bins are null."""
    lags, values = spike_autocorrelogram(
        spikes, dt_ms, max_lag_ms=max_lag_ms, bin_ms=1.0
    )
    return {
        "lag_ms": lags.tolist(),
        "correlation": [
            float(value) if np.isfinite(value) else None for value in values
        ],
    }


def raster_measurements(source):
    cfg = source.record["execution"]["sample_raster_configuration"]
    measured = {
        "selection": cfg["sample_selection"],
        "class": cfg["sample_class"],
        "official_train_index": cfg["official_train_index"],
        "noise_seed": cfg["inference_noise_seed"],
        "duration_ms": cfg["duration_ms"],
        "dt_ms": cfg["dt_ms"],
        "models": {},
    }
    filename = (
        "recording.npz" if source.file("recording.npz").exists() else "rasters.npz"
    )
    with np.load(source.file(filename), allow_pickle=False) as arrays:
        steps = round(cfg["duration_ms"] / cfg["dt_ms"])
        for model, populations in recipe.POPULATIONS.items():
            scores = arrays[f"{model}_scores"]
            if scores.shape != (20,) or not np.isfinite(scores).all():
                raise ValueError("invalid sample scores")
            counts, autocorrelations = {}, {}
            for pop in populations:
                spikes = arrays[f"{model}_{pop}"]
                if (
                    spikes.shape != (steps, 32 if pop.startswith("i") else 128)
                    or not np.isin(spikes, (0, 1)).all()
                ):
                    raise ValueError("invalid sample spikes")
                counts[pop] = {
                    "neurons": spikes.shape[1],
                    "spikes": int(spikes.sum()),
                    "rate_hz": float(spikes.mean() * 1000 / cfg["dt_ms"]),
                }
                autocorrelations[pop] = spike_autocorrelation(spikes, cfg["dt_ms"])
            prediction = int(scores.argmax())
            measured["models"][model] = {
                "prediction": prediction,
                "correct": prediction == cfg["sample_class"],
                "populations": counts,
                "autocorrelations": autocorrelations,
            }
    return measured


def measurements(source):
    cfg = source.record["execution"]["configuration"]
    if cfg.get("schema") not in ("exp124.recipe/v3", "exp124.recipe/v4") or cfg[
        "models"
    ] != list(recipe.MODELS):
        raise ValueError("expected complete paired calibrated SHD training")
    selection = (
        "selection.npz" if source.file("selection.npz").exists() else "spikes.npz"
    )
    with np.load(source.file(selection), allow_pickle=False) as data:
        labels, split, indices = (
            data[key] for key in ("labels", "split", "official_train_indices")
        )
        if len(np.unique(indices)) != len(labels) or not np.isin(split, (0, 1)).all():
            raise ValueError("invalid or overlapping dataset selection")
        for code, name in enumerate(("train", "validation")):
            if not np.array_equal(
                np.bincount(labels[split == code], minlength=20),
                cfg[f"{name}_class_counts"],
            ):
                raise ValueError(
                    "dataset class counts differ from executed configuration"
                )
        if (
            cfg["full_dataset"]
            and len(labels)
            != source.record["execution"]["dataset"]["official_train_count"]
        ):
            raise ValueError("incomplete full dataset")
        sample = source.record["execution"]["sample_raster_configuration"]
        first = np.flatnonzero(split == 1)[0]
        if (sample["official_train_index"], sample["sample_class"]) != (
            indices[first],
            labels[first],
        ):
            raise ValueError("raster is not the first validation utterance")
    conditions, initial = {}, {}
    for model in recipe.MODELS:
        start, end = (
            parameters(source, model, "start"),
            parameters(source, model, "end"),
        )
        if set(start) != set(end) or any(
            not np.isfinite(v).all() for v in end.values()
        ):
            raise ValueError("invalid final parameter inventory")
        changes = {
            key: float(np.linalg.norm(end[key] - start[key]))
            for key in recipe.TRAINABLE
        }
        if not all(changes.values()) or any(
            not np.array_equal(start[key], end[key])
            for key in start
            if key not in recipe.TRAINABLE
        ):
            raise ValueError(
                "learned weights did not change or fixed parameters changed"
            )
        initial[model] = start
        filename = (
            "metrics.json"
            if source.file(model, "metrics.json").exists()
            else "native_metrics.json"
        )
        epochs = load_json(source.file(model, filename))["epochs"]
        if [row["epoch"] for row in epochs] != list(range(cfg["epochs"] + 1)):
            raise ValueError("incomplete epoch history")
        rows = [
            {
                "epoch": row["epoch"],
                **{
                    split: {
                        "loss": next(
                            iter((row.get(f"{split}_cross_entropies") or {}).values()),
                            None,
                        ),
                        "accuracy": row.get(f"{split}_accuracy"),
                    }
                    for split in ("train", "validation")
                },
            }
            for row in epochs
        ]
        for row in rows:
            for split in ("train", "validation"):
                loss, accuracy = row[split].values()
                if split == "train" and row["epoch"] == 0:
                    continue
                if (
                    loss is None
                    or accuracy is None
                    or not np.isfinite(loss)
                    or not 0 <= accuracy <= 1
                ):
                    raise ValueError("invalid epoch metrics")
        conditions[model] = {
            "epochs": rows,
            "weight_change_l2": changes,
            "learned_weight_changes": {
                key: value > 0 for key, value in changes.items()
            },
        }
    if any(
        not np.array_equal(initial["coba"][key], initial["ping"][key])
        for key in recipe.TRAINABLE
    ):
        raise ValueError("paired initial learned weights differ")
    return {"conditions": conditions, "parameter_table": parameter_table(source)}


def analyse(identity):
    source = source_run(
        REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    results, sample = measurements(source), raster_measurements(source)
    execution = source.record["execution"]
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs={"compute": source},
        configuration=execution["configuration"],
    ) as run:
        write_json_atomic(run.export / "results.json", results)
        write_json_atomic(run.export / "raster_measurements.json", sample)
        run.record["execution"]["source_measurements"] = {
            key: execution[key] for key in ("models", "dataset", "snnlab_version")
        }
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    analyse(parser.parse_args().source)
