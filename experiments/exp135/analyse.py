"""Summarise retained training measurements without generating new activity."""

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp135 import recipe
from experiments.helpers.network_artifacts import snapshot_parameters
from pingstore.contracts import file_sha256, write_json_atomic
from pingstore.stages import source_run, stage_run
from snnlab.analysis import spike_autocorrelogram


def summarise(history, configuration):
    epochs = history["epochs"]
    complete = [e for e in epochs if e["epoch"] > 0]
    if [e["epoch"] for e in epochs] != list(range(configuration["epochs"] + 1)):
        raise ValueError("training history does not cover the requested epochs")
    for row in epochs:
        for key in (
            "train_loss",
            "validation_loss",
            "train_accuracy",
            "validation_accuracy",
        ):
            if key in row and not math.isfinite(row[key]):
                raise ValueError(f"nonfinite measurement: {key}")
    data = configuration["resolved_dataset"]
    first, last = complete[0], complete[-1]
    return dict(
        schema="exp135.case-analysis/v2",
        epochs=epochs,
        n_train=len(data["splits"]["train"]),
        n_validation=len(data["splits"]["validation"]),
        source_training_samples=data["source_samples"],
        actual_train_fraction=len(data["splits"]["train"]) / data["source_samples"],
        presentation_ms=configuration.get("crop_ms", data["presentation_ms"]),
        initial_validation_loss=epochs[0]["validation_loss"],
        final_validation_loss=last["validation_loss"],
        final_validation_accuracy=last["validation_accuracy"],
        first_online_train_loss=first["train_loss"],
        final_online_train_loss=last["train_loss"],
        online_train_loss_decreased=last["train_loss"] < first["train_loss"],
        selection=history["selection"],
        timing=history["timing"],
        official_test_used=False,
        limitation="single-seed development trial; online training measurements use changing weights; validation has one sample per class; fixed centred validation crops; random training crops",
    )


def parameter_changes(upstream, model):
    start = snapshot_parameters(upstream, model, "training", "start")
    end = snapshot_parameters(upstream, model, "training", "end")
    if set(start) != set(end):
        raise ValueError("start/end parameter inventory mismatch")
    changed = {}
    for name in start:
        if start[name].shape != end[name].shape or not np.isfinite(end[name]).all():
            raise ValueError(f"invalid final parameters: {name}")
        changed[name] = int(np.count_nonzero(start[name] != end[name]))
    return start, changed


def sample_measurements(upstream, model, cfg):
    sample = json.loads(upstream.file(model, "sample.json").read_text())
    expected = min(
        cfg["resolved_dataset"]["splits"]["validation"], key=lambda r: r["sample_id"]
    )
    if (
        sample["sample_id"] != expected["sample_id"]
        or sample["label"] != expected["label"]
        or sample["checkpoint_role"] != "final"
        or sample["model"] != model
        or sample["presentation_ms"] != cfg["resolved_dataset"]["presentation_ms"]
    ):
        raise ValueError(
            "sample does not match the predeclared paired final-weight protocol"
        )
    graph = json.loads(
        upstream.file(model, "training", "network.bundle", "graph.json").read_text()
    )
    populations = {
        row["id"]: row["size"]
        for row in graph["populations"]
        if row["id"] in recipe.POPULATIONS[model]
    }
    if populations != sample["populations"]:
        raise ValueError("sample population sizes differ from the retained network")
    steps = round(sample["presentation_ms"] / cfg["dt_ms"])
    measured = dict(sample, activity={})
    with np.load(upstream.file(model, "spikes.npz"), allow_pickle=False) as spikes:
        scores = spikes["scores"]
        if scores.shape != (cfg["n_classes"],) or not np.isfinite(scores).all():
            raise ValueError("invalid sample class scores")
        measured["prediction"] = int(scores.argmax())
        measured["correct"] = measured["prediction"] == sample["label"]
        inputs = spikes["input_events"].copy()
        for pop, size in populations.items():
            events = spikes[pop]
            if (
                events.ndim != 2
                or events.shape[1] != 3
                or not np.issubdtype(events.dtype, np.integer)
                or np.any(events < 0)
                or np.any(events[:, 0] >= steps)
                or np.any(events[:, 1] != 0)
                or np.any(events[:, 2] >= size)
            ):
                raise ValueError(f"invalid sparse spikes for {pop}")
            counts = np.bincount(events[:, 0], minlength=steps)
            # Summed counts are exactly the library estimator's population reduction.
            lags, values = spike_autocorrelogram(
                counts[:, None],
                cfg["dt_ms"],
                max_lag_ms=recipe.ACF_MAX_LAG_MS,
                bin_ms=1.0,
            )
            measured["activity"][pop] = dict(
                spike_count=len(events),
                mean_rate_hz=len(events) / size / (sample["presentation_ms"] / 1000),
                autocorrelation=dict(
                    lag_ms=lags.tolist(),
                    correlation=[float(v) if np.isfinite(v) else None for v in values],
                ),
            )
    measured["parameter_count"] = sum(
        math.prod(p["shape"]) for p in graph["parameters"]
    )
    measured["population_geometry"] = {
        p["id"]: p.get("spatial_shape") for p in graph["populations"]
    }
    return measured, inputs


def parameter_table(upstream, report):
    """Derive the comparison from executed definitions and the frozen selection."""
    cfg = report["configuration"]
    definitions = {}
    for model in recipe.MODELS:
        definitions[model] = {
            role: json.loads(
                upstream.file(model, "training", "network.bundle", role).read_text()
            )
            for role in ("graph.json", "training.json")
        }
    graph = definitions["ping"]["graph.json"]
    training = definitions["ping"]["training.json"]
    populations = {p["id"]: p for p in graph["populations"]}
    projections = {p["id"]: p for p in graph["projections"]}
    parameters = {p["id"]: p for p in graph["parameters"]}
    e, i = (populations[name]["neuron"] for name in ("block1_E", "block1_I"))

    def shape(values):
        return " × ".join(str(n) for n in values)

    def initial(name):
        values = parameters[name]["initializer"]["config"]
        return f"{values['mean']:g} / {values['std']:g}"

    def tau(name):
        return f"{projections[name]['synapse']['tau']['value']:g} ms"

    rows = [
        [
            "Input",
            f"{shape(graph['inputs'][0]['spatial_shape'])} polarity / spatial channels",
            "Same",
        ],
        [
            "Hidden neuron model",
            "Conductance-based leaky integrate-and-fire (COBA LIF)",
            "Same for E and I",
        ],
    ]
    for layer in (1, 2):
        ep, ip = (populations[f"block{layer}_{kind}"] for kind in ("E", "I"))
        e_shape = f"{shape(ep['spatial_shape'])} E ({ep['size']:,} neurons)"
        rows.append(
            [
                f"Hidden layer {layer}",
                e_shape,
                f"{e_shape}; {shape(ip['spatial_shape'])} I ({ip['size']:,} neurons)",
            ]
        )
    rows.extend(
        [
            [
                "Feedforward pathways",
                "Input → E1 → average pool → E2 → average pool → readout",
                "Same",
            ],
            [
                "Feedforward kernels / pooling",
                "3 × 3 kernels, padding 1 / 2 × 2 average pools, stride 2",
                "Same",
            ],
            [
                "Within-layer feedback",
                "None",
                "Learned 1 × 1 E→I and I→E kernels in both layers",
            ],
            [
                "Feedback delay",
                "Not used",
                f"{projections['block1_E_to_I']['delay']['value']:g} ms in each direction",
            ],
            [
                "Readout",
                f"{populations['readout']['size']} non-spiking leaky-integrator units; {populations['readout']['neuron']['tau']['value']:g} ms decay",
                "Same",
            ],
            [
                "Readout synapse",
                f"Signed leaky-integrator filter; {tau('readout_projection')} decay",
                "Same",
            ],
            [
                "Classification score",
                "Mean readout voltage over the complete padded presentation",
                "Same",
            ],
            [
                {"math": r"C_m\;\text{(membrane capacitance)}"},
                f"E: {e['capacitance_nf']:g} nF",
                f"E: {e['capacitance_nf']:g} nF; I: {i['capacitance_nf']:g} nF",
            ],
            [
                {"math": r"g_L\;\text{(leak conductance)}"},
                f"E: {e['leak_us']:g} μS",
                f"E: {e['leak_us']:g} μS; I: {i['leak_us']:g} μS",
            ],
            [
                {"math": r"\tau_{\mathrm{passive}}=C_m/g_L"},
                f"E: {e['capacitance_nf'] / e['leak_us']:g} ms (derived)",
                f"E: {e['capacitance_nf'] / e['leak_us']:g} ms; I: {i['capacitance_nf'] / i['leak_us']:g} ms (derived)",
            ],
            [
                "Rest / reset / initial voltage",
                f"{e['resting_mv']:g} / {e['reset_mv']:g} / {e['initial_voltage_mv']:g} mV",
                "Same for E and I",
            ],
            ["Spike threshold", f"{e['threshold_mv']:g} mV", "Same for E and I"],
            [
                "Refractory period",
                f"E: {e['refractory']['value']:g} ms",
                f"E: {e['refractory']['value']:g} ms; I: {i['refractory']['value']:g} ms",
            ],
            [
                "AMPA decay",
                f"Feedforward: {tau('block1_input_to_E')}",
                f"Feedforward: {tau('block1_input_to_E')}; E→I: {tau('block1_E_to_I')}",
            ],
            ["GABA decay", "Not used", tau("block1_I_to_E")],
            [
                "Simulation timestep / presentation",
                f"{graph['timebase']['dt']['value']:g} ms / {training['presentation_duration']['value']:g} ms",
                "Same",
            ],
            [
                "Learned projections",
                "Three: input→E1, pooled E1→E2, pooled E2→readout",
                "Seven: same three plus four reciprocal kernels",
            ],
            [
                "Trainable parameters",
                *[
                    report["models"][m]["sample"]["parameter_count"]
                    for m in recipe.MODELS
                ],
            ],
            [
                "Training / validation examples",
                f"{report['models']['coba']['n_train']} / {report['models']['coba']['n_validation']}",
                "Same; subject-disjoint",
            ],
            [
                "Epochs / batch size / training seed",
                f"{cfg['epochs']} / {cfg['batch_size']} / {cfg['seed']}",
                "Same",
            ],
            ["Selection seed", cfg["selection_seed"], "Same"],
            [
                "Optimizer / objective",
                f"{training['optimizer']['kind'].replace('adamw', 'AdamW')} / example-average cross-entropy",
                "Same",
            ],
            [
                "Learning rate / weight decay / gradient clipping",
                f"{training['parameter_groups'][0]['lr']:g} / {training['optimizer']['config']['weight_decay']:g} / {training['gradient_clip']:g}",
                "Same",
            ],
            ["Fast-sigmoid surrogate slope", training["surrogate"]["slope"], "Same"],
            [
                "Voltage-gradient damping divisor",
                e["voltage_grad_dampen"],
                "Same for E and I",
            ],
            [
                "Feedforward initial Gaussian mean / SD",
                f"{initial('block1_input_to_E.weight')} μS, lower-clamped at zero",
                "Identical draws",
            ],
            [
                "Feedback initial Gaussian mean / SD",
                "Not used",
                f"E→I: {initial('block1_E_to_I.weight')} μS; I→E: {initial('block1_I_to_E.weight')} μS, lower-clamped at zero",
            ],
            [
                "Signed readout initial Gaussian mean / SD",
                initial("readout_projection.weight"),
                "Identical draws",
            ],
            ["Initialization scaling", "Direct; no fan-in normalization", "Same"],
        ]
    )
    return rows


def analyse(source, *, run_id=None):
    upstream = source_run(
        REPO / ".pingstore", source, stage="compute", experiment=recipe.SLUG
    )
    cfg = upstream.record["execution"]["configuration"]
    if cfg.get("local_check"):
        raise ValueError(
            "local-check diagnostics cannot supply the full-gesture comparison report"
        )
    if cfg.get("models") != list(recipe.MODELS):
        raise ValueError("paired comparison requires both committed cases")
    for name, expected in cfg["dataset_files"].items():
        if file_sha256(upstream.file(name)) != expected:
            raise ValueError("retained dataset differs from the worker input pins")
    report = dict(
        schema="exp135.analysis/v2",
        configuration=cfg,
        models={},
        paired_initialization=True,
        official_test_used=False,
    )
    starts, inputs = {}, {}
    for model in recipe.MODELS:
        history = json.loads(upstream.file(model, "training-history.json").read_text())
        summary = summarise(history, cfg)
        starts[model], changed = parameter_changes(upstream, model)
        sample, inputs[model] = sample_measurements(upstream, model, cfg)
        report["models"][model] = dict(
            summary,
            sample=sample,
            changed_parameter_elements=changed,
            any_weights_changed=any(changed.values()),
        )
    if not np.array_equal(inputs["coba"], inputs["ping"]):
        raise ValueError("models received different sample inputs")
    for name in recipe.COMMON_WEIGHTS:
        if not np.array_equal(starts["coba"][name], starts["ping"][name]):
            raise ValueError(f"common initialization differs: {name}")
    report["parameter_table"] = parameter_table(upstream, report)
    with stage_run(
        REPO, recipe.SLUG, "analyse", inputs={"training": upstream}, run_id=run_id
    ) as run:
        write_json_atomic(run.export / "analysis.json", report)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    analyse(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
