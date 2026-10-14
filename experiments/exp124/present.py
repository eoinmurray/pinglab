"""Render learning curves from explicit local and/or Modal analyses."""

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pinglab-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.exp124 import recipe
from matplotlib.ticker import MaxNLocator
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def present(identities, reference=None):
    sources = [source_run(REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG) for identity in identities]
    cfg = sources[0].record["execution"]["configuration"]
    if any(source.record["execution"]["configuration"] != cfg for source in sources):
        raise ValueError("cannot compare different smoke recipes")
    selected, numbers = {}, {"configuration": cfg, "executions": {}}
    selection = None
    for source in sources:
        metadata = source.record["execution"]["source_measurements"]
        label = "modal" if metadata["origin"].startswith("modal-") else "local"
        if label in selected:
            raise ValueError("only one explicit analysis per backend is supported")
        digest = metadata["dataset"]["selection_sha256"]
        if selection is not None and selection != digest:
            raise ValueError("backend examples differ")
        selection = digest
        selected[label] = source
        numbers["executions"][label] = {"conditions": load_json(source.file("results.json"))["conditions"], **metadata}
    if set(numbers["executions"]) == {"local", "modal"}:
        local, remote = (numbers["executions"][key] for key in ("local", "modal"))
        if local["dataset"]["archive_sha256"] != remote["dataset"]["archive_sha256"]:
            raise ValueError("backend source archives differ")
        if local["snnlab_version"] != remote["snnlab_version"]:
            raise ValueError("backend snnlab versions differ")
        numbers["backend_comparison"] = {
            condition: {
                "max_loss_difference": max(abs(a[split]["loss"] - b[split]["loss"])
                                           for a, b in zip(local["conditions"][condition]["epochs"], remote["conditions"][condition]["epochs"], strict=True)
                                           for split in ("train", "validation")),
                "max_accuracy_difference": max(abs(a[split]["accuracy"] - b[split]["accuracy"])
                                               for a, b in zip(local["conditions"][condition]["epochs"], remote["conditions"][condition]["epochs"], strict=True)
                                               for split in ("train", "validation")),
            } for condition in cfg["models"]
        }
    retained = source_run(REPO / ".pingstore", reference, stage="present", experiment=recipe.SLUG) if reference else None
    if retained:
        selected["smoke_reference"] = retained
        numbers["smoke"] = load_json(retained.file("numbers.json"))
    with stage_run(REPO, recipe.SLUG, "present", inputs=selected, configuration=cfg) as run:
        if retained:
            for condition in recipe.MODELS:
                for extension in ("svg", "png"):
                    filename = f"{condition}_training_curves.{extension}"
                    shutil.copyfile(retained.file(filename), run.export / f"smoke_{filename}")
        for condition in cfg["models"]:
            figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
            for backend, execution in numbers["executions"].items():
                rows = execution["conditions"][condition]["epochs"]
                for split, color in (("train", "#202020"), ("validation", "#a34d45")):
                    style = "-" if backend == "local" else "--"
                    epochs = [row["epoch"] for row in rows]
                    marker = "o" if backend == "local" else "x"
                    axes[0].plot(epochs, [row[split]["loss"] for row in rows], style, marker=marker, markersize=3, color=color, label=f"{backend.capitalize()} {split}")
                    axes[1].plot(epochs, [100 * row[split]["accuracy"] for row in rows], style, marker=marker, markersize=3, color=color)
            axes[0].set(title="A  Classification loss", ylabel="Cross-entropy")
            max_accuracy = max(row[split]["accuracy"] * 100
                               for execution in numbers["executions"].values()
                               for row in execution["conditions"][condition]["epochs"]
                               for split in ("train", "validation"))
            axes[1].set(title="B  Classification accuracy", ylabel="Accuracy (%)",
                        ylim=(0, max(20, max_accuracy * 1.2)))
            axes[1].axhline(5, linestyle=":", color="#777777", linewidth=1)
            for axis in axes:
                axis.set(xlabel="Completed epoch", xlim=(0, cfg["epochs"]))
                axis.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
                axis.spines[["top", "right"]].set_visible(False)
            axes[0].legend(fontsize=8, frameon=False)
            figure.suptitle(f"{condition.upper()} — SHD, {cfg['train_per_class'] * cfg['classes']} training / {cfg['validation_per_class'] * cfg['classes']} validation", fontsize=11)
            for extension in ("svg", "png"):
                figure.savefig(run.export / f"{condition}_training_curves.{extension}", dpi=180)
            plt.close(figure)
        write_json_atomic(run.export / "numbers.json", numbers)
    return run.run_id


def present_diagnostics(identities, reference=None, selected_source=None):
    sources = [source_run(REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG) for identity in identities]
    cases = []
    selections = {}
    for source in sources:
        cfg = source.record["execution"]["configuration"]
        counts = (cfg['train_per_class'] * cfg['classes'], cfg['validation_per_class'] * cfg['classes'])
        if counts not in ((20, 20), (200, 100)):
            raise ValueError("diagnostic panels require 20/20 or 200/100 examples")
        key = (*counts, cfg['seed'])
        dataset = source.record["execution"]["source_measurements"]["dataset"]
        selection = (dataset['archive_sha256'], dataset['selection_sha256'])
        if key in selections and selections[key] != selection:
            raise ValueError("matched diagnostic settings use different examples")
        selections[key] = selection
        cases.append({"analysis_run": source.record["run_id"], "configuration": cfg, "conditions": load_json(source.file("results.json"))["conditions"],
                      "measurements": source.record["execution"]["source_measurements"]})
    selected_case = next((case for case in cases if case['analysis_run'] == selected_source), None)
    if selected_source and selected_case is None:
        raise ValueError("selected diagnostic must be an explicit analysis input")
    if set(counts[:2] for counts in selections) != {(20, 20), (200, 100)}:
        raise ValueError("diagnostic presentation requires tiny and pilot evidence")
    selected_cases = []
    if selected_case:
        matched_configuration = {key: value for key, value in selected_case['configuration'].items()
                                 if key not in ('seed', 'variant')}
        selected_cases = [case for case in cases
                          if {key: value for key, value in case['configuration'].items()
                              if key not in ('seed', 'variant')} == matched_configuration]
    diagnostic_numbers = {"cases": cases, "selected_case": selected_case,
                          "selected_cases": selected_cases}
    inputs = {f"case_{index}": source for index, source in enumerate(sources)}
    retained = source_run(REPO / ".pingstore", reference, stage="present", experiment=recipe.SLUG) if reference else None
    if retained:
        inputs["retained_one_layer"] = retained
    with stage_run(REPO, recipe.SLUG, "present", inputs=inputs,
                   configuration={"comparison": "tiny-subset overfitting and reduced pilot",
                                  "selected_source": selected_source}) as run:
        if retained:
            for path in retained.export.iterdir():
                if path.is_file() and path.name != "numbers.json":
                    shutil.copyfile(path, run.export / path.name)
        for condition in (model for model in recipe.MODELS if any(model in case["conditions"] for case in cases)):
            fig, grid = plt.subplots(2, 2, figsize=(7.1, 6.5))
            axes = grid.ravel()
            for index, case in enumerate(cases):
                if condition not in case["conditions"]:
                    continue
                cfg = case["configuration"]
                rows = case["conditions"][condition]["epochs"]
                count = cfg['train_per_class'] * cfg['classes']
                if count == 20:
                    label = f"{cfg.get('variant', 'baseline')}, seed {cfg['seed']}"
                    if cfg.get("readout_learning_rate", cfg["learning_rate"]) != cfg["learning_rate"]:
                        label += ", faster readout"
                    if cfg.get('variant') == 'tuned':
                        label += f", {cfg['epochs']} updates"
                else:
                    label = cfg.get('readout_reduction', 'mean')
                    if cfg.get('readout_gain', 1) != 20:
                        label += f", gain {cfg.get('readout_gain', 1):g}"
                    if not cfg.get("balanced_batches", False):
                        label += ", shuffled"
                    if not cfg.get("center_readout_initialization", False):
                        label += ", uncentered"
                    if cfg.get("normalize_readout", False):
                        label += ", rate normalized"
                    if cfg['learning_rate'] != 0.0004:
                        label += f", hidden LR {cfg['learning_rate']:g}"
                    if cfg.get("interlayer_weight_mean", 1.8) != 1.8:
                        label += f", interlayer {cfg['interlayer_weight_mean']:g}"
                    if cfg.get("input_weight_mean", 0.9) != 0.9:
                        label += f", input {cfg['input_weight_mean']:g}"
                    if cfg['seed'] != 42:
                        label += f", seed {cfg['seed']}"
                steps = [(row["epoch"] * ((cfg["train_per_class"] * cfg["classes"] + cfg["batch_size"] - 1) // cfg["batch_size"])) for row in rows]
                offset = 0 if count == 20 else 2
                color = plt.get_cmap("tab10")(index % 10)
                axes[offset].plot(steps, [100 * row['train']['accuracy'] for row in rows], label=label, color=color, linewidth=1.2)
                axes[offset + 1].plot(steps, [100 * row['validation']['accuracy'] for row in rows], color=color, linewidth=1.2)
            titles = ("A  Tiny training (20 examples)", "B  Tiny validation (20 examples)",
                      "C  Pilot training (200 examples)", "D  Pilot validation (100 examples)")
            for axis, title in zip(axes, titles):
                axis.set(title=title, xlabel="Optimizer updates", ylabel="Accuracy (%)", ylim=(0, 105))
                axis.title.set_fontsize(8)
                axis.xaxis.label.set_fontsize(8)
                axis.yaxis.label.set_fontsize(8)
                axis.tick_params(labelsize=7)
                axis.set_box_aspect(3 / 4)
                axis.axhline(5, linestyle=":", color="#777777", linewidth=1)
                axis.spines[["top", "right"]].set_visible(False)
            for axis in (axes[0], axes[2]):
                axis.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2)
            fig.subplots_adjust(hspace=0.8, wspace=0.3, top=0.91, bottom=0.24)
            fig.suptitle(f"{condition.upper()} — SHD training diagnostics", fontsize=11)
            for extension in ("svg", "png"):
                fig.savefig(run.export / f"{condition}_diagnostics.{extension}", dpi=180, bbox_inches="tight")
            plt.close(fig)
        write_json_atomic(run.export / "diagnostic_numbers.json", diagnostic_numbers)
        if retained:
            numbers = load_json(retained.file("numbers.json"))
            numbers["diagnostics"] = diagnostic_numbers
            write_json_atomic(run.export / "numbers.json", numbers)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--reference-present", help="Explicit earlier smoke presentation to retain alongside new curves")
    parser.add_argument("--diagnostics", action="store_true", help="Compare explicit diagnostic analyses across recipes")
    parser.add_argument("--selected-source", help="Explicit diagnostic analysis whose final metrics are highlighted")
    args = parser.parse_args()
    if args.diagnostics:
        present_diagnostics(args.source, args.reference_present, args.selected_source)
    else:
        if args.selected_source:
            parser.error("selected-source requires diagnostics")
        present(args.source, args.reference_present)
