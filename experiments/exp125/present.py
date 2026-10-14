"""Render training curves from an explicit Scikit Digits analysis."""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pinglab-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.exp125.recipe import SLUG
from experiments.helpers.figsave import save_figure
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def present(identity):
    source = source_run(REPO / ".pingstore", identity, stage="analyse", experiment=SLUG)
    numbers = load_json(source.file("results.json"))
    cfg = source.record["execution"]["configuration"]
    numbers["execution"] = source.record["execution"]["source_measurements"]
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "svg.fonttype": "none"})
    with stage_run(REPO, SLUG, "present", inputs={"analyse": source}, configuration=cfg) as run:
        fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 76 / 25.4), constrained_layout=True)
        rows = numbers["epochs"]
        epochs = [row["epoch"] for row in rows]
        for split, color in (("train", "#202020"), ("validation", "#a34d45")):
            axes[0].plot(epochs, [row[split]["loss"] for row in rows], color=color, label=split.capitalize())
            axes[1].plot(epochs, [100 * row[split]["accuracy"] for row in rows], color=color)
        axes[0].set(title="A  Classification loss", ylabel="Cross-entropy")
        axes[1].set(title="B  Classification accuracy", ylabel="Accuracy (%)", ylim=(0, 102))
        axes[1].axhline(10, color="#777777", linestyle=":", linewidth=0.8)
        for axis in axes:
            axis.axvline(numbers["selected_epoch"], color="#777777", linestyle="--", linewidth=0.8)
            axis.set(xlabel="Completed epoch", xlim=(0, cfg["epochs"]))
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False)
        save_figure(fig, run.export / "training_curves")
        fig.savefig(run.scratch / "training_curves.png", dpi=160)
        # A preview is disposable; the scientific exports remain vector files.
        import shutil
        shutil.copyfile(run.scratch / "training_curves.png", "/tmp/exp125-training-curves.png")
        plt.close(fig)
        write_json_atomic(run.export / "numbers.json", numbers)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    present(parser.parse_args().source)
