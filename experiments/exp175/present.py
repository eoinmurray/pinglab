"""Plot retained learning curves, without making an independent-test claim."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]

from experiments import theme
from experiments.exp175 import recipe
from pingstore.library import source_run, stage_run, write_json_atomic


def present(identity):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    source = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="analyse"
    )
    summary = json.loads(source.file("summary.json").read_text())
    theme.apply()
    with stage_run(ROOT, recipe.SLUG, "present", inputs={"analyse": source}) as run:
        conditions = [("Single PING", summary, theme.DEEP_RED)]
        if "direct" in summary:
            conditions.insert(
                0, ("Direct relay training", summary["direct"], theme.INK_BLACK)
            )
        fig, axes = plt.subplots(
            1,
            len(conditions),
            figsize=(6 * len(conditions), 3.5),
            squeeze=False,
            layout="constrained",
        )
        for ax, (label, result, color) in zip(axes[0], conditions, strict=True):
            curve = result["curve"]
            ax.plot(
                [r["epoch"] for r in curve],
                [r["validation_loss"] for r in curve],
                color=color,
                marker="o",
                label="Validation",
            )
            trained = [r for r in curve if "train_loss" in r]
            ax.plot(
                [r["epoch"] for r in trained],
                [r["train_loss"] for r in trained],
                color=color,
                linestyle="--",
                label="Training (online)",
            )
            ax.scatter(
                [result["selected_epoch"]],
                [result["selected_validation_mse"]],
                color=color,
                marker="*",
                s=130,
                zorder=4,
                label="Selected",
            )
            ax.set(title=label, xlabel="Epoch", ylabel="Pixel MSE (dimensionless)")
            ax.legend()
        fig.savefig(run.export / "learning-curves.svg")
        fig.savefig(run.export / "learning-curves.png", dpi=180)
        plt.close(fig)

        def examples(filename, output, keys, titles, indices=None):
            with np.load(source.file(filename)) as images:
                indices = (
                    list(range(len(images["targets"])))
                    if indices is None
                    else indices(images)
                )
                fig, axes = plt.subplots(
                    len(indices),
                    len(keys),
                    figsize=(2 * len(keys), 1.6 * len(indices)),
                    squeeze=False,
                    layout="constrained",
                )
                for row, index in enumerate(indices):
                    for col, key in enumerate(keys):
                        axes[row, col].imshow(
                            images[key][index].reshape(28, 28),
                            cmap="gray_r",
                            vmin=0,
                            vmax=1,
                        )
                        axes[row, col].set_xticks([])
                        axes[row, col].set_yticks([])
                    axes[row, 0].set_ylabel(
                        f"Cue {'AB'[images['cue'][index]]} · phase {images['phase'][index] + 1}"
                    )
                for ax, title in zip(axes[0], titles, strict=True):
                    ax.set_title(title)
                fig.savefig(run.export / f"{output}.png", dpi=150)
                fig.savefig(run.export / f"{output}.svg")
                plt.close(fig)

        if "direct" in summary:
            examples(
                "direct-reconstructions.npz",
                "reconstructions",
                ("images_A", "images_B", "targets", "reconstructed"),
                ("Image A", "Image B", "Target", "Direct relay"),
            )
        else:
            examples(
                "reconstructions.npz",
                "reconstructions",
                ("images_A", "images_B", "targets", "reconstructed"),
                ("Image A", "Image B", "Target", "Single PING"),
            )
        if "transfer" in summary:

            def first_targets(images):
                ids = list(dict.fromkeys(images["original_ids"].tolist()))[:2]
                return np.flatnonzero(np.isin(images["original_ids"], ids)).tolist()

            examples(
                "transfer-reconstructions.npz",
                "transfer-reconstructions",
                ("images_A", "images_B", "targets", "single", "relay"),
                ("Image A", "Image B", "Target", "Single PING", "Transferred relay"),
                first_targets,
            )
            values = summary["transfer"]["comparison"]
            fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
            ax.bar(
                ["Single PING", "Transferred relay"],
                [values[k]["pixel_mse"] for k in ("single", "relay")],
                color=[theme.INK_BLACK, theme.DEEP_RED],
            )
            ax.set(ylabel="Validation pixel MSE (dimensionless)")
            fig.savefig(run.export / "transfer-comparison.svg")
            plt.close(fig)
            shutil.copyfile(
                source.file("relay-circuit.svg"), run.export / "relay-circuit.svg"
            )
        shutil.copyfile(source.file("circuit.svg"), run.export / "circuit.svg")
        write_json_atomic(run.export / "numbers.json", summary)
        write_json_atomic(run.export / "summary.json", summary)
        write_json_atomic(
            run.export / "timing.json",
            dict(total_seconds=time.perf_counter() - started),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    present(parser.parse_args().source)
