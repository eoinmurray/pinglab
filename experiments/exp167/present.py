"""Combine both saved rasters in one figure without executing a network."""

import argparse
import shutil
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments import theme
from experiments.exp167.recipe import SLUG
from pingstore.library import load_json, source_run, stage_run, write_json_atomic


def raster_panel(ax, path, cfg, label):
    with np.load(path, allow_pickle=False) as z:
        for pop, color in (("E", theme.INK_BLACK), ("I", theme.DEEP_RED)):
            offset = 0 if pop == "E" else cfg["n_e"]
            ax.scatter(
                z[pop + "_time_ms"],
                z[pop + "_cell"] + offset,
                s=0.1,
                c=color,
                linewidths=0,
                rasterized=True,
            )
    ax.axhline(cfg["n_e"], color=theme.MUTED, lw=0.5, ls="--")
    ax.set(
        xlim=cfg["measurement_ms"],
        ylim=(0, cfg["n_e"] + cfg["n_i"]),
        xlabel="Time after simulation start (ms)",
        ylabel="Cell index",
        title=label,
    )


def present(source, *, run_id=None):
    started = perf_counter()
    upstream = source_run(REPO / ".pingstore", source, experiment=SLUG, stage="analyse")
    pin = upstream.record["inputs"]["compute"]
    compute = source_run(
        REPO / ".pingstore",
        pin["run_id"],
        reference=pin,
        experiment=SLUG,
        stage="compute",
    )
    numbers = load_json(upstream.file("numbers.json"))
    cfg = numbers["configuration"]
    imported = numbers["evidence"]["kind"] == "historical-scratch-import"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme.apply()
    with stage_run(
        REPO,
        SLUG,
        "present",
        inputs={"analyse": upstream},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        figure, axes = plt.subplots(2, 1, figsize=(9, 6.8), layout="constrained")
        for index, (ax, label) in enumerate(zip(axes, cfg["conditions"], strict=True)):
            if imported:
                # Keep the saved raster pixels; replace the source's clipped labels.
                pixels = plt.imread(compute.file(f"{label}/raster.png"))
                if pixels.shape[:2] != (440, 1200):
                    raise ValueError("unexpected historical raster dimensions")
                ax.imshow(
                    pixels[74:322, 161:1169],
                    extent=(50, 1050, 0, 5000),
                    origin="upper",
                    aspect="auto",
                    interpolation="nearest",
                )
                ax.set(
                    xlabel="Time after simulation start (ms)",
                    ylabel="Cell index",
                    title=f"{chr(65 + index)}  {label}",
                    xlim=(50, 1050),
                    ylim=(0, 5000),
                )
                ax.set_yticks([0, 2000, 4000, 5000])
            else:
                raster_panel(
                    ax,
                    compute.file(f"{label}/events.npz"),
                    cfg,
                    f"{chr(65 + index)}  {label}",
                )
                shutil.copyfile(
                    compute.file(f"{label}/circuit.svg"),
                    run.export / f"{label}--circuit.svg",
                )
        for extension in ("png", "svg", "pdf"):
            figure.savefig(run.export / f"state-rasters.{extension}", dpi=200)
        plt.close(figure)
        write_json_atomic(
            run.export / "numbers.json", {**numbers, "figure": "state-rasters.svg"}
        )
        run.record["execution"]["elapsed_s"] = perf_counter() - started
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    present(**vars(parser.parse_args()))
