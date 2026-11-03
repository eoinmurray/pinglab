"""Render the selected two-state raster comparison from retained spike events."""

import argparse
from pathlib import Path
import shutil
import sys
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments import theme
from experiments.exp168.recipe import SLUG, CONDITIONS
from pingstore.library import source_run, stage_run, load_json, write_json_atomic


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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme.apply()
    with stage_run(
        REPO,
        SLUG,
        "present",
        inputs={"analyse": upstream},
        configuration=upstream.record["execution"]["configuration"],
        run_id=run_id,
    ) as run:
        figure, axes = plt.subplots(2, 1, figsize=(12, 8.5), layout="constrained")
        for index, (ax, label) in enumerate(zip(axes, CONDITIONS, strict=True)):
            cfg = numbers["configurations"][label]
            row = numbers["metrics"][label]
            start, end = cfg["measurement_ms"]
            with np.load(compute.file(label, "events.npz"), allow_pickle=False) as z:
                for pop, color in (("E", theme.INK_BLACK), ("I", theme.DEEP_RED)):
                    times, cells = z[pop + "_time_ms"], z[pop + "_cell"]
                    mask = (times >= start) & (times < end)
                    ax.scatter(
                        times[mask] - start,
                        cells[mask] + (0 if pop == "E" else cfg["n_e"]),
                        s=0.25,
                        c=color,
                        linewidths=0,
                        rasterized=True,
                    )
            ax.axhline(cfg["n_e"] - 0.5, color=theme.MUTED, lw=0.5)
            name = "AI-like" if label == "AI" else "PING"
            ax.set(
                xlim=(0, end - start),
                ylim=(-0.5, cfg["n_e"] + cfg["n_i"] - 0.5),
                xlabel="Time after burn-in (ms)",
                ylabel="Neuron index",
                title=f"{chr(65 + index)}  {name} · E rhythmicity {row['E_rhythmicity']:.3f} · I rhythmicity {row['I_rhythmicity']:.3f}",
            )
            ax.spines[["top", "right"]].set_visible(False)
        for suffix in ("png", "svg", "pdf"):
            figure.savefig(run.export / f"state-rasters.{suffix}", dpi=180)
        plt.close(figure)
        shutil.copyfile(compute.file("circuit.svg"), run.export / "circuit.svg")
        write_json_atomic(
            run.export / "numbers.json", {**numbers, "figure": "state-rasters.svg"}
        )
        run.record["execution"]["elapsed_s"] = perf_counter() - started
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    present(**vars(p.parse_args()))
