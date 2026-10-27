"""Render one unwrapped relative-phase panel from a validated example.

The plotted trajectory is unchanged scientific evidence. An uncoupled control
is retained as numerical calibration; no sweep figures are produced.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import theme
from experiments.exp153 import recipe
from experiments.exp153.compute import ancestry
from pingstore.library import load_json, stage_run, write_json_atomic


def phase_trace_figure(trace, selection):
    """Single unwrapped phase panel, with the physical coupling-onset time."""
    fig, axis = plt.subplots(figsize=(11, 3.3), layout="constrained")
    axis.plot(
        trace["time_ms"], trace["relative_phase_cycles"], color=theme.INK_BLACK, lw=1.2
    )
    axis.axvline(selection["onset_ms"], color=theme.DEEP_RED, ls="--", lw=1.2)
    axis.set(
        xlim=selection["display_window_ms"],
        xlabel="Time (ms)",
        ylabel="Relative phase change\nA−B (cycles)",
    )
    return fig


def present(*, source, run_id=None):
    ancestors = ancestry(REPO, source)
    analysed = ancestors[source]
    if (
        analysed.record["experiment"] != recipe.SLUG
        or analysed.record["stage"] != "analyse"
    ):
        raise ValueError("requires an exp153 analysis")
    computed = ancestors[analysed.record["inputs"]["compute"]["run_id"]]
    cfg = analysed.record["execution"]["configuration"]
    numbers = load_json(analysed.file("results.json"))
    if (
        numbers["configuration"] != cfg
        or computed.record["execution"]["configuration"] != cfg
    ):
        raise ValueError("presentation lineage settings differ")
    if cfg["protocol"] != "phase-trace":
        raise ValueError("requires the phase-trace example")
    theme.apply()
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs=dict(analyse=analysed, compute=computed),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        figure = phase_trace_figure(
            load_json(analysed.file("phase-trace.json")), numbers["phase_trace"]
        )
        for suffix in ("png", "svg", "pdf"):
            figure.savefig(run.export / f"phase-trace.{suffix}")
        plt.close(figure)
        for name in ("phase-trace.json", "control-trace.json"):
            shutil.copyfile(analysed.file(name), run.export / name)
        shutil.copyfile(
            computed.file(numbers["coupled"]["label"] + "-model.json"),
            run.export / "meanfield-model.json",
        )
        shutil.copyfile(computed.file("circuit.svg"), run.export / "circuit.svg")
        write_json_atomic(run.export / "numbers.json", numbers)
        for item in ancestors.values():
            item.check_unchanged()
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    present(**vars(parser.parse_args()))
