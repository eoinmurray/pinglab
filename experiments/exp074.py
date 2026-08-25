"""Experiment 074 — the first end-to-end snnlang demo.

The runner authors a PING circuit with the Python snnlang API, compiles it to
a data-only bundle, sends an explicit Poisson spike tensor through tools/snnsim's
CLI bundle route, and publishes the circuit diagram plus aligned input/E/I
rasters.  This is an integration demonstration, not a scientific comparison.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# snnlang is an authoring/compiler library, not the simulator.  This intentional
# import produces a data-only bundle; tools/snnsim itself remains a subprocess.
from tools import snnlang as snn  # noqa: E402, TID251

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp074"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

DT_MS = 0.1
T_MS = 200.0
N_BATCH = 4
N_INPUT = 784
N_E = 256
N_I = 64
N_CLASSES = 10
INPUT_RATE_HZ = 100.0
SEED = 74
DISPLAY_TRIAL = 0

SCALE = {
    "dt_ms": DT_MS,
    "t_ms": T_MS,
    "n_batch": N_BATCH,
    "n_input": N_INPUT,
    "n_e": N_E,
    "n_i": N_I,
    "input_rate_hz": INPUT_RATE_HZ,
    "seed": SEED,
}


def author_network() -> snn.Bundle:
    """Define the graph in Python; no simulator implementation leaks in here."""
    net = snn.Network("snnlang_ping_demo", dt=DT_MS * snn.ms)
    spikes = net.input(
        "spike_input",
        shape=("time", "batch", N_INPUT),
        signal_type="spikes",
        unit="spike",
    )
    cell = snn.components.ping(
        net,
        name="sensory_ping",
        n_e=N_E,
        n_i=N_I,
        source=spikes,
    )
    logits = snn.readouts.MeanVoltage(
        source=cell.E.spikes,
        classes=N_CLASSES,
        name="classifier",
        tau=2 * snn.ms,
        weight=snn.Normal(5.1, 3.8),
    )
    net.output("class_logits", logits)
    net.expose(cell.E.spikes, cell.I.spikes, name="cell")
    return snn.compile(net, target="tools/snnsim")


def make_input(path: Path) -> np.ndarray:
    """Create the exact spike tensor consumed by the simulator."""
    rng = np.random.default_rng(SEED)
    n_steps = round(T_MS / DT_MS)
    p_step = INPUT_RATE_HZ * DT_MS / 1000.0
    spikes = (
        rng.random((n_steps, N_BATCH, N_INPUT), dtype=np.float32) < p_step
    ).astype(np.uint8)
    np.savez_compressed(path, input_spikes=spikes)
    return spikes


def run_simulator(bundle_dir: Path, input_path: Path, sim_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO / "tools/snnsim/tool.py"),
        "sim",
        "--bundle",
        str(bundle_dir),
        "--input-file",
        str(input_path),
        "--t-ms",
        str(T_MS),
        "--n-batch",
        str(N_BATCH),
        "--input-rate",
        str(INPUT_RATE_HZ),
        "--seed",
        str(SEED),
        "--outputs",
        "rasters",
        "--out-dir",
        str(sim_dir),
        "--wipe-dir",
    ]
    print("[simulate]", " ".join(cmd))
    env = dict(os.environ)
    env.setdefault("PINGLAB_NO_COMPILE", "1")
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def _trial_events(
    rasters: np.lib.npyio.NpzFile, prefix: str, trial: int
) -> tuple[np.ndarray, np.ndarray]:
    mask = rasters[f"{prefix}_trial"] == trial
    return rasters[f"{prefix}_t"][mask], rasters[f"{prefix}_cell"][mask]


def plot_rasters(
    input_spikes: np.ndarray, raster_path: Path, out_path: Path
) -> dict[str, int]:
    """Plot the exact input and resulting E/I spikes for one aligned trial."""
    theme.apply()
    with np.load(raster_path) as rasters:
        dt_ms = float(rasters["dt"])
        e_t, e_cell = _trial_events(rasters, "e", DISPLAY_TRIAL)
        i_t, i_cell = _trial_events(rasters, "i", DISPLAY_TRIAL)

    input_t, input_cell = np.nonzero(input_spikes[:, DISPLAY_TRIAL, :])
    panels = [
        ("INPUT", input_t, input_cell, theme.GREY_MID, N_INPUT),
        ("E", e_t, e_cell, theme.INK_BLACK, N_E),
        ("I", i_t, i_cell, theme.DEEP_RED, N_I),
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.5, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.35, 0.9], "hspace": 0.16},
    )
    for ax, (label, times, cells, colour, size) in zip(axes, panels):
        ax.scatter(
            times * dt_ms,
            cells,
            s=2.0,
            marker=".",
            linewidths=0,
            color=colour,
            alpha=0.7,
            rasterized=True,
        )
        ax.set_ylim(-1, size)
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
        ax.text(
            0.995,
            0.92,
            f"{size} cells · {len(times):,} spikes",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=theme.SIZE_ANNOTATION,
            color=colour,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlim(0, T_MS)
    axes[-1].set_xlabel("time (ms)")
    fig.align_ylabels(axes)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "input": int(len(input_t)),
        "e": int(len(e_t)),
        "i": int(len(i_t)),
    }


def main() -> None:
    meta = parse_meta(sys.argv)
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id}")

    with published_run(
        SLUG,
        run_id,
        scale=SCALE,
        plot_only=meta.plot_only,
    ) as (artifacts, figures):
        bundle_dir = artifacts / "network.bundle"
        input_path = artifacts / "input_spikes.npz"
        sim_dir = artifacts / "simulation"

        bundle = author_network()
        bundle.write(bundle_dir, visualise=True)
        input_spikes = make_input(input_path)
        run_simulator(bundle_dir, input_path, sim_dir)

        shutil.copytree(bundle_dir, figures / "network.bundle")
        shutil.copy2(bundle_dir / "reports/circuit.svg", figures / "network.svg")
        shutil.copy2(input_path, figures / "input_spikes.npz")
        shutil.copy2(sim_dir / "rasters.npz", figures / "rasters.npz")

        event_counts = plot_rasters(
            input_spikes, sim_dir / "rasters.npz", figures / "rasters.png"
        )
        metrics = json.loads((sim_dir / "metrics.json").read_text())
        graph = bundle.graph
        total_input = int(input_spikes.sum())
        realised_input_rate = total_input / (
            N_BATCH * N_INPUT * (T_MS / 1000.0)
        )
        payload = {
            "purpose": "end-to-end integration demonstration",
            "graph": {
                "name": graph["name"],
                "digest": bundle.manifest["graph_digest"],
                "digest_short": bundle.manifest["graph_digest"][:19],
                "populations": len(graph["populations"]),
                "projections": len(graph["projections"]),
                "operations": len(graph["operations"]),
                "parameter_tensors": len(graph["parameters"]),
            },
            "config": SCALE,
            "input": {
                "shape": list(input_spikes.shape),
                "shape_text": " × ".join(str(n) for n in input_spikes.shape),
                "total_spikes": total_input,
                "realised_rate_hz": realised_input_rate,
            },
            "output": {
                "rate_e_hz": metrics["rate_e_hz"],
                "rate_i_hz": metrics["rate_i_hz"],
                "display_trial": DISPLAY_TRIAL,
                "display_trial_spikes": event_counts,
            },
        }
        write_numbers(
            figures,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload=payload,
        )


if __name__ == "__main__":
    main()
