"""Runner for exp001 — EIF neuron and EIF network.

The neuron tool emits data only (CSVs + metrics); the figures are rendered *here*
from those CSVs and staged into artifacts/data/exp001/ alongside numbers.json. Plotting
is the experiment's job, not the tool's.
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import sh

from helpers import style as _style  # shared figure style (HOUSESTYLE H10-H16); applies rcParams on import

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "neuron" / "tool.py"
TEMP = ROOT / "temp" / "neuron"
ARTIFACTS = ROOT / "artifacts" / "data" / "exp001"

COMMANDS = ("eif", "enet")


def run_tool(*args: str) -> None:
    sh.uv.run("python", str(TOOL), *args, _fg=True)


def read_csv(path: Path) -> dict[str, list[float]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {col: [float(r[col]) for r in rows] for col in rows[0]}


def plot_trace(command: str, dst: Path) -> None:
    # A single voltage trace: line plot, black, no title (the caption carries it). SVG (H10).
    d = read_csv(TEMP / command / f"{command}.csv")
    fig, ax = plt.subplots()
    ax.plot(d["time_ms"], d["voltage_mV"], color=_style.INK)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane potential (mV)")
    fig.savefig(dst)
    plt.close(fig)


def plot_network(command: str, dst: Path) -> None:
    # Raster (dense scatter) over a mean-current panel: both black, band grey, no titles.
    # Dense scatter, so PNG (H10). Taller than the default to fit the two stacked panels.
    spikes = read_csv(TEMP / command / f"{command}.csv")
    cur = read_csv(TEMP / command / f"{command}_current.csv")
    fig, (ax_r, ax_c) = plt.subplots(
        2, 1, figsize=(6.5, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_r.scatter(spikes["time_ms"], spikes["neuron_id"], s=2, color=_style.INK)
    ax_r.set_ylabel("Neuron index")

    t, mean, std = cur["time_ms"], cur["i_total_mean_nA"], cur["i_total_std_nA"]
    lo = [m - s for m, s in zip(mean, std)]
    hi = [m + s for m, s in zip(mean, std)]
    ax_c.plot(t, mean, color=_style.INK)
    ax_c.fill_between(t, lo, hi, color=_style.BAND)
    ax_c.set_xlabel("Time (ms)")
    ax_c.set_ylabel("Input current (nA)")

    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)


def load_manifest(command: str) -> dict:
    return json.loads((TEMP / command / "manifest.json").read_text())


def collect_numbers() -> dict:
    numbers = {}
    for command in COMMANDS:
        manifest = load_manifest(command)
        config = json.loads((TEMP / command / "config.json").read_text())
        output = json.loads((TEMP / command / "output.json").read_text())
        numbers[command] = {
            "config": config,
            **{f: output[f] for f in manifest["headline_metrics"]},
        }
    return numbers


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for command in COMMANDS:
        run_tool(command)
    plot_trace("eif", ARTIFACTS / "eif.svg")
    plot_network("enet", ARTIFACTS / "enet.png")
    print(f"rendered eif.svg, enet.png -> {ARTIFACTS}")

    numbers_path = ARTIFACTS / "numbers.json"
    numbers_path.write_text(json.dumps(collect_numbers(), indent=2) + "\n")
    print(f"wrote {numbers_path}")


if __name__ == "__main__":
    main()
