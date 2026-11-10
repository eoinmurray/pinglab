# ruff: noqa: E402  # Set repository imports and the headless plotting backend first.
"""Present the best joint-selection candidate already measured in both sweeps."""

import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from snnlab.sim import NetworkSnapshot

from experiments import theme
from experiments.exp172.analyse import compare_metrics, cycle_selection_metrics

DATA = Path(__file__).resolve().parent / "data"
COLORS = dict(A=theme.DEEP_RED, B="#2457c5", C="#23844f")


def main():
    started = perf_counter()
    results = json.loads((DATA / "results.json").read_text())
    by_id = {}
    for r in results["rows"]:
        by_id.setdefault(r["id"], {})[r["attended"]] = r

    # Maximise worst-period S, then mean S; use mean F to break accuracy ties.
    # This criterion is explicit because a lower B F alone sacrifices accuracy.
    def key(item):
        return (
            min(item[1][a]["S"] for a in "AB"),
            sum(item[1][a]["S"] for a in "AB") / 2,
            -sum(item[1][a]["F"] for a in "AB") / 2,
        )

    identity, chosen = max(by_id.items(), key=key)
    assert identity == "background-32"
    for a in "AB":
        assert chosen[a]["S"] == max(
            r["S"] for r in results["rows"] if r["attended"] == a
        )
    directory = DATA / "trials" / identity
    cfg = json.loads((directory / "configuration.json").read_text())
    with np.load(directory / "spikes.npz") as z:
        spikes = dict(z)
    measured = cycle_selection_metrics(spikes, cfg, False, sustained_cycles=3)
    compare_metrics(measured, results["cycle_metrics"][identity])
    current = NetworkSnapshot.read(directory / "start/snapshot.json").load_parameters()
    baseline = NetworkSnapshot.read(
        DATA / "trials/baseline/start/snapshot.json"
    ).load_parameters()
    for k in baseline:
        np.testing.assert_array_equal(current[k], baseline[k])
    assert (
        cfg["spatial_baseline_hz"] == 22
        and cfg["spatial_mean_hz"] == 26
        and cfg["background_hz"] == 32
    )
    baseline_rows = by_id["baseline"]
    theme.apply()
    fig, axes = plt.subplots(
        3, 1, figsize=(13, 9.2), sharex=True, sharey=True, layout="constrained"
    )
    dt = cfg["dt_ms"]
    for ax, population in zip(axes, "ABC", strict=True):
        ne = spikes[population + "_E"].shape[1]
        ni = spikes[population + "_I"].shape[1]
        for pool, offset in [("E", 0), ("I", ne)]:
            ii, jj = np.nonzero(spikes[population + "_" + pool])
            ax.scatter(
                ii * dt,
                jj + offset + 1,
                color=COLORS[population],
                s=0.8,
                marker=".",
                linewidths=0,
                rasterized=True,
            )
        ax.axhline(ne + 0.5, color=theme.GREY_MID, lw=0.7)
        ax.axvline(cfg["switch_ms"], color=theme.GREY_MID, ls="--", lw=1)
        ax.text(
            0.99,
            ne + ni / 2,
            "I",
            ha="right",
            va="center",
            color=COLORS[population],
            transform=ax.get_yaxis_transform(),
            fontsize=theme.SIZE_ANNOTATION,
        )
        ax.set(
            ylabel="Neuron index",
            ylim=(0.5, ne + ni + 0.5),
            yticks=[1, ne / 2, ne, ne + ni],
            title=population + (" receiver" if population == "C" else " sender"),
        )
        if population == "C":
            for pattern, centre in zip("AB", cfg["centres"], strict=True):
                ax.axhline(centre * ne + 1, color=theme.GREY_LIGHT, lw=0.6, ls=":")
                ax.text(
                    0.99,
                    centre * ne + 1,
                    "Pattern " + pattern,
                    ha="right",
                    va="bottom",
                    transform=ax.get_yaxis_transform(),
                    color=COLORS[pattern],
                    fontsize=theme.SIZE_ANNOTATION,
                )
            for epoch in measured["epochs"]:
                for cycle in epoch["cycles"]:
                    if (
                        cycle["dominant"] is not None
                        and cycle["dominant"] != epoch["attended"]
                    ):
                        ax.axvspan(
                            cycle["start_ms"],
                            cycle["end_ms"],
                            color=theme.GREY_MID,
                            alpha=0.13,
                            zorder=-1,
                        )
    axes[-1].set(xlabel="Time (ms)", xlim=(0, cfg["duration_ms"]))
    fig.suptitle(
        "Selected candidate · contrast 1× · C background 32 Hz/channel · cue 3 Hz/channel",
        x=0.075,
        ha="left",
        fontsize=theme.SIZE_PANEL,
    )
    theme.label_panels(list(axes))
    for suffix in ["png", "svg"]:
        fig.savefig(DATA / f"candidate.{suffix}", dpi=240)
    plt.close(fig)
    comparison = [
        dict(
            attended=a,
            S=chosen[a]["S"],
            F=chosen[a]["F"],
            latency_cycles=chosen[a]["latency_cycles"],
            baseline_S=baseline_rows[a]["S"],
            baseline_F=baseline_rows[a]["F"],
            baseline_latency_cycles=baseline_rows[a]["latency_cycles"],
        )
        for a in "AB"
    ]
    record = dict(
        candidate=identity,
        selection_rule="Maximise worst-period S, then mean S; break accuracy ties by lower mean F.",
        source=results["source"],
        trial_path=str(directory.relative_to(REPO)),
        contrast_multiplier=1,
        background_hz=32,
        cue_hz=3,
        metrics=measured,
        comparison=comparison,
        simulations_run=0,
        elapsed_seconds=perf_counter() - started,
        hashes={
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                DATA / "results.json",
                directory / "spikes.npz",
                directory / "configuration.json",
            ]
        },
        caveat="Best joint-selection candidate within these separate one-seed sweeps, not a new jointly optimised grid. B F alone is lower at contrast 0.5 but accuracy and latency are worse. No held-out claim.",
    )
    (DATA / "candidate.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Best measured combination",
        "",
        "Contrast 1× (unchanged original spatial profile), C independent background 32 Hz/channel, cue 3 Hz/channel; original fixed weights and seed 185. This exact combination was already simulated as background-32, so presentation requires no new simulation.",
        "",
        "Selection maximises the poorer of the two S values, then mean S, with mean F as an accuracy-tie breaker. The candidate independently has the highest S in both periods and the lowest A F. Lower B F alone is attainable by sacrificing accuracy and speed.",
        "",
        "| Attend | S | F | Latency (cycles) | Baseline S | Baseline F |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in comparison:
        lines.append(
            f"| {r['attended']} | {r['S']:.3f} | {r['F']:.3f} | {r['latency_cycles']} | {r['baseline_S']:.3f} | {r['baseline_F']:.3f} |"
        )
    lines += [
        "",
        "![Candidate raster](candidate.png)",
        "",
        "A red, B blue, C green; inhibitory neurons are above excitatory neurons. The dashed line at 1200 ms marks the cue switch. Pale grey intervals in C mark unattended-dominant complete cycles inside the settled metric windows; they are retained rather than hidden. The first A latency is acquisition; B is switching from A.",
        "",
        "The tiny A F reduction is denominator-only (4 changes/39 pairs versus 4/38); B has 7/38 versus 8/37, with different settled-window initial state. These are modest one-seed changes, not elimination of flicker. Formal exp172 settings are unchanged.",
        "",
        f"Validation/rendering: {record['elapsed_seconds']:.2f} s. New simulations: 0.",
    ]
    (DATA / "candidate.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(
                candidate=identity,
                comparison=comparison,
                seconds=record["elapsed_seconds"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
