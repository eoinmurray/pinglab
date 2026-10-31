"""Render one cue-switch trajectory from explicitly pinned native evidence."""

import argparse
import shutil
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp161 import recipe
from experiments.exp161.analyse import read_trajectory
from pingstore.library import load_json, source_run, stage_run, write_json_atomic


def competition_figure(sol, cfg, metrics, diag):
    t = sol.time_ms
    s = sol.state
    fig, axs = plt.subplots(3, 1, figsize=(13, 9), layout="constrained")
    colors = dict(A="#168047", B="#5367c4", C="#222222")
    for c in "ABC":
        axs[0].plot(t, s[c + "_E.rate"], color=colors[c], lw=0.85, label=c + " E")
    axs[0].set(ylabel="population rate (Hz)", title="A. Competing senders and receiver")
    axs[0].legend(ncol=3, loc="upper right")
    for c in "AB":
        axs[1].plot(
            t,
            diag["phases"][c] - diag["phases"]["C"],
            color=colors[c],
            label=c + " − C",
            lw=1.0,
        )
    axs[1].set(
        ylabel="phase difference (cycles)",
        title="B. Flat trace: locking; drifting trace: phase slips",
    )
    axs[1].legend(ncol=2)
    left = cfg["switch_ms"] - cfg["timing"]["closeup_before_ms"]
    right = cfg["switch_ms"] + cfg["timing"]["closeup_after_ms"]
    v = (t >= left) & (t <= right)
    gi = s["C_I_to_E.conductance"]
    late = t > cfg["timing"]["reference_start_ms"]
    threshold = gi[late].min() + cfg["low_inhibition_fraction"] * np.ptp(gi[late])
    axs[2].fill_between(
        t[v],
        0,
        1,
        where=gi[v] <= threshold,
        transform=axs[2].get_xaxis_transform(),
        color="#e5bc48",
        alpha=0.25,
        label="C low inhibition (proxy)",
    )
    scale = max(np.max(s[c + "_E_to_C_E.conductance"][late]) for c in "AB")
    for c in "AB":
        axs[2].plot(
            t[v],
            s[c + "_E_to_C_E.conductance"][v] / scale,
            color=colors[c],
            label=c + " arriving excitation",
        )
    axs[2].plot(t[v], gi[v] / np.max(gi[late]), color="#bc2942", label="C inhibition")
    axs[2].set(
        xlim=(left, right),
        ylabel="conductance / peak",
        xlabel="time (ms)",
        title="C. Cue-switch close-up; excitation shares one scale",
    )
    axs[2].legend(ncol=2, fontsize=8, loc="upper right")
    for ax in axs:
        ax.axvline(cfg["switch_ms"], color="#777777", ls="--", lw=1)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs[:2]:
        ax.set_xlim(0, cfg["duration_ms"])
        ax.set_xlabel("time (ms)")
        for a, b in cfg["windows_ms"]:
            ax.axvspan(a, b, color="#888888", alpha=0.07)
    text = " | ".join(
        f"cue {r['cued']}: PLV A {r['senders']['A']['plv']:.3f}, B {r['senders']['B']['plv']:.3f}"
        for r in metrics["epochs"]
    )
    fig.suptitle(
        f"Mean-field competition | equal sensory 24 Hz/channel | constant cue {cfg['cue_hz']:g} Hz/channel, A → B at 1200 ms\ncoupling {cfg['coupling_us']:g} µS per pathway | C background {cfg['background_hz']:g} Hz/channel\n{text}",
        fontsize=11,
    )
    return fig


def present(*, source, run_id=None):
    started = perf_counter()
    analysed = source_run(
        REPO / ".pingstore", source, experiment=recipe.SLUG, stage="analyse"
    )
    pin = analysed.record["inputs"]["compute"]
    computed = source_run(
        REPO / ".pingstore",
        pin["run_id"],
        reference=pin,
        experiment=recipe.SLUG,
        stage="compute",
    )
    cfg = analysed.record["execution"]["configuration"]
    if cfg != computed.record["execution"]["configuration"]:
        raise ValueError("lineage configuration differs")
    sol = read_trajectory(computed, cfg)
    results = load_json(analysed.file("results.json"))
    with np.load(analysed.file("phase.npz"), allow_pickle=False) as d:
        np.testing.assert_array_equal(d["time_ms"], sol.time_ms)
        diag = dict(phases={c: d[c + "_phase"] for c in "ABC"})
    fig = competition_figure(sol, cfg, results["metrics"], diag)
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs=dict(analyse=analysed, compute=computed),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        for ext in ("png", "svg", "pdf"):
            fig.savefig(run.export / ("competition." + ext), dpi=170)
        plt.close(fig)
        shutil.copyfile(computed.file("circuit.svg"), run.export / "circuit.svg")
        write_json_atomic(run.export / "numbers.json", dict(results, configuration=cfg))
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
        analysed.check_unchanged()
        computed.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    present(**vars(p.parse_args()))
