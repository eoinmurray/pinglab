# ruff: noqa: E402  # Set repository imports and the headless plotting backend first.
"""Separate fixed-mean contrast and receiver-background sweeps; native exp172."""

import argparse
import copy
import csv
import hashlib
import json
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
import torch
from pingstore.library import source_run
from snnlab.sim import NetworkSnapshot

from experiments import theme
from experiments.exp172.analyse import cycle_selection_metrics
from experiments.exp172.compute import execute
from experiments.exp172.recipe import inputs

PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
SOURCE = dict(
    run_id="exp172-r132-compute",
    payload_digest="sha256:704ffbe472523276a21c7bd8fda1edc89d2e9a9c1ef496e10c00c230fd16e96a",
)
TRIAL = "seed185-swap0-cue3"
CONTRAST = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.5]
BACKGROUND = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0]
COLORS = dict(A=theme.DEEP_RED, B="#2457c5", C="#23844f")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest_array(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def plan(base):
    cases = [
        dict(
            id="baseline",
            variable="baseline",
            value=1.0,
            configuration=copy.deepcopy(base),
        )
    ]
    for value in CONTRAST:
        if value == 1:
            continue
        cfg = copy.deepcopy(base)
        cfg["spatial_baseline_hz"] = base["spatial_mean_hz"] - value * (
            base["spatial_mean_hz"] - base["spatial_baseline_hz"]
        )
        cases.append(
            dict(
                id=f"contrast-{value:g}",
                variable="contrast",
                value=value,
                configuration=cfg,
            )
        )
    for value in BACKGROUND:
        if value == base["background_hz"]:
            continue
        cfg = copy.deepcopy(base)
        cfg["background_hz"] = value
        cases.append(
            dict(
                id=f"background-{value:g}",
                variable="background",
                value=value,
                configuration=cfg,
            )
        )
    return cases


def compute():
    started = perf_counter()
    torch.set_num_threads(1)
    source = source_run(
        REPO / ".pingstore",
        SOURCE["run_id"],
        experiment="exp172",
        stage="compute",
        reference=SOURCE,
    )
    base = json.loads(source.file(TRIAL, "configuration.json").read_text())
    original_item = base.pop("trial")
    base.pop("input_seed_sequences", None)
    parameters = NetworkSnapshot.read(
        source.file(TRIAL, "start/snapshot.json")
    ).load_parameters()
    with np.load(source.file(TRIAL, "spikes.npz")) as z:
        original = dict(z)
    base_arrays, base_rates = inputs(
        base, original_item["seed"], original_item["cue_hz"], False
    )
    cases = plan(base)
    assert len(cases) == len(CONTRAST) + len(BACKGROUND) - 1 == 17
    for case in cases:
        arrays, rates = inputs(
            case["configuration"], original_item["seed"], original_item["cue_hz"], False
        )
        for c in "AB":
            assert np.isclose(
                rates["X_" + c].mean(), base["spatial_mean_hz"], atol=1e-12
            )
            assert (rates["X_" + c] >= 0).all()
        unchanged = (
            ["X_C", "U"] if case["variable"] == "contrast" else ["X_A", "X_B", "U"]
        )
        for k in unchanged:
            np.testing.assert_array_equal(arrays[k], base_arrays[k])
    if DATA.exists():
        shutil.rmtree(DATA)
    (DATA / "trials").mkdir(parents=True)
    dump(
        DATA / "plan.json",
        dict(
            source=SOURCE,
            source_trial=TRIAL,
            base_configuration=base,
            contrasts=CONTRAST,
            backgrounds_hz=BACKGROUND,
            cases=cases,
            seed=original_item["seed"],
            expected_mean_sensory_hz=base["spatial_mean_hz"],
            common_random_numbers=True,
            contrast_definition="r_c(x) = mean + c*(r_baseline(x)-mean); baseline(x) has spatial baseline 22 and spatial mean 26 Hz/channel. Same contrast applied to both A and B.",
            code_sha256={
                str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    REPO / "experiments/exp172/recipe.py",
                    REPO / "experiments/exp172/analyse.py",
                    REPO / "experiments/exp172/compute.py",
                ]
            },
        ),
    )
    records = []
    baseline_spikes = None
    for i, case in enumerate(cases):
        item = dict(original_item, id=case["id"])
        record = execute(
            case["configuration"], parameters, item, DATA / "trials", diagram=i == 0
        )
        directory = DATA / "trials" / case["id"]
        with np.load(directory / "spikes.npz") as z:
            spikes = dict(z)
        actual_params = NetworkSnapshot.read(
            directory / "start/snapshot.json"
        ).load_parameters()
        assert actual_params.keys() == parameters.keys()
        for k in parameters:
            np.testing.assert_array_equal(actual_params[k], parameters[k])
        arrays, rates = inputs(
            case["configuration"], item["seed"], item["cue_hz"], False
        )
        if case["variable"] == "baseline":
            for k in original:
                np.testing.assert_array_equal(spikes[k], original[k])
            baseline_spikes = spikes
        elif case["variable"] == "background":
            for k in ["A_E", "A_I", "B_E", "B_I"]:
                np.testing.assert_array_equal(spikes[k], baseline_spikes[k])
        record.update(
            variable=case["variable"],
            value=case["value"],
            input_sha256={k: digest_array(a) for k, a in arrays.items()},
            expected_input_rates_hz={k: float(v.mean()) for k, v in rates.items()},
            realised_input_rates_hz={
                k: float(a.mean() * 1000 / base["dt_ms"]) for k, a in arrays.items()
            },
            weights_equal_to_source=True,
        )
        records.append(record)
        dump(
            DATA / "compute.json",
            dict(
                source=SOURCE,
                records=records,
                completed=len(records),
                expected=len(cases),
                simulation_seconds=sum(r["simulation_seconds"] for r in records),
                elapsed_seconds=perf_counter() - started,
            ),
        )
        print(
            f"Sweep progress {len(records)}/{len(cases)}; elapsed {perf_counter() - started:.1f} s",
            flush=True,
        )
    analyse()


def analyse():
    started = perf_counter()
    definition = json.loads((DATA / "plan.json").read_text())
    records = json.loads((DATA / "compute.json").read_text())
    assert records["completed"] == records["expected"] == 17
    rows = []
    all_metrics = {}
    for case, record in zip(definition["cases"], records["records"], strict=True):
        assert record["id"] == case["id"]
        cfg = case["configuration"]
        with np.load(DATA / "trials" / case["id"] / "spikes.npz") as z:
            spikes = dict(z)
        measured = cycle_selection_metrics(spikes, cfg, False, sustained_cycles=3)
        assert measured["decoder_version"] == "trough-to-trough/v1"
        all_metrics[case["id"]] = measured
        for epoch in measured["epochs"]:
            lo, hi = [round(v / cfg["dt_ms"]) for v in epoch["window_ms"]]
            count = len(epoch["cycles"])
            resolved = sum(c["dominant"] is not None for c in epoch["cycles"])
            states = [c["dominant"] for c in epoch["cycles"]]
            pairs = [
                (a, b)
                for a, b in zip(states, states[1:], strict=False)
                if a is not None and b is not None
            ]
            rows.append(
                dict(
                    id=case["id"],
                    variable=case["variable"],
                    value=case["value"],
                    attended=epoch["attended"],
                    S=epoch["switching_success"],
                    F=epoch["flicker_fraction"],
                    latency_cycles=epoch["switching_latency"]["cycles"],
                    C_E_hz=float(spikes["C_E"][lo:hi].mean() * 1000 / cfg["dt_ms"]),
                    C_I_hz=float(spikes["C_I"][lo:hi].mean() * 1000 / cfg["dt_ms"]),
                    cycles=count,
                    resolved_cycles=resolved,
                    resolved_fraction=resolved / count if count else None,
                    dominance_changes=sum(a != b for a, b in pairs),
                    eligible_pairs=len(pairs),
                )
            )
    baseline = {r["attended"]: r for r in rows if r["id"] == "baseline"}
    np.testing.assert_allclose(
        [
            baseline["A"]["S"],
            baseline["A"]["F"],
            baseline["B"]["S"],
            baseline["B"]["F"],
        ],
        [33 / 39, 4 / 38, 32 / 38, 8 / 37],
        atol=1e-10,
    )
    assert [baseline[a]["latency_cycles"] for a in "AB"] == [3, 1]
    with (DATA / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    theme.apply()
    sweep_rows = {}
    for variable, grid, xlabel in [
        ("contrast", CONTRAST, "Sensory contrast (× baseline; mean 26 Hz/channel)"),
        ("background", BACKGROUND, "C independent background rate (Hz/channel)"),
    ]:
        selected = []
        for value in grid:
            identity = (
                "baseline"
                if (variable == "contrast" and value == 1)
                or (variable == "background" and value == 28)
                else f"{variable}-{value:g}"
            )
            selected.extend(dict(r, control=value) for r in rows if r["id"] == identity)
        sweep_rows[variable] = selected
        fig, axes = plt.subplots(
            2, 2, figsize=(12, 8), sharex="col", layout="constrained"
        )
        for col, att in enumerate("AB"):
            subset = [r for r in selected if r["attended"] == att]
            x = [r["control"] for r in subset]
            ax = axes[0, col]
            speed = ax.twinx()
            speed.grid(False)

            def nan(v):
                return np.nan if v is None else v

            for key, ls, label in [("S", "-", "Selection S"), ("F", "--", "Flicker F")]:
                ax.plot(
                    x,
                    [nan(r[key]) for r in subset],
                    color=COLORS[att],
                    ls=ls,
                    marker="o",
                    markersize=4,
                    label=label,
                )
            speed.plot(
                x,
                [nan(r["latency_cycles"]) for r in subset],
                color=COLORS["C"],
                ls=":",
                marker="s",
                markersize=4,
                label="Latency",
            )
            ax.set(title=f"Attend {att}", ylabel="S / F", ylim=(-0.03, 1.03))
            speed.set_ylabel("Latency (PING cycles)", color=COLORS["C"])
            speed.tick_params(axis="y", labelcolor=COLORS["C"])
            speed.set_ylim(0.5, max([r["latency_cycles"] or 1 for r in subset]) + 1)
            handles, labels = ax.get_legend_handles_labels()
            hh, ll = speed.get_legend_handles_labels()
            ax.legend(
                handles + hh,
                labels + ll,
                loc="lower left",
                bbox_to_anchor=(0, 1.08),
                ncol=3,
                fontsize=theme.SIZE_ANNOTATION,
            )
            rate = axes[1, col]
            rate.plot(
                x,
                [r["C_E_hz"] for r in subset],
                color=COLORS["C"],
                marker="o",
                markersize=4,
                label="C E",
            )
            rate.plot(
                x,
                [r["C_I_hz"] for r in subset],
                color=COLORS["C"],
                ls="--",
                marker="^",
                markersize=4,
                label="C I",
            )
            rate.set(xlabel=xlabel, ylabel="C firing rate (Hz)", ylim=(0, None))
            rate.legend(loc="best", fontsize=theme.SIZE_ANNOTATION)
            reference = 1 if variable == "contrast" else 28
            for panel in axes[:, col]:
                panel.axvline(reference, color=theme.GREY_MID, ls=":", lw=0.8)
        theme.label_panels(list(axes.flat))
        for ext in ["png", "svg"]:
            fig.savefig(DATA / f"{variable}.{ext}", dpi=240)
        plt.close(fig)
    dump(
        DATA / "results.json",
        dict(
            source=SOURCE,
            rows=rows,
            sweeps=sweep_rows,
            cycle_metrics=all_metrics,
            analysis_code_sha256=hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            simulation_seconds=records["simulation_seconds"],
            compute_seconds=records["elapsed_seconds"],
            analysis_seconds=perf_counter() - started,
            limitations="Separate one-dimensional screens on seed 185, using common random input numbers and exact fixed weights. Expected sensory mean is fixed; realised spike counts fluctuate. No held-out or multi-seed claim; no joint combination tested.",
        ),
    )
    lines = [
        "# exp172 contrast and C-background sweeps",
        "",
        "1. Two separate sweeps; cue 3 Hz/channel, fixed seed 185 and exact exp172-r132 representative-trial weights. Baseline is freshly rerun and matches all six original spike arrays exactly.",
        "2. Contrast c rescales the spatial modulation about its fixed mean: r_c(x) = 26 + c × [r_1(x) − 26] Hz/channel, where x is channel position and r_1 is the original spatial rate profile. Both A and B change together. c=0 removes spatial contrast; c=1 is baseline. Width and expected mean stay fixed, while the flat component becomes 26 − 4c. At c=6.5 that component reaches zero.",
        "3. C background changes only the independent X_C input rate. All four sender spike arrays are checked identical across that sweep. Synaptic weights are exactly identical in every run.",
        "4. Metrics use the corrected trough-to-trough decoder. S/F and C firing rates use the same settled A/B windows; latency retains the three-consecutive-cycle criterion from cue onset. Missing F or latency stays missing rather than becoming zero. Raw cycle counts and resolution are in metrics.csv.",
        "5. The green right axis is latency in cycles; coloured left-axis lines are S (solid) and F (dashed). Lower panels show C E/I rates. Vertical dotted lines identify the current parameters. Inputs use paired random-number streams, not identical realised spike counts when rates change. No seed averaging or uncertainty intervals.",
        "",
        "![Contrast](contrast.png)",
        "",
        "![C background](background.png)",
        "",
        "| Sweep | Control | Attend | S | F | Latency (cycles) | C E (Hz) | C I (Hz) | Resolved/total |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(v):
        return "not observed" if v is None else f"{v:.3g}"

    for variable, selected in sweep_rows.items():
        for r in selected:
            lines.append(
                f"| {variable} | {r['control']:g} | {r['attended']} | {fmt(r['S'])} | {fmt(r['F'])} | {fmt(r['latency_cycles'])} | {r['C_E_hz']:.2f} | {r['C_I_hz']:.2f} | {r['resolved_cycles']}/{r['cycles']} |"
            )
    lines += [
        "",
        f"17 fresh simulations. Summed simulation time {records['simulation_seconds']:.2f} s; compute wall time {records['elapsed_seconds']:.2f} s; analysis/rendering {perf_counter() - started:.2f} s.",
        "",
        "These measurements are a first screen on one tuned seed. Low F alone can mean stable incorrect selection or insufficient receiver activity. No settings have been copied into the formal experiment.",
        "",
        "## Interpretation",
        "",
        "1. No non-baseline contrast value improves S, F and latency simultaneously in both attention periods. Increased contrast generally weakens attended dominance; a lower F in one period can accompany worse selection or slower acquisition.",
        "2. Reducing C background below 28 Hz/channel does not improve the full set of objectives. At 24 Hz/channel the B metrics equal baseline but A worsens; stronger reductions substantially weaken selection.",
        "3. Raising C background from 28 to 32 Hz/channel is the best tested direction: A S 0.846→0.900 and F 0.105→0.103; B S 0.842→0.872 and F 0.216→0.184. Acquisition/switch latency stays at 3/1 cycles. C E firing rises from 25.5/27.8 to 27.5/28.6 Hz in A/B periods, so this is not silence. All retained cycles at 32 are resolved. The tiny A F decrease is denominator-only: both cases have four changes, over 38 versus 39 eligible pairs. B has eight changes over 37 pairs at baseline versus seven over 38 at 32, but the 32-Hz settled window starts in the wrong state. These short-window counts are not evidence of a reliable reduction in event frequency.",
        "4. No tested condition has zero F in both periods. Continuous background reduction changes C’s operating regime; a brief event-specific background interruption is not equivalent to reducing the drive throughout the trial. These plots identify candidate settings, not an isolated mechanism or a validated optimum.",
    ]
    (DATA / "results.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(
                completed=17,
                baseline=baseline,
                analysis_seconds=perf_counter() - started,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    if args.analyse_only:
        analyse()
    else:
        compute()
